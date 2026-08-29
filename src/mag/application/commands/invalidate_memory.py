import dataclasses
import uuid
from datetime import datetime

from src.mag.domain.entities import SemanticMemory
from src.mag.domain.ports import (
    MemoryGraphRepository,
    SemanticMemoryIndex,
    SemanticMemoryRepository,
)
from src.mag.infrastructure._graph_write_safety import best_effort_graph_write
from src.mag.infrastructure._index_write_safety import best_effort_index_write


class InvalidateMemory:
    # Invalidate (#63): old information is no longer true at all, action
    # is to mark the fact stale and exclude it from retrieval, without
    # necessarily replacing it with anything -- a status flip on the SAME
    # value, unlike Update/Refine, which replace fact_value outright.
    def __init__(
        self,
        semantic_memory_repository: SemanticMemoryRepository,
        semantic_memory_index: SemanticMemoryIndex,
        memory_graph_repository: MemoryGraphRepository,
    ) -> None:
        self._repository = semantic_memory_repository
        self._index = semantic_memory_index
        self._graph = memory_graph_repository

    async def execute(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        fact_key: str,
        invalidated_at: datetime | None = None,
    ) -> SemanticMemory:
        existing = await self._repository.find_by_key(user_id, fact_key, tenant_id)
        if existing is None:
            raise ValueError(
                f"no existing fact for user_id={user_id} fact_key={fact_key!r} to invalidate"
            )
        # invalidated_at=None flows through to the repository as "use the
        # database's own now()" -- avoids a clock-skew window against the
        # later valid_until > now() comparison search_by_similarity makes
        # using that same database's clock. The repository hands back
        # whichever value it actually used, so Qdrant/Neo4j stay
        # consistent with Postgres rather than each computing their own.
        actual_invalidated_at = await self._repository.invalidate(
            user_id, fact_key, tenant_id, invalidated_at
        )
        # set_valid_until (unlike upsert()) never touches the stored
        # vector, which matters since `existing` never carries a real
        # embedding (Postgres isn't this system's embedding-bearing read
        # path) -- and never touches archived_at either, closing a race
        # where a concurrent ArchiveMemory call could otherwise be
        # clobbered by a stale snapshot of the field it owns. Best-effort:
        # if this fact's Qdrant point is missing (an earlier write already
        # failed -- DATABASE.md's own documented non-atomic-stores
        # consequence), the Postgres UPDATE above has already committed,
        # so an uncaught exception here would misrepresent a partially-
        # successful, recoverable state as a hard failure.
        await best_effort_index_write(
            self._index.set_valid_until(existing.id, tenant_id, actual_invalidated_at),
            "set valid_until (invalidate)",
        )
        # set_fact_valid_until (not upsert_fact_node) for the same reason
        # as set_valid_until above, one layer further: upsert_fact_node
        # writes BOTH valid_until and archived_at from whatever entity
        # it's given, so building that entity via dataclasses.replace on
        # `existing` (read at the TOP of this method) would ship a stale
        # snapshot of archived_at -- a concurrent ArchiveMemory call
        # landing in between could be silently clobbered back to that
        # stale value. set_fact_valid_until touches only its own
        # property, closing that race for Neo4j the same way
        # set_valid_until already closes it for Qdrant.
        await best_effort_graph_write(
            self._graph.set_fact_valid_until(existing.id, tenant_id, actual_invalidated_at),
            "set fact valid_until (invalidate)",
        )
        return dataclasses.replace(existing, valid_until=actual_invalidated_at)
