import dataclasses
import uuid
from datetime import UTC, datetime

from src.mag.domain.entities import SemanticMemory
from src.mag.domain.ports import (
    MemoryGraphRepository,
    SemanticMemoryIndex,
    SemanticMemoryRepository,
)
from src.mag.infrastructure._graph_write_safety import best_effort_graph_write


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
        invalidated_at = invalidated_at or datetime.now(UTC)
        await self._repository.invalidate(user_id, fact_key, tenant_id, invalidated_at)
        # A targeted payload update, not a full re-upsert -- update_status
        # (unlike upsert()) never touches the stored vector, which matters
        # here since `existing` never carries a real embedding (Postgres
        # isn't this system's embedding-bearing read path).
        await self._index.update_status(
            existing.id, tenant_id, invalidated_at, existing.archived_at
        )
        updated = dataclasses.replace(existing, valid_until=invalidated_at)
        await best_effort_graph_write(
            self._graph.upsert_fact_node(updated, tenant_id), "upsert fact node (invalidate)"
        )
        return updated
