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


class ArchiveMemory:
    # Archive (#64): a fact is rarely accessed, action is to move it to
    # cold storage while keeping it available for reference -- excluded
    # from default retrieval (search_by_similarity/search), still
    # reachable via find_by_key. Unlike Invalidate/Update/Refine, the
    # trigger here is access frequency, not new information contradicting
    # or refining anything -- a caller tracks that on its own and invokes
    # this directly; it is deliberately not part of EvolveMemory's
    # content-comparison dispatch.
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
        archived_at: datetime | None = None,
    ) -> SemanticMemory:
        existing = await self._repository.find_by_key(user_id, fact_key, tenant_id)
        if existing is None:
            raise ValueError(
                f"no existing fact for user_id={user_id} fact_key={fact_key!r} to archive"
            )
        archived_at = archived_at or datetime.now(UTC)
        await self._repository.archive(user_id, fact_key, tenant_id, archived_at)
        await self._index.update_status(existing.id, tenant_id, existing.valid_until, archived_at)
        updated = dataclasses.replace(existing, archived_at=archived_at)
        await best_effort_graph_write(
            self._graph.upsert_fact_node(updated, tenant_id), "upsert fact node (archive)"
        )
        return updated
