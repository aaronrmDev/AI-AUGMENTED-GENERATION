import uuid
from collections.abc import Callable

from src.orchestration.domain import sync_mixer
from src.orchestration.domain.entities import SyncConflict
from src.orchestration.domain.ports import WarmStore


class MagSyncCycle:
    """The Sync Mixer's RAG-vs-MAG batch invalidation trigger: for each
    currently tracked document, reconcile what MAG's warm tier holds
    against RAG's current authoritative content and demote on a real
    conflict (RAG wins) -- MAG's own "state correction," so the next
    lookup falls through to a fresh RAG retrieval rather than serving a
    stale value.

    Talks only to the abstract WarmStore port, the same posture Batch D's
    SyncCycle holds toward FrozenCache -- it has no MAG-specific knowledge
    of what "demote" does underneath (see SemanticMemoryWarmStore.demote
    for that).
    """

    def __init__(self, warm_store: WarmStore) -> None:
        self._warm_store = warm_store

    async def run(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        document_ids: list[uuid.UUID],
        authoritative_content: Callable[[uuid.UUID], str],
    ) -> list[SyncConflict]:
        conflicts: list[SyncConflict] = []
        for document_id in document_ids:
            warm_entry = await self._warm_store.lookup(tenant_id, user_id, document_id)
            cached_content_hash = warm_entry.content_hash if warm_entry is not None else None
            conflict = sync_mixer.reconcile(
                cached_content_hash, authoritative_content(document_id), document_id
            )
            if conflict is not None:
                await self._warm_store.demote(tenant_id, user_id, document_id)
                conflicts.append(conflict)
        return conflicts
