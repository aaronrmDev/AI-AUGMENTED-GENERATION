import uuid
from collections.abc import Callable

from src.orchestration.domain import sync_mixer
from src.orchestration.domain.entities import SyncConflict
from src.orchestration.domain.ports import FrozenCache


class SyncCycle:
    """The Sync Mixer's batch invalidation trigger: for each currently
    tracked document, reconcile what's cached against RAG's current
    authoritative content and evict on a real conflict (RAG wins). A
    caller drives `run` on whatever cadence it chooses -- a test on a real
    short interval, a future scheduler on a longer one; this class owns no
    clock of its own.
    """

    def __init__(self, frozen_cache: FrozenCache) -> None:
        self._frozen_cache = frozen_cache

    def run(
        self,
        tenant_id: uuid.UUID,
        document_ids: list[uuid.UUID],
        authoritative_content: Callable[[uuid.UUID], str],
    ) -> list[SyncConflict]:
        conflicts: list[SyncConflict] = []
        for document_id in document_ids:
            cached_hit = self._frozen_cache.lookup(tenant_id, document_id)
            cached_content_hash = cached_hit.content_hash if cached_hit is not None else None
            conflict = sync_mixer.reconcile(
                cached_content_hash, authoritative_content(document_id), document_id
            )
            if conflict is not None:
                self._frozen_cache.evict(tenant_id, document_id)
                conflicts.append(conflict)
        return conflicts
