import uuid
from collections.abc import Callable
from datetime import datetime, timedelta

from src.orchestration.domain.ports import AccessFrequencyTracker, FrozenCache


class WarmCache:
    """Cache-Warmed RAG's periodic re-warming step: preload the current
    top-N most-accessed documents into a FrozenCache, skipping whatever is
    already there. Returns the full top-N -- what's warm after this call,
    whether freshly preloaded now or already present from an earlier call.
    """

    def __init__(self, tracker: AccessFrequencyTracker, frozen_cache: FrozenCache) -> None:
        self._tracker = tracker
        self._frozen_cache = frozen_cache

    def execute(
        self,
        tenant_id: uuid.UUID,
        n: int,
        window: timedelta,
        now: datetime,
        content_provider: Callable[[uuid.UUID], str],
    ) -> list[uuid.UUID]:
        top_n = self._tracker.most_accessed(tenant_id, n, window, now)
        for document_id in top_n:
            if not self._frozen_cache.contains(tenant_id, document_id):
                self._frozen_cache.preload(tenant_id, document_id, content_provider(document_id))
        return top_n
