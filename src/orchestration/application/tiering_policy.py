import uuid
from collections.abc import Callable
from datetime import datetime, timedelta

from src.orchestration.domain.entities import TierDecision
from src.orchestration.domain.ports import AccessFrequencyTracker, FrozenCache


class TieringPolicy:
    """The CAG-RAG hot/cold tiering boundary: promote a cold (RAG-only)
    document into CAG's hot tier once its access rate crosses
    promote_threshold; demote an already-hot document back to cold once its
    access rate falls below demote_threshold, freeing whatever the cache
    entry was occupying.

    Reuses the same AccessFrequencyTracker/FrozenCache ports WarmCache
    uses -- promotion and Cache-Warmed RAG's warming step are, honestly,
    the same underlying mechanism (preload driven by access frequency)
    under two different trigger policies (threshold-crossing here,
    top-N snapshot there).
    """

    def __init__(self, tracker: AccessFrequencyTracker, frozen_cache: FrozenCache) -> None:
        self._tracker = tracker
        self._frozen_cache = frozen_cache

    def evaluate(
        self,
        document_id: uuid.UUID,
        content_provider: Callable[[uuid.UUID], str],
        promote_threshold: int,
        demote_threshold: int,
        window: timedelta,
        now: datetime,
    ) -> TierDecision:
        if promote_threshold < demote_threshold:
            raise ValueError(
                "promote_threshold must be >= demote_threshold -- a hysteresis band, "
                "not a single crossing point that could flap a document in and out of cache"
            )

        count = self._tracker.access_count(document_id, window, now)
        is_cached = self._frozen_cache.contains(document_id)

        if count >= promote_threshold and not is_cached:
            self._frozen_cache.preload(document_id, content_provider(document_id))
            return TierDecision.PROMOTED

        if count < demote_threshold and is_cached:
            self._frozen_cache.evict(document_id)
            return TierDecision.DEMOTED

        return TierDecision.UNCHANGED
