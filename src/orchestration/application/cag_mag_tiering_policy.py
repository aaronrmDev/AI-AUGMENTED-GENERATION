import uuid
from collections.abc import Callable
from datetime import datetime, timedelta

from src.orchestration.domain import cag_mag_keys
from src.orchestration.domain.entities import TierDecision
from src.orchestration.domain.ports import FrozenCache, UserScopedAccessFrequencyTracker


class CagMagTieringPolicy:
    """The CAG-MAG hot/warm tiering boundary: promote a MAG warm-tier
    piece of content into CAG's hot tier once THIS USER's access rate
    crosses promote_threshold; demote an already-hot entry back to MAG's
    warm tier once it falls below demote_threshold, freeing whatever the
    cache entry was occupying.

    The same promote/threshold/demote/hysteresis shape as TieringPolicy
    (CAG's RAG-vs-CAG version) and MagTieringPolicy (RAG-vs-MAG), built
    directly on Batch D's FrozenCache and Batch E's
    UserScopedAccessFrequencyTracker -- no new port. Every FrozenCache
    call is keyed via cag_mag_keys.cache_key(user_id, mag_content_key),
    which folds user_id into the key itself since FrozenCache has no
    user_id parameter of its own; every tracker call uses
    cag_mag_keys.tracker_key(mag_content_key) alongside the tracker's own
    real user_id argument, which already provides that scoping.

    Sync, not async, unlike MagTieringPolicy -- FrozenCache and
    UserScopedAccessFrequencyTracker are both pure in-memory/local-CPU
    with no I/O, even though the content being promoted originates from
    MAG (a real caller fetches that content from MAG first; this class
    only ever sees the resulting string).
    """

    def __init__(
        self, tracker: UserScopedAccessFrequencyTracker, frozen_cache: FrozenCache
    ) -> None:
        self._tracker = tracker
        self._frozen_cache = frozen_cache

    def evaluate(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        mag_content_key: str,
        content_provider: Callable[[str], str],
        promote_threshold: int,
        demote_threshold: int,
        window: timedelta,
        now: datetime,
    ) -> TierDecision:
        if promote_threshold < demote_threshold:
            raise ValueError(
                "promote_threshold must be >= demote_threshold -- a hysteresis band, "
                "not a single crossing point that could flap content in and out of the hot tier"
            )

        count = self._tracker.access_count(
            tenant_id, user_id, cag_mag_keys.tracker_key(mag_content_key), window, now
        )
        cache_id = cag_mag_keys.cache_key(user_id, mag_content_key)
        is_hot = self._frozen_cache.contains(tenant_id, cache_id)

        if count >= promote_threshold and not is_hot:
            self._frozen_cache.preload(tenant_id, cache_id, content_provider(mag_content_key))
            return TierDecision.PROMOTED

        if count < demote_threshold and is_hot:
            self._frozen_cache.evict(tenant_id, cache_id)
            return TierDecision.DEMOTED

        return TierDecision.UNCHANGED
