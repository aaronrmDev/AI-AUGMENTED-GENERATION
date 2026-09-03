import uuid
from collections.abc import Callable

from src.orchestration.domain import cag_mag_keys, sync_mixer
from src.orchestration.domain.entities import SyncConflict
from src.orchestration.domain.ports import FrozenCache


class CagMagSyncCycle:
    """The Sync Mixer's CAG-vs-MAG batch invalidation trigger.

    Unlike RAG-vs-CAG (SyncCycle) and RAG-vs-MAG (MagSyncCycle), the
    source material names no rule for this case at all -- #5's own
    investigation (see the design spec) concludes a genuine CAG-vs-MAG-
    only conflict CAN arise, created by this batch's own tiering
    mechanism (CagMagTieringPolicy) freezing a point-in-time copy of MAG
    content into CAG's hot tier while MAG's own live data keeps changing.
    The resolution rule here is MAG wins, not RAG wins: CAG's hot-tier
    entry is a cached acceleration of a specific MAG record, not an
    independent paradigm holding a competing fact, so MAG -- the one
    paradigm that actually owns this data -- is the side that's
    authoritative. `sync_mixer.reconcile` doesn't know or care which
    paradigm is "authoritative"; it only compares a hash, which is why
    the same function serves all three pairings' Sync Mixer mechanisms
    without any change.

    Talks only to FrozenCache (via cag_mag_keys.cache_key), the same
    posture SyncCycle holds toward FrozenCache and MagSyncCycle holds
    toward WarmStore. Sync, not async, matching CagMagTieringPolicy and
    FrozenCache -- unlike MagSyncCycle, which is async because WarmStore
    does real I/O.
    """

    def __init__(self, frozen_cache: FrozenCache) -> None:
        self._frozen_cache = frozen_cache

    def run(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        mag_content_keys: list[str],
        authoritative_content: Callable[[str], str],
    ) -> list[SyncConflict]:
        conflicts: list[SyncConflict] = []
        for mag_content_key in mag_content_keys:
            cache_id = cag_mag_keys.cache_key(user_id, mag_content_key)
            cached_hit = self._frozen_cache.lookup(tenant_id, cache_id)
            cached_content_hash = cached_hit.content_hash if cached_hit is not None else None
            conflict = sync_mixer.reconcile(
                cached_content_hash, authoritative_content(mag_content_key), cache_id
            )
            if conflict is not None:
                self._frozen_cache.evict(tenant_id, cache_id)
                conflicts.append(conflict)
        return conflicts
