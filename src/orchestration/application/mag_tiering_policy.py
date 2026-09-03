import uuid
from collections.abc import Callable
from datetime import datetime, timedelta

from src.orchestration.domain.entities import TierDecision
from src.orchestration.domain.ports import UserScopedAccessFrequencyTracker, WarmStore


class MagTieringPolicy:
    """The RAG-MAG warm/cold tiering boundary: promote a cold (RAG-only)
    document into MAG's warm tier once THIS USER's access rate crosses
    promote_threshold; demote an already-warm document back to cold once
    it falls below demote_threshold.

    The same promote/threshold/demote/hysteresis shape as
    src.orchestration.application.tiering_policy.TieringPolicy (CAG's
    version), built on UserScopedAccessFrequencyTracker/WarmStore instead
    of AccessFrequencyTracker/FrozenCache -- MAG's warm tier is inherently
    personal, so this is a parallel implementation, not a shared one; see
    the design spec for why the two ports genuinely differ.

    async, unlike TieringPolicy, because WarmStore's real implementation
    makes real network calls to Postgres/Qdrant.
    """

    def __init__(
        self, tracker: UserScopedAccessFrequencyTracker, warm_store: WarmStore
    ) -> None:
        self._tracker = tracker
        self._warm_store = warm_store

    async def evaluate(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
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
                "not a single crossing point that could flap a document in and out of the warm tier"
            )

        count = self._tracker.access_count(tenant_id, user_id, document_id, window, now)
        is_warm = await self._warm_store.contains(tenant_id, user_id, document_id)

        if count >= promote_threshold and not is_warm:
            await self._warm_store.promote(
                tenant_id, user_id, document_id, content_provider(document_id)
            )
            return TierDecision.PROMOTED

        if count < demote_threshold and is_warm:
            await self._warm_store.demote(tenant_id, user_id, document_id)
            return TierDecision.DEMOTED

        return TierDecision.UNCHANGED
