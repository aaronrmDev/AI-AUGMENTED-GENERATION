import uuid
from datetime import datetime, timedelta

from src.orchestration.domain.ports import UserScopedAccessFrequencyTracker
from src.orchestration.infrastructure._sliding_window_counter import SlidingWindowCounter

_Key = tuple[uuid.UUID, uuid.UUID, uuid.UUID]  # (tenant_id, user_id, document_id)
_DEFAULT_RETENTION = timedelta(days=30)


class InMemoryUserScopedAccessFrequencyTracker(UserScopedAccessFrequencyTracker):
    """A real sliding-window access counter, scoped per user -- MAG's warm
    tier is personal, unlike CAG's tenant-wide shared cache, so this
    tracks (tenant_id, user_id, document_id) rather than
    InMemoryAccessFrequencyTracker's (tenant_id, document_id).

    The actual counting/pruning algorithm lives in the shared
    SlidingWindowCounter (also used by InMemoryAccessFrequencyTracker,
    CAG's own tracker) -- this class only owns the extra user_id key
    dimension its own port requires.
    """

    def __init__(self, retention: timedelta = _DEFAULT_RETENTION) -> None:
        self._counter: SlidingWindowCounter[_Key] = SlidingWindowCounter(retention)

    def record_access(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID, at: datetime
    ) -> None:
        self._counter.record((tenant_id, user_id, document_id), at)

    def access_count(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        window: timedelta,
        now: datetime,
    ) -> int:
        return self._counter.count((tenant_id, user_id, document_id), window, now)

    def most_accessed(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, n: int, window: timedelta, now: datetime
    ) -> list[uuid.UUID]:
        counts = {
            document_id: self._counter.count((tid, uid, document_id), window, now)
            for (tid, uid, document_id) in self._counter.keys()
            if tid == tenant_id and uid == user_id
        }
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [document_id for document_id, count in ranked if count > 0][:n]
