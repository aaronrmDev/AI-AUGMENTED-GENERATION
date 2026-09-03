import uuid
from datetime import datetime, timedelta

from src.orchestration.domain.ports import AccessFrequencyTracker
from src.orchestration.infrastructure._sliding_window_counter import SlidingWindowCounter

_Key = tuple[uuid.UUID, uuid.UUID]  # (tenant_id, document_id)
_DEFAULT_RETENTION = timedelta(days=30)


class InMemoryAccessFrequencyTracker(AccessFrequencyTracker):
    """A real sliding-window access counter -- not a test double.

    Every mechanism in this batch (Cache-Warmed RAG's warming, tiering's
    promote/demote, the umbrella validation's hit-rate reporting) exercises
    this for real, the same way qdrant_vector_store.py is RAG's real
    VectorStore implementation rather than a fake standing in for one.

    The actual counting/pruning algorithm lives in the shared
    SlidingWindowCounter (also used by InMemoryUserScopedAccessFrequencyTracker,
    RAG+MAG's own tracker) -- this class only owns the (tenant_id,
    document_id) key shape its own port requires. Retention is a fixed
    garbage-collection horizon, deliberately decoupled from any individual
    access_count/most_accessed call's own `window`/`now`: pruning against
    a per-call window instead (an earlier draft of this fix) would have
    silently corrupted results whenever two callers queried with different
    window sizes.
    """

    def __init__(self, retention: timedelta = _DEFAULT_RETENTION) -> None:
        self._counter: SlidingWindowCounter[_Key] = SlidingWindowCounter(retention)

    def record_access(self, tenant_id: uuid.UUID, document_id: uuid.UUID, at: datetime) -> None:
        self._counter.record((tenant_id, document_id), at)

    def access_count(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, window: timedelta, now: datetime
    ) -> int:
        return self._counter.count((tenant_id, document_id), window, now)

    def most_accessed(
        self, tenant_id: uuid.UUID, n: int, window: timedelta, now: datetime
    ) -> list[uuid.UUID]:
        counts = {
            document_id: self._counter.count((tid, document_id), window, now)
            for (tid, document_id) in self._counter.keys()
            if tid == tenant_id
        }
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [document_id for document_id, count in ranked if count > 0][:n]
