import uuid
from collections import defaultdict
from datetime import datetime, timedelta

from src.orchestration.domain.ports import AccessFrequencyTracker

_Key = tuple[uuid.UUID, uuid.UUID]  # (tenant_id, document_id)
_DEFAULT_RETENTION = timedelta(days=30)


class InMemoryAccessFrequencyTracker(AccessFrequencyTracker):
    """A real sliding-window access counter -- not a test double.

    Every mechanism in this batch (Cache-Warmed RAG's warming, tiering's
    promote/demote, the umbrella validation's hit-rate reporting) exercises
    this for real, the same way qdrant_vector_store.py is RAG's real
    VectorStore implementation rather than a fake standing in for one.

    Prunes raw access timestamps older than `retention` on every read --
    a review finding caught the first version of this class accumulating
    every access forever, growing memory and per-call scan cost without
    bound under sustained real traffic. Retention is a SEPARATE, fixed
    garbage-collection horizon from the `window`/`now` any individual
    access_count/most_accessed call passes: pruning against a per-call
    window instead (an earlier draft of this fix) would have silently
    corrupted results whenever two callers queried the same tracker with
    different window sizes -- a narrower-window call would have deleted
    timestamps a later wider-window call still legitimately needed. As
    long as `retention` is at least as large as any window this tracker
    is actually queried with, every access_count/most_accessed result
    stays exactly what it would be with no pruning at all.
    """

    def __init__(self, retention: timedelta = _DEFAULT_RETENTION) -> None:
        self._retention = retention
        self._accesses: dict[_Key, list[datetime]] = defaultdict(list)

    def record_access(self, tenant_id: uuid.UUID, document_id: uuid.UUID, at: datetime) -> None:
        key = (tenant_id, document_id)
        self._accesses[key].append(at)
        self._prune(key, now=at)

    def access_count(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, window: timedelta, now: datetime
    ) -> int:
        key = (tenant_id, document_id)
        self._prune(key, now)
        cutoff = now - window
        return sum(1 for at in self._accesses.get(key, []) if cutoff <= at <= now)

    def most_accessed(
        self, tenant_id: uuid.UUID, n: int, window: timedelta, now: datetime
    ) -> list[uuid.UUID]:
        counts = {
            document_id: self.access_count(tenant_id, document_id, window, now)
            for (tid, document_id) in list(self._accesses)
            if tid == tenant_id
        }
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [document_id for document_id, count in ranked if count > 0][:n]

    def _prune(self, key: _Key, now: datetime) -> None:
        cutoff = now - self._retention
        accesses = self._accesses.get(key)
        if accesses is None:
            return
        kept = [at for at in accesses if at >= cutoff]
        if kept:
            self._accesses[key] = kept
        else:
            del self._accesses[key]
