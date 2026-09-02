import uuid
from collections import defaultdict
from datetime import datetime, timedelta

from src.orchestration.domain.ports import AccessFrequencyTracker


class InMemoryAccessFrequencyTracker(AccessFrequencyTracker):
    """A real sliding-window access counter -- not a test double.

    Every mechanism in this batch (Cache-Warmed RAG's warming, tiering's
    promote/demote, the umbrella validation's hit-rate reporting) exercises
    this for real, the same way qdrant_vector_store.py is RAG's real
    VectorStore implementation rather than a fake standing in for one.
    """

    def __init__(self) -> None:
        self._accesses: dict[uuid.UUID, list[datetime]] = defaultdict(list)

    def record_access(self, document_id: uuid.UUID, at: datetime) -> None:
        self._accesses[document_id].append(at)

    def access_count(self, document_id: uuid.UUID, window: timedelta, now: datetime) -> int:
        cutoff = now - window
        return sum(1 for at in self._accesses.get(document_id, []) if cutoff <= at <= now)

    def most_accessed(self, n: int, window: timedelta, now: datetime) -> list[uuid.UUID]:
        counts = {
            document_id: self.access_count(document_id, window, now)
            for document_id in self._accesses
        }
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [document_id for document_id, count in ranked if count > 0][:n]
