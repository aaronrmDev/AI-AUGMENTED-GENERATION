import uuid
from datetime import datetime, timedelta

from src.orchestration.infrastructure.in_memory_access_tracker import (
    InMemoryAccessFrequencyTracker,
)

_NOW = datetime(2026, 9, 2, 12, 0, 0)
_WINDOW = timedelta(hours=1)


def test_empty_tracker_has_zero_count_and_no_most_accessed():
    tracker = InMemoryAccessFrequencyTracker()
    doc = uuid.uuid4()

    assert tracker.access_count(doc, _WINDOW, _NOW) == 0
    assert tracker.most_accessed(5, _WINDOW, _NOW) == []


def test_recorded_accesses_within_the_window_are_counted():
    tracker = InMemoryAccessFrequencyTracker()
    doc = uuid.uuid4()

    tracker.record_access(doc, _NOW - timedelta(minutes=1))
    tracker.record_access(doc, _NOW - timedelta(minutes=30))
    tracker.record_access(doc, _NOW)

    assert tracker.access_count(doc, _WINDOW, _NOW) == 3


def test_accesses_outside_the_window_do_not_count():
    tracker = InMemoryAccessFrequencyTracker()
    doc = uuid.uuid4()

    tracker.record_access(doc, _NOW - timedelta(hours=2))
    tracker.record_access(doc, _NOW - timedelta(minutes=59))

    assert tracker.access_count(doc, _WINDOW, _NOW) == 1


def test_most_accessed_ranks_by_count_descending():
    tracker = InMemoryAccessFrequencyTracker()
    hot, warm, cold = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    for _ in range(5):
        tracker.record_access(hot, _NOW)
    for _ in range(2):
        tracker.record_access(warm, _NOW)
    tracker.record_access(cold, _NOW)

    assert tracker.most_accessed(3, _WINDOW, _NOW) == [hot, warm, cold]


def test_most_accessed_respects_n():
    tracker = InMemoryAccessFrequencyTracker()
    hot, warm = uuid.uuid4(), uuid.uuid4()
    tracker.record_access(hot, _NOW)
    tracker.record_access(warm, _NOW)

    assert tracker.most_accessed(1, _WINDOW, _NOW) == [hot]


def test_most_accessed_excludes_documents_with_zero_accesses_in_window():
    tracker = InMemoryAccessFrequencyTracker()
    doc = uuid.uuid4()
    tracker.record_access(doc, _NOW - timedelta(hours=5))  # outside window

    assert tracker.most_accessed(5, _WINDOW, _NOW) == []
