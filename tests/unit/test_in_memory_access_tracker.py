import uuid
from datetime import datetime, timedelta

from src.orchestration.infrastructure.in_memory_access_tracker import (
    InMemoryAccessFrequencyTracker,
)

_NOW = datetime(2026, 9, 2, 12, 0, 0)
_WINDOW = timedelta(hours=1)
_TENANT = uuid.uuid4()


def test_empty_tracker_has_zero_count_and_no_most_accessed():
    tracker = InMemoryAccessFrequencyTracker()
    doc = uuid.uuid4()

    assert tracker.access_count(_TENANT, doc, _WINDOW, _NOW) == 0
    assert tracker.most_accessed(_TENANT, 5, _WINDOW, _NOW) == []


def test_recorded_accesses_within_the_window_are_counted():
    tracker = InMemoryAccessFrequencyTracker()
    doc = uuid.uuid4()

    tracker.record_access(_TENANT, doc, _NOW - timedelta(minutes=1))
    tracker.record_access(_TENANT, doc, _NOW - timedelta(minutes=30))
    tracker.record_access(_TENANT, doc, _NOW)

    assert tracker.access_count(_TENANT, doc, _WINDOW, _NOW) == 3


def test_accesses_outside_the_window_do_not_count():
    tracker = InMemoryAccessFrequencyTracker()
    doc = uuid.uuid4()

    tracker.record_access(_TENANT, doc, _NOW - timedelta(hours=2))
    tracker.record_access(_TENANT, doc, _NOW - timedelta(minutes=59))

    assert tracker.access_count(_TENANT, doc, _WINDOW, _NOW) == 1


def test_most_accessed_ranks_by_count_descending():
    tracker = InMemoryAccessFrequencyTracker()
    hot, warm, cold = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    for _ in range(5):
        tracker.record_access(_TENANT, hot, _NOW)
    for _ in range(2):
        tracker.record_access(_TENANT, warm, _NOW)
    tracker.record_access(_TENANT, cold, _NOW)

    assert tracker.most_accessed(_TENANT, 3, _WINDOW, _NOW) == [hot, warm, cold]


def test_most_accessed_respects_n():
    tracker = InMemoryAccessFrequencyTracker()
    hot, warm = uuid.uuid4(), uuid.uuid4()
    tracker.record_access(_TENANT, hot, _NOW)
    tracker.record_access(_TENANT, warm, _NOW)

    assert tracker.most_accessed(_TENANT, 1, _WINDOW, _NOW) == [hot]


def test_most_accessed_excludes_documents_with_zero_accesses_in_window():
    tracker = InMemoryAccessFrequencyTracker()
    doc = uuid.uuid4()
    tracker.record_access(_TENANT, doc, _NOW - timedelta(hours=5))  # outside window

    assert tracker.most_accessed(_TENANT, 5, _WINDOW, _NOW) == []


def test_a_three_way_tie_breaks_by_first_recorded_and_is_pinned_by_this_test():
    # Explicit, dedicated ties coverage -- a review finding caught that the
    # design spec's own testing plan calls for "most_accessed ordering and
    # ties" but the only tie this file exercised was incidental to an n=1
    # truncation test, never a labeled test verifying the full ordering
    # among 3+ genuinely tied documents.
    tracker = InMemoryAccessFrequencyTracker()
    first, second, third = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    tracker.record_access(_TENANT, first, _NOW)
    tracker.record_access(_TENANT, second, _NOW)
    tracker.record_access(_TENANT, third, _NOW)

    assert tracker.most_accessed(_TENANT, 3, _WINDOW, _NOW) == [first, second, third]


def test_access_counts_are_isolated_per_tenant():
    tracker = InMemoryAccessFrequencyTracker()
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()  # same document_id, two different tenants
    for _ in range(5):
        tracker.record_access(tenant_a, doc, _NOW)
    tracker.record_access(tenant_b, doc, _NOW)

    assert tracker.access_count(tenant_a, doc, _WINDOW, _NOW) == 5
    assert tracker.access_count(tenant_b, doc, _WINDOW, _NOW) == 1
    assert tracker.most_accessed(tenant_b, 5, _WINDOW, _NOW) == [doc]


def test_retention_prunes_old_accesses_without_corrupting_a_wider_later_window():
    # A review finding caught the first fix attempt at this problem pruning
    # against each CALL's own window, which would have silently corrupted
    # results whenever a narrower-window call ran before a wider-window
    # call on the same tracker. Retention must be independent of any
    # individual query window.
    tracker = InMemoryAccessFrequencyTracker(retention=timedelta(hours=2))
    doc = uuid.uuid4()
    old_access = _NOW - timedelta(hours=1, minutes=30)  # within retention, outside a 1h window
    tracker.record_access(_TENANT, doc, old_access)
    tracker.record_access(_TENANT, doc, _NOW)

    # A narrow-window read first...
    assert tracker.access_count(_TENANT, doc, timedelta(hours=1), _NOW) == 1
    # ...must not have pruned away the still-within-retention older access.
    assert tracker.access_count(_TENANT, doc, timedelta(hours=3), _NOW) == 2


def test_accesses_older_than_retention_are_actually_dropped():
    tracker = InMemoryAccessFrequencyTracker(retention=timedelta(hours=1))
    doc = uuid.uuid4()
    tracker.record_access(_TENANT, doc, _NOW - timedelta(hours=5))
    tracker.record_access(_TENANT, doc, _NOW)

    # A read at `_NOW` prunes the 5-hours-old access permanently (older
    # than the 1-hour retention horizon)...
    assert tracker.access_count(_TENANT, doc, timedelta(hours=1), _NOW) == 1
    # ...so a second read moments later with a much WIDER window still
    # can't recover it -- it's genuinely gone, not merely excluded by the
    # first call's own narrower window.
    assert tracker.access_count(_TENANT, doc, timedelta(days=2), _NOW) == 1
