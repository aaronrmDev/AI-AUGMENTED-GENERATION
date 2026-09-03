import uuid
from datetime import datetime, timedelta

from src.orchestration.infrastructure.in_memory_user_scoped_access_tracker import (
    InMemoryUserScopedAccessFrequencyTracker,
)

_NOW = datetime(2026, 9, 2, 12, 0, 0)
_WINDOW = timedelta(hours=1)
_TENANT = uuid.uuid4()
_USER = uuid.uuid4()


def test_empty_tracker_has_zero_count_and_no_most_accessed():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    doc = uuid.uuid4()

    assert tracker.access_count(_TENANT, _USER, doc, _WINDOW, _NOW) == 0
    assert tracker.most_accessed(_TENANT, _USER, 5, _WINDOW, _NOW) == []


def test_recorded_accesses_within_the_window_are_counted():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    doc = uuid.uuid4()

    tracker.record_access(_TENANT, _USER, doc, _NOW - timedelta(minutes=1))
    tracker.record_access(_TENANT, _USER, doc, _NOW - timedelta(minutes=30))
    tracker.record_access(_TENANT, _USER, doc, _NOW)

    assert tracker.access_count(_TENANT, _USER, doc, _WINDOW, _NOW) == 3


def test_accesses_outside_the_window_do_not_count():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    doc = uuid.uuid4()

    tracker.record_access(_TENANT, _USER, doc, _NOW - timedelta(hours=2))
    tracker.record_access(_TENANT, _USER, doc, _NOW - timedelta(minutes=59))

    assert tracker.access_count(_TENANT, _USER, doc, _WINDOW, _NOW) == 1


def test_most_accessed_ranks_by_count_descending():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    hot, warm, cold = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    for _ in range(5):
        tracker.record_access(_TENANT, _USER, hot, _NOW)
    for _ in range(2):
        tracker.record_access(_TENANT, _USER, warm, _NOW)
    tracker.record_access(_TENANT, _USER, cold, _NOW)

    assert tracker.most_accessed(_TENANT, _USER, 3, _WINDOW, _NOW) == [hot, warm, cold]


def test_a_three_way_tie_breaks_by_first_recorded_and_is_pinned_by_this_test():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    first, second, third = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    tracker.record_access(_TENANT, _USER, first, _NOW)
    tracker.record_access(_TENANT, _USER, second, _NOW)
    tracker.record_access(_TENANT, _USER, third, _NOW)

    assert tracker.most_accessed(_TENANT, _USER, 3, _WINDOW, _NOW) == [first, second, third]


def test_access_counts_are_isolated_per_user_even_for_the_same_document_and_tenant():
    # The specific bug this port's whole design exists to avoid: heavy
    # traffic from one user must never promote a document into a
    # DIFFERENT user's personal memory.
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    for _ in range(5):
        tracker.record_access(_TENANT, user_a, doc, _NOW)
    tracker.record_access(_TENANT, user_b, doc, _NOW)

    assert tracker.access_count(_TENANT, user_a, doc, _WINDOW, _NOW) == 5
    assert tracker.access_count(_TENANT, user_b, doc, _WINDOW, _NOW) == 1
    assert tracker.most_accessed(_TENANT, user_b, 5, _WINDOW, _NOW) == [doc]


def test_access_counts_are_also_isolated_per_tenant():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    for _ in range(5):
        tracker.record_access(tenant_a, _USER, doc, _NOW)
    tracker.record_access(tenant_b, _USER, doc, _NOW)

    assert tracker.access_count(tenant_a, _USER, doc, _WINDOW, _NOW) == 5
    assert tracker.access_count(tenant_b, _USER, doc, _WINDOW, _NOW) == 1


def test_retention_prunes_old_accesses_without_corrupting_a_wider_later_window():
    tracker = InMemoryUserScopedAccessFrequencyTracker(retention=timedelta(hours=2))
    doc = uuid.uuid4()
    old_access = _NOW - timedelta(hours=1, minutes=30)  # within retention, outside a 1h window
    tracker.record_access(_TENANT, _USER, doc, old_access)
    tracker.record_access(_TENANT, _USER, doc, _NOW)

    assert tracker.access_count(_TENANT, _USER, doc, timedelta(hours=1), _NOW) == 1
    assert tracker.access_count(_TENANT, _USER, doc, timedelta(hours=3), _NOW) == 2


def test_accesses_older_than_retention_are_actually_dropped():
    tracker = InMemoryUserScopedAccessFrequencyTracker(retention=timedelta(hours=1))
    doc = uuid.uuid4()
    tracker.record_access(_TENANT, _USER, doc, _NOW - timedelta(hours=5))
    tracker.record_access(_TENANT, _USER, doc, _NOW)

    assert tracker.access_count(_TENANT, _USER, doc, timedelta(hours=1), _NOW) == 1
    assert tracker.access_count(_TENANT, _USER, doc, timedelta(days=2), _NOW) == 1
