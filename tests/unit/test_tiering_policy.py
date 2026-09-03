import uuid
from datetime import datetime, timedelta

import pytest

from src.orchestration.application.tiering_policy import TieringPolicy
from src.orchestration.domain.entities import TierDecision
from src.orchestration.infrastructure.in_memory_access_tracker import (
    InMemoryAccessFrequencyTracker,
)
from tests.unit.orchestration_fakes import FakeFrozenCache

_NOW = datetime(2026, 9, 2, 12, 0, 0)
_WINDOW = timedelta(hours=1)
_PROMOTE = 10
_DEMOTE = 3
_TENANT = uuid.uuid4()


def _content_provider(text: str = "some content"):
    return lambda document_id: text


def test_promotes_when_threshold_crossed_and_not_yet_cached():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    for _ in range(_PROMOTE):
        tracker.record_access(_TENANT, doc, _NOW)
    policy = TieringPolicy(tracker, cache)

    decision = policy.evaluate(
        _TENANT, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision == TierDecision.PROMOTED
    assert cache.contains(_TENANT, doc)


def test_no_double_promotion_once_already_cached():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    for _ in range(_PROMOTE):
        tracker.record_access(_TENANT, doc, _NOW)
    cache.preload(_TENANT, doc, "already hot")
    policy = TieringPolicy(tracker, cache)

    decision = policy.evaluate(
        _TENANT, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision == TierDecision.UNCHANGED
    assert cache.preload_calls == [(_TENANT, doc, "already hot")]


def test_demotes_when_access_falls_below_threshold_on_a_cached_doc():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    tracker.record_access(_TENANT, doc, _NOW)  # count == 1, below demote threshold of 3
    cache.preload(_TENANT, doc, "cooling off")
    policy = TieringPolicy(tracker, cache)

    decision = policy.evaluate(
        _TENANT, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision == TierDecision.DEMOTED
    assert not cache.contains(_TENANT, doc)
    assert cache.evict_calls == [(_TENANT, doc)]


def test_unchanged_in_the_hysteresis_band_between_thresholds():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    for _ in range(5):  # between demote(3) and promote(10)
        tracker.record_access(_TENANT, doc, _NOW)
    policy = TieringPolicy(tracker, cache)

    not_cached_decision = policy.evaluate(
        _TENANT, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )
    assert not_cached_decision == TierDecision.UNCHANGED
    assert not cache.contains(_TENANT, doc)

    cache.preload(_TENANT, doc, "already cached")
    cached_decision = policy.evaluate(
        _TENANT, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )
    assert cached_decision == TierDecision.UNCHANGED
    assert cache.contains(_TENANT, doc)


def test_a_document_sitting_exactly_between_thresholds_does_not_flap_across_repeated_evaluations():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    for _ in range(5):
        tracker.record_access(_TENANT, doc, _NOW)
    policy = TieringPolicy(tracker, cache)

    decisions = [
        policy.evaluate(_TENANT, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW)
        for _ in range(4)
    ]

    assert decisions == [TierDecision.UNCHANGED] * 4
    assert not cache.contains(_TENANT, doc)


def test_rejects_a_promote_threshold_below_the_demote_threshold():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    policy = TieringPolicy(tracker, cache)

    with pytest.raises(ValueError, match="promote_threshold"):
        policy.evaluate(
            _TENANT,
            uuid.uuid4(),
            _content_provider(),
            demote_threshold=10,
            promote_threshold=3,
            window=_WINDOW,
            now=_NOW,
        )


def test_tiering_is_isolated_per_tenant():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    for _ in range(_PROMOTE):
        tracker.record_access(tenant_a, doc, _NOW)
    policy = TieringPolicy(tracker, cache)

    decision_a = policy.evaluate(
        tenant_a, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )
    decision_b = policy.evaluate(
        tenant_b, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision_a == TierDecision.PROMOTED
    assert decision_b == TierDecision.UNCHANGED
    assert cache.contains(tenant_a, doc)
    assert not cache.contains(tenant_b, doc)
