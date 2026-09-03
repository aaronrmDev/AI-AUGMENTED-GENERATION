import uuid
from datetime import datetime, timedelta

import pytest

from src.orchestration.application.mag_tiering_policy import MagTieringPolicy
from src.orchestration.domain.entities import TierDecision
from src.orchestration.infrastructure.in_memory_user_scoped_access_tracker import (
    InMemoryUserScopedAccessFrequencyTracker,
)
from tests.unit.orchestration_fakes import FakeWarmStore

_NOW = datetime(2026, 9, 2, 12, 0, 0)
_WINDOW = timedelta(hours=1)
_PROMOTE = 10
_DEMOTE = 3
_TENANT = uuid.uuid4()
_USER = uuid.uuid4()


def _content_provider(text: str = "some content"):
    return lambda document_id: text


async def test_promotes_when_threshold_crossed_and_not_yet_warm():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    warm_store = FakeWarmStore()
    doc = uuid.uuid4()
    for _ in range(_PROMOTE):
        tracker.record_access(_TENANT, _USER, doc, _NOW)
    policy = MagTieringPolicy(tracker, warm_store)

    decision = await policy.evaluate(
        _TENANT, _USER, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision == TierDecision.PROMOTED
    assert await warm_store.contains(_TENANT, _USER, doc)


async def test_no_double_promotion_once_already_warm():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    warm_store = FakeWarmStore()
    doc = uuid.uuid4()
    for _ in range(_PROMOTE):
        tracker.record_access(_TENANT, _USER, doc, _NOW)
    await warm_store.promote(_TENANT, _USER, doc, "already warm")
    policy = MagTieringPolicy(tracker, warm_store)

    decision = await policy.evaluate(
        _TENANT, _USER, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision == TierDecision.UNCHANGED
    assert warm_store.promote_calls == [(_TENANT, _USER, doc, "already warm")]


async def test_demotes_when_access_falls_below_threshold_on_a_warm_doc():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    warm_store = FakeWarmStore()
    doc = uuid.uuid4()
    tracker.record_access(_TENANT, _USER, doc, _NOW)  # count == 1, below demote threshold of 3
    await warm_store.promote(_TENANT, _USER, doc, "cooling off")
    policy = MagTieringPolicy(tracker, warm_store)

    decision = await policy.evaluate(
        _TENANT, _USER, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision == TierDecision.DEMOTED
    assert not await warm_store.contains(_TENANT, _USER, doc)
    assert warm_store.demote_calls == [(_TENANT, _USER, doc)]


async def test_a_document_sitting_exactly_between_thresholds_does_not_flap():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    warm_store = FakeWarmStore()
    doc = uuid.uuid4()
    for _ in range(5):  # between demote(3) and promote(10)
        tracker.record_access(_TENANT, _USER, doc, _NOW)
    policy = MagTieringPolicy(tracker, warm_store)

    decisions = [
        await policy.evaluate(
            _TENANT, _USER, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
        )
        for _ in range(4)
    ]

    assert decisions == [TierDecision.UNCHANGED] * 4
    assert not await warm_store.contains(_TENANT, _USER, doc)


async def test_rejects_a_promote_threshold_below_the_demote_threshold():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    warm_store = FakeWarmStore()
    policy = MagTieringPolicy(tracker, warm_store)

    with pytest.raises(ValueError, match="promote_threshold"):
        await policy.evaluate(
            _TENANT,
            _USER,
            uuid.uuid4(),
            _content_provider(),
            demote_threshold=10,
            promote_threshold=3,
            window=_WINDOW,
            now=_NOW,
        )


async def test_tiering_is_isolated_per_user():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    warm_store = FakeWarmStore()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    for _ in range(_PROMOTE):
        tracker.record_access(_TENANT, user_a, doc, _NOW)
    policy = MagTieringPolicy(tracker, warm_store)

    decision_a = await policy.evaluate(
        _TENANT, user_a, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )
    decision_b = await policy.evaluate(
        _TENANT, user_b, doc, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision_a == TierDecision.PROMOTED
    assert decision_b == TierDecision.UNCHANGED
    assert await warm_store.contains(_TENANT, user_a, doc)
    assert not await warm_store.contains(_TENANT, user_b, doc)
