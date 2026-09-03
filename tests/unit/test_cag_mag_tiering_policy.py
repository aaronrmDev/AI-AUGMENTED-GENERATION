import uuid
from datetime import datetime, timedelta

import pytest

from src.orchestration.application.cag_mag_tiering_policy import CagMagTieringPolicy
from src.orchestration.domain import cag_mag_keys
from src.orchestration.domain.entities import TierDecision
from src.orchestration.infrastructure.in_memory_user_scoped_access_tracker import (
    InMemoryUserScopedAccessFrequencyTracker,
)
from tests.unit.orchestration_fakes import FakeFrozenCache

_NOW = datetime(2026, 9, 2, 12, 0, 0)
_WINDOW = timedelta(hours=1)
_PROMOTE = 10
_DEMOTE = 3
_TENANT = uuid.uuid4()
_USER = uuid.uuid4()
_CONTENT_KEY = "preferred_visualization_library"


def _content_provider(text: str = "some content"):
    return lambda mag_content_key: text


def _record_accesses(tracker, tenant_id, user_id, mag_content_key, count, at):
    tracker_id = cag_mag_keys.tracker_key(mag_content_key)
    for _ in range(count):
        tracker.record_access(tenant_id, user_id, tracker_id, at)


def test_promotes_when_threshold_crossed_and_not_yet_hot():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    cache = FakeFrozenCache()
    _record_accesses(tracker, _TENANT, _USER, _CONTENT_KEY, _PROMOTE, _NOW)
    policy = CagMagTieringPolicy(tracker, cache)

    decision = policy.evaluate(
        _TENANT, _USER, _CONTENT_KEY, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision == TierDecision.PROMOTED
    cache_id = cag_mag_keys.cache_key(_USER, _CONTENT_KEY)
    assert cache.contains(_TENANT, cache_id)


def test_no_double_promotion_once_already_hot():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    cache = FakeFrozenCache()
    _record_accesses(tracker, _TENANT, _USER, _CONTENT_KEY, _PROMOTE, _NOW)
    cache_id = cag_mag_keys.cache_key(_USER, _CONTENT_KEY)
    cache.preload(_TENANT, cache_id, "already hot")
    policy = CagMagTieringPolicy(tracker, cache)

    decision = policy.evaluate(
        _TENANT, _USER, _CONTENT_KEY, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision == TierDecision.UNCHANGED
    assert cache.preload_calls == [(_TENANT, cache_id, "already hot")]


def test_demotes_when_access_falls_below_threshold_on_a_hot_entry():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    cache = FakeFrozenCache()
    _record_accesses(tracker, _TENANT, _USER, _CONTENT_KEY, 1, _NOW)  # below demote threshold
    cache_id = cag_mag_keys.cache_key(_USER, _CONTENT_KEY)
    cache.preload(_TENANT, cache_id, "cooling off")
    policy = CagMagTieringPolicy(tracker, cache)

    decision = policy.evaluate(
        _TENANT, _USER, _CONTENT_KEY, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision == TierDecision.DEMOTED
    assert not cache.contains(_TENANT, cache_id)


def test_unchanged_in_the_hysteresis_band_between_thresholds():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    cache = FakeFrozenCache()
    _record_accesses(tracker, _TENANT, _USER, _CONTENT_KEY, 5, _NOW)  # between the two thresholds
    policy = CagMagTieringPolicy(tracker, cache)

    decision = policy.evaluate(
        _TENANT, _USER, _CONTENT_KEY, _content_provider(), _PROMOTE, _DEMOTE, _WINDOW, _NOW
    )

    assert decision == TierDecision.UNCHANGED
    cache_id = cag_mag_keys.cache_key(_USER, _CONTENT_KEY)
    assert not cache.contains(_TENANT, cache_id)


def test_rejects_a_promote_threshold_below_the_demote_threshold():
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    cache = FakeFrozenCache()
    policy = CagMagTieringPolicy(tracker, cache)

    with pytest.raises(ValueError, match="promote_threshold"):
        policy.evaluate(
            _TENANT,
            _USER,
            _CONTENT_KEY,
            _content_provider(),
            demote_threshold=10,
            promote_threshold=3,
            window=_WINDOW,
            now=_NOW,
        )


def test_two_users_promoting_the_same_content_key_never_collide_in_frozen_cache():
    # The exact behavior this batch's whole key-namespacing design exists
    # to guarantee: FrozenCache has no user_id parameter of its own, so
    # cag_mag_keys.cache_key must keep two users' entries for "the same"
    # conceptual content from ever colliding.
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    cache = FakeFrozenCache()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    _record_accesses(tracker, _TENANT, user_a, _CONTENT_KEY, _PROMOTE, _NOW)
    _record_accesses(tracker, _TENANT, user_b, _CONTENT_KEY, _PROMOTE, _NOW)
    policy = CagMagTieringPolicy(tracker, cache)

    decision_a = policy.evaluate(
        _TENANT, user_a, _CONTENT_KEY, _content_provider("user a's content"),
        _PROMOTE, _DEMOTE, _WINDOW, _NOW,
    )
    decision_b = policy.evaluate(
        _TENANT, user_b, _CONTENT_KEY, _content_provider("user b's content"),
        _PROMOTE, _DEMOTE, _WINDOW, _NOW,
    )

    assert decision_a == TierDecision.PROMOTED
    assert decision_b == TierDecision.PROMOTED
    cache_id_a = cag_mag_keys.cache_key(user_a, _CONTENT_KEY)
    cache_id_b = cag_mag_keys.cache_key(user_b, _CONTENT_KEY)
    assert cache_id_a != cache_id_b

    # Both independently readable...
    hit_a = cache.lookup(_TENANT, cache_id_a)
    hit_b = cache.lookup(_TENANT, cache_id_b)
    assert hit_a is not None and hit_b is not None
    assert hit_a.content_hash != hit_b.content_hash  # different content per user

    # ...and independently demotable: demoting user_a's entry directly
    # must never touch user_b's.
    cache.evict(_TENANT, cache_id_a)
    assert not cache.contains(_TENANT, cache_id_a)
    assert cache.contains(_TENANT, cache_id_b)
