import uuid
from datetime import UTC, datetime, timedelta

from src.orchestration.infrastructure.expiring_frozen_cache import ExpiringFrozenCache
from tests.unit.orchestration_fakes import FakeFrozenCache

T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
TENANT, OTHER_TENANT, DOC = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _cache() -> tuple[ExpiringFrozenCache, FakeFrozenCache, _Clock]:
    inner, clock = FakeFrozenCache(), _Clock(T0)
    return ExpiringFrozenCache(inner, clock), inner, clock


def test_an_entry_is_served_until_its_expiry_and_absent_from_that_instant():
    cache, inner, clock = _cache()
    cache.preload_until(TENANT, DOC, "text", T0 + HOUR)

    clock.now = T0 + HOUR - timedelta(microseconds=1)
    assert cache.contains(TENANT, DOC)
    assert cache.lookup(TENANT, DOC) is not None

    clock.now = T0 + HOUR
    assert cache.lookup(TENANT, DOC) is None
    assert not cache.contains(TENANT, DOC)
    # Evicted from the wrapped cache the first time the expiry was seen, and only once.
    assert inner.evict_calls == [(TENANT, DOC)]


def test_a_plain_preload_and_a_none_expiry_never_expire():
    cache, _, clock = _cache()
    cache.preload(TENANT, DOC, "text")
    other = uuid.uuid4()
    cache.preload_until(TENANT, other, "text", None)
    clock.now = T0 + timedelta(days=3650)
    assert cache.contains(TENANT, DOC)
    assert cache.contains(TENANT, other)


def test_renewing_a_live_entry_moves_its_expiry():
    cache, _, clock = _cache()
    cache.preload_until(TENANT, DOC, "text", T0 + HOUR)
    assert cache.renew(TENANT, DOC, T0 + 2 * HOUR) is True
    clock.now = T0 + timedelta(minutes=90)
    assert cache.contains(TENANT, DOC)


def test_renewing_a_missing_or_expired_entry_changes_nothing():
    cache, _, clock = _cache()
    assert cache.renew(TENANT, DOC, T0 + HOUR) is False
    cache.preload_until(TENANT, DOC, "text", T0 + HOUR)
    clock.now = T0 + HOUR
    assert cache.renew(TENANT, DOC, T0 + 5 * HOUR) is False
    assert not cache.contains(TENANT, DOC)


def test_re_preloading_replaces_the_expiry():
    cache, _, clock = _cache()
    cache.preload_until(TENANT, DOC, "v1", T0 + HOUR)
    cache.preload_until(TENANT, DOC, "v2", None)
    clock.now = T0 + 10 * HOUR
    assert cache.contains(TENANT, DOC)


def test_evict_removes_the_entry_and_its_expiry():
    cache, inner, _ = _cache()
    cache.preload_until(TENANT, DOC, "text", T0 + HOUR)
    cache.evict(TENANT, DOC)
    assert not cache.contains(TENANT, DOC)
    assert not inner.contains(TENANT, DOC)


def test_expiry_is_scoped_by_tenant():
    cache, _, clock = _cache()
    cache.preload_until(TENANT, DOC, "text", T0 + HOUR)
    cache.preload_until(OTHER_TENANT, DOC, "text", None)
    clock.now = T0 + HOUR
    assert not cache.contains(TENANT, DOC)
    assert cache.contains(OTHER_TENANT, DOC)


class _PreloadHookCache(FakeFrozenCache):
    """Runs a callback after storing a preload, to interleave another caller mid-preload."""

    def __init__(self) -> None:
        super().__init__()
        self.during_preload = lambda: None

    def preload(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        super().preload(tenant_id, document_id, content)
        self.during_preload()


def test_an_expiry_check_during_a_re_preload_does_not_evict_the_new_entry():
    inner, clock = _PreloadHookCache(), _Clock(T0)
    cache = ExpiringFrozenCache(inner, clock)
    cache.preload_until(TENANT, DOC, "v1", T0 + HOUR)
    clock.now = T0 + 2 * HOUR  # v1 has expired, but nobody has looked at it yet
    seen_mid_preload: list[bool] = []
    inner.during_preload = lambda: seen_mid_preload.append(cache.contains(TENANT, DOC))

    cache.preload_until(TENANT, DOC, "v2", T0 + 5 * HOUR)

    assert seen_mid_preload == [True]
    assert cache.contains(TENANT, DOC)
