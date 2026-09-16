import uuid

from src.orchestration.infrastructure.null_frozen_cache import NullFrozenCache

TENANT = uuid.uuid4()
DOC = uuid.uuid4()


def test_lookup_always_reports_absent():
    cache = NullFrozenCache()
    assert cache.lookup(TENANT, DOC) is None


def test_contains_is_always_false():
    cache = NullFrozenCache()
    assert cache.contains(TENANT, DOC) is False


def test_preload_and_evict_do_not_raise():
    cache = NullFrozenCache()
    cache.preload(TENANT, DOC, "content")
    cache.evict(TENANT, DOC)


def test_preload_until_does_not_raise():
    cache = NullFrozenCache()
    cache.preload_until(TENANT, DOC, "content", expires_at=None)


def test_renew_reports_no_live_entry():
    cache = NullFrozenCache()
    assert cache.renew(TENANT, DOC, expires_at=None) is False
