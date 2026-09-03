import uuid

from src.orchestration.application.cag_mag_sync_cycle import CagMagSyncCycle
from src.orchestration.domain import cag_mag_keys
from tests.unit.orchestration_fakes import FakeFrozenCache

_TENANT = uuid.uuid4()
_USER = uuid.uuid4()
_CONTENT_KEY = "preferred_visualization_library"


def _promote(cache: FakeFrozenCache, tenant_id, user_id, mag_content_key, content):
    cache.preload(tenant_id, cag_mag_keys.cache_key(user_id, mag_content_key), content)


def test_no_conflict_when_mags_current_content_matches():
    cache = FakeFrozenCache()
    _promote(cache, _TENANT, _USER, _CONTENT_KEY, "prefers matplotlib")

    conflicts = CagMagSyncCycle(cache).run(
        _TENANT, _USER, [_CONTENT_KEY], lambda key: "prefers matplotlib"
    )

    assert conflicts == []
    cache_id = cag_mag_keys.cache_key(_USER, _CONTENT_KEY)
    assert cache.contains(_TENANT, cache_id)


def test_evicts_and_reports_a_conflict_when_mags_live_content_changed():
    cache = FakeFrozenCache()
    _promote(cache, _TENANT, _USER, _CONTENT_KEY, "prefers matplotlib")

    conflicts = CagMagSyncCycle(cache).run(
        _TENANT, _USER, [_CONTENT_KEY], lambda key: "prefers seaborn"  # MAG's value changed
    )

    assert len(conflicts) == 1
    cache_id = cag_mag_keys.cache_key(_USER, _CONTENT_KEY)
    assert conflicts[0].document_id == cache_id
    assert not cache.contains(_TENANT, cache_id)


def test_nothing_hot_means_nothing_to_reconcile():
    cache = FakeFrozenCache()

    conflicts = CagMagSyncCycle(cache).run(_TENANT, _USER, [_CONTENT_KEY], lambda key: "anything")

    assert conflicts == []


def test_only_the_changed_content_among_several_is_evicted():
    cache = FakeFrozenCache()
    stale_key, fresh_key = "stale_fact", "fresh_fact"
    _promote(cache, _TENANT, _USER, stale_key, "old value")
    _promote(cache, _TENANT, _USER, fresh_key, "current value")
    authoritative = {stale_key: "new value", fresh_key: "current value"}

    conflicts = CagMagSyncCycle(cache).run(
        _TENANT, _USER, [stale_key, fresh_key], lambda key: authoritative[key]
    )

    assert len(conflicts) == 1
    assert not cache.contains(_TENANT, cag_mag_keys.cache_key(_USER, stale_key))
    assert cache.contains(_TENANT, cag_mag_keys.cache_key(_USER, fresh_key))


def test_a_conflict_for_one_user_does_not_touch_anothers_hot_entry():
    cache = FakeFrozenCache()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    _promote(cache, _TENANT, user_a, _CONTENT_KEY, "prefers matplotlib")
    _promote(cache, _TENANT, user_b, _CONTENT_KEY, "prefers matplotlib")

    conflicts = CagMagSyncCycle(cache).run(
        _TENANT, user_a, [_CONTENT_KEY], lambda key: "prefers seaborn"
    )

    assert len(conflicts) == 1
    assert not cache.contains(_TENANT, cag_mag_keys.cache_key(user_a, _CONTENT_KEY))
    assert cache.contains(_TENANT, cag_mag_keys.cache_key(user_b, _CONTENT_KEY))


def test_a_conflict_for_one_tenant_does_not_touch_anothers_hot_entry():
    cache = FakeFrozenCache()
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    _promote(cache, tenant_a, _USER, _CONTENT_KEY, "prefers matplotlib")
    _promote(cache, tenant_b, _USER, _CONTENT_KEY, "prefers matplotlib")

    conflicts = CagMagSyncCycle(cache).run(
        tenant_a, _USER, [_CONTENT_KEY], lambda key: "prefers seaborn"
    )

    assert len(conflicts) == 1
    cache_id = cag_mag_keys.cache_key(_USER, _CONTENT_KEY)
    assert not cache.contains(tenant_a, cache_id)
    assert cache.contains(tenant_b, cache_id)
