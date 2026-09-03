import uuid

from src.orchestration.application.sync_cycle import SyncCycle
from tests.unit.orchestration_fakes import FakeFrozenCache

_TENANT = uuid.uuid4()


def test_no_conflict_when_content_matches():
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    cache.preload(_TENANT, doc, "price: $80")

    conflicts = SyncCycle(cache).run(_TENANT, [doc], lambda document_id: "price: $80")

    assert conflicts == []
    assert cache.contains(_TENANT, doc)
    assert cache.evict_calls == []


def test_evicts_and_reports_a_conflict_when_rag_content_changed():
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    cache.preload(_TENANT, doc, "price: $100")

    conflicts = SyncCycle(cache).run(_TENANT, [doc], lambda document_id: "price: $80")

    assert len(conflicts) == 1
    assert conflicts[0].document_id == doc
    assert not cache.contains(_TENANT, doc)
    assert cache.evict_calls == [(_TENANT, doc)]


def test_nothing_cached_means_nothing_to_reconcile():
    cache = FakeFrozenCache()
    doc = uuid.uuid4()

    conflicts = SyncCycle(cache).run(_TENANT, [doc], lambda document_id: "anything")

    assert conflicts == []
    assert cache.evict_calls == []


def test_only_the_changed_document_among_several_is_evicted():
    cache = FakeFrozenCache()
    stale, fresh = uuid.uuid4(), uuid.uuid4()
    cache.preload(_TENANT, stale, "old value")
    cache.preload(_TENANT, fresh, "current value")
    authoritative = {stale: "new value", fresh: "current value"}

    conflicts = SyncCycle(cache).run(
        _TENANT, [stale, fresh], lambda document_id: authoritative[document_id]
    )

    assert [c.document_id for c in conflicts] == [stale]
    assert not cache.contains(_TENANT, stale)
    assert cache.contains(_TENANT, fresh)


def test_a_conflict_in_one_tenant_does_not_touch_anothers_cache_entry():
    cache = FakeFrozenCache()
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    cache.preload(tenant_a, doc, "price: $100")
    cache.preload(tenant_b, doc, "price: $100")

    conflicts = SyncCycle(cache).run(tenant_a, [doc], lambda document_id: "price: $80")

    assert len(conflicts) == 1
    assert not cache.contains(tenant_a, doc)
    assert cache.contains(tenant_b, doc)  # untouched -- different tenant, same document_id
