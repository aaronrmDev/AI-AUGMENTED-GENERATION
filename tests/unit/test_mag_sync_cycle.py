import uuid

from src.orchestration.application.mag_sync_cycle import MagSyncCycle
from tests.unit.orchestration_fakes import FakeWarmStore

_TENANT = uuid.uuid4()
_USER = uuid.uuid4()


async def test_no_conflict_when_content_matches():
    warm_store = FakeWarmStore()
    doc = uuid.uuid4()
    await warm_store.promote(_TENANT, _USER, doc, "price: $80")

    conflicts = await MagSyncCycle(warm_store).run(
        _TENANT, _USER, [doc], lambda document_id: "price: $80"
    )

    assert conflicts == []
    assert await warm_store.contains(_TENANT, _USER, doc)
    assert warm_store.demote_calls == []


async def test_demotes_and_reports_a_conflict_when_rag_content_changed():
    warm_store = FakeWarmStore()
    doc = uuid.uuid4()
    await warm_store.promote(_TENANT, _USER, doc, "price: $100")

    conflicts = await MagSyncCycle(warm_store).run(
        _TENANT, _USER, [doc], lambda document_id: "price: $80"
    )

    assert len(conflicts) == 1
    assert conflicts[0].document_id == doc
    assert not await warm_store.contains(_TENANT, _USER, doc)
    assert warm_store.demote_calls == [(_TENANT, _USER, doc)]


async def test_nothing_warm_means_nothing_to_reconcile():
    warm_store = FakeWarmStore()
    doc = uuid.uuid4()

    conflicts = await MagSyncCycle(warm_store).run(
        _TENANT, _USER, [doc], lambda document_id: "anything"
    )

    assert conflicts == []
    assert warm_store.demote_calls == []


async def test_only_the_changed_document_among_several_is_demoted():
    warm_store = FakeWarmStore()
    stale, fresh = uuid.uuid4(), uuid.uuid4()
    await warm_store.promote(_TENANT, _USER, stale, "old value")
    await warm_store.promote(_TENANT, _USER, fresh, "current value")
    authoritative = {stale: "new value", fresh: "current value"}

    conflicts = await MagSyncCycle(warm_store).run(
        _TENANT, _USER, [stale, fresh], lambda document_id: authoritative[document_id]
    )

    assert [c.document_id for c in conflicts] == [stale]
    assert not await warm_store.contains(_TENANT, _USER, stale)
    assert await warm_store.contains(_TENANT, _USER, fresh)


async def test_a_conflict_for_one_user_does_not_touch_anothers_warm_entry():
    warm_store = FakeWarmStore()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    await warm_store.promote(_TENANT, user_a, doc, "price: $100")
    await warm_store.promote(_TENANT, user_b, doc, "price: $100")

    conflicts = await MagSyncCycle(warm_store).run(
        _TENANT, user_a, [doc], lambda document_id: "price: $80"
    )

    assert len(conflicts) == 1
    assert not await warm_store.contains(_TENANT, user_a, doc)
    assert await warm_store.contains(_TENANT, user_b, doc)  # untouched -- different user


async def test_a_conflict_for_one_tenant_does_not_touch_anothers_warm_entry():
    # Parallels test_sync_cycle.py's (Batch D, RAG-vs-CAG) own tenant-
    # isolation test -- this file previously only exercised the user axis.
    warm_store = FakeWarmStore()
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    await warm_store.promote(tenant_a, _USER, doc, "price: $100")
    await warm_store.promote(tenant_b, _USER, doc, "price: $100")

    conflicts = await MagSyncCycle(warm_store).run(
        tenant_a, _USER, [doc], lambda document_id: "price: $80"
    )

    assert len(conflicts) == 1
    assert not await warm_store.contains(tenant_a, _USER, doc)
    assert await warm_store.contains(tenant_b, _USER, doc)  # untouched -- different tenant
