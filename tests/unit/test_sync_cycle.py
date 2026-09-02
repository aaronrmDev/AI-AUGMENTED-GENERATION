import uuid

from src.orchestration.application.sync_cycle import SyncCycle
from tests.unit.orchestration_fakes import FakeFrozenCache


def test_no_conflict_when_content_matches():
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    cache.preload(doc, "price: $80")

    conflicts = SyncCycle(cache).run([doc], lambda document_id: "price: $80")

    assert conflicts == []
    assert cache.contains(doc)
    assert cache.evict_calls == []


def test_evicts_and_reports_a_conflict_when_rag_content_changed():
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    cache.preload(doc, "price: $100")

    conflicts = SyncCycle(cache).run([doc], lambda document_id: "price: $80")

    assert len(conflicts) == 1
    assert conflicts[0].document_id == doc
    assert not cache.contains(doc)
    assert cache.evict_calls == [doc]


def test_nothing_cached_means_nothing_to_reconcile():
    cache = FakeFrozenCache()
    doc = uuid.uuid4()

    conflicts = SyncCycle(cache).run([doc], lambda document_id: "anything")

    assert conflicts == []
    assert cache.evict_calls == []


def test_only_the_changed_document_among_several_is_evicted():
    cache = FakeFrozenCache()
    stale, fresh = uuid.uuid4(), uuid.uuid4()
    cache.preload(stale, "old value")
    cache.preload(fresh, "current value")
    authoritative = {stale: "new value", fresh: "current value"}

    conflicts = SyncCycle(cache).run(
        [stale, fresh], lambda document_id: authoritative[document_id]
    )

    assert [c.document_id for c in conflicts] == [stale]
    assert not cache.contains(stale)
    assert cache.contains(fresh)
