import uuid
from datetime import datetime, timedelta

from src.orchestration.application.warm_cache import WarmCache
from src.orchestration.infrastructure.in_memory_access_tracker import (
    InMemoryAccessFrequencyTracker,
)
from tests.unit.orchestration_fakes import FakeFrozenCache

_NOW = datetime(2026, 9, 2, 12, 0, 0)
_WINDOW = timedelta(hours=1)
_TENANT = uuid.uuid4()


def _content_provider_for(contents: dict[uuid.UUID, str]) -> object:
    def provider(document_id: uuid.UUID) -> str:
        return contents[document_id]

    return provider


def test_warms_exactly_the_top_n():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    hot, warm, cold = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    for _ in range(5):
        tracker.record_access(_TENANT, hot, _NOW)
    for _ in range(3):
        tracker.record_access(_TENANT, warm, _NOW)
    tracker.record_access(_TENANT, cold, _NOW)
    contents = {hot: "hot content", warm: "warm content", cold: "cold content"}

    warmed = WarmCache(tracker, cache).execute(
        _TENANT, 2, _WINDOW, _NOW, _content_provider_for(contents)
    )

    assert warmed == [hot, warm]
    assert cache.contains(_TENANT, hot)
    assert cache.contains(_TENANT, warm)
    assert not cache.contains(_TENANT, cold)


def test_skips_documents_already_cached():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    tracker.record_access(_TENANT, doc, _NOW)
    cache.preload(_TENANT, doc, "already here")

    warmed = WarmCache(tracker, cache).execute(
        _TENANT, 1, _WINDOW, _NOW, _content_provider_for({doc: "should not be used"})
    )

    assert warmed == [doc]
    assert cache.preload_calls == [(_TENANT, doc, "already here")]  # no second preload call


def test_empty_tracker_warms_nothing():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()

    warmed = WarmCache(tracker, cache).execute(_TENANT, 5, _WINDOW, _NOW, _content_provider_for({}))

    assert warmed == []
    assert cache.preload_calls == []


def test_warming_is_isolated_per_tenant():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    tracker.record_access(tenant_a, doc, _NOW)

    warmed_a = WarmCache(tracker, cache).execute(
        tenant_a, 5, _WINDOW, _NOW, _content_provider_for({doc: "tenant a's content"})
    )
    warmed_b = WarmCache(tracker, cache).execute(
        tenant_b, 5, _WINDOW, _NOW, _content_provider_for({})
    )

    assert warmed_a == [doc]
    assert warmed_b == []
    assert cache.contains(tenant_a, doc)
    assert not cache.contains(tenant_b, doc)
