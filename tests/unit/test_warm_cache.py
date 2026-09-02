import uuid
from datetime import datetime, timedelta

from src.orchestration.application.warm_cache import WarmCache
from src.orchestration.infrastructure.in_memory_access_tracker import (
    InMemoryAccessFrequencyTracker,
)
from tests.unit.orchestration_fakes import FakeFrozenCache

_NOW = datetime(2026, 9, 2, 12, 0, 0)
_WINDOW = timedelta(hours=1)


def _content_provider_for(contents: dict[uuid.UUID, str]) -> object:
    def provider(document_id: uuid.UUID) -> str:
        return contents[document_id]

    return provider


def test_warms_exactly_the_top_n():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    hot, warm, cold = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    for _ in range(5):
        tracker.record_access(hot, _NOW)
    for _ in range(3):
        tracker.record_access(warm, _NOW)
    tracker.record_access(cold, _NOW)
    contents = {hot: "hot content", warm: "warm content", cold: "cold content"}

    warmed = WarmCache(tracker, cache).execute(2, _WINDOW, _NOW, _content_provider_for(contents))

    assert warmed == [hot, warm]
    assert cache.contains(hot)
    assert cache.contains(warm)
    assert not cache.contains(cold)


def test_skips_documents_already_cached():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()
    doc = uuid.uuid4()
    tracker.record_access(doc, _NOW)
    cache.preload(doc, "already here")

    warmed = WarmCache(tracker, cache).execute(
        1, _WINDOW, _NOW, _content_provider_for({doc: "should not be used"})
    )

    assert warmed == [doc]
    assert cache.preload_calls == [(doc, "already here")]  # no second preload call


def test_empty_tracker_warms_nothing():
    tracker = InMemoryAccessFrequencyTracker()
    cache = FakeFrozenCache()

    warmed = WarmCache(tracker, cache).execute(5, _WINDOW, _NOW, _content_provider_for({}))

    assert warmed == []
    assert cache.preload_calls == []
