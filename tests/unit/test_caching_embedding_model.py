import threading

import pytest

from src.rag.domain.ports import EmbeddingModel
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel


class _CountingEmbedder(EmbeddingModel):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def embed(self, text: str) -> list[float]:
        with self._lock:
            self.calls.append(text)
        return [float(len(text)), float(sum(map(ord, text)) % 97)]


def test_the_same_text_is_embedded_once_and_then_served_from_the_cache():
    inner = _CountingEmbedder()
    model = CachingEmbeddingModel(inner)

    first = model.embed("what changed today?")
    second = model.embed("what changed today?")

    assert first == second
    assert inner.calls == ["what changed today?"]
    assert (model.hits, model.misses) == (1, 1)


def test_different_texts_are_each_embedded():
    inner = _CountingEmbedder()
    model = CachingEmbeddingModel(inner)
    model.embed("a")
    model.embed("b")
    assert inner.calls == ["a", "b"]


def test_the_least_recently_used_entry_is_evicted_at_capacity():
    inner = _CountingEmbedder()
    model = CachingEmbeddingModel(inner, max_entries=2)
    model.embed("a")
    model.embed("b")
    model.embed("a")  # touches a, so b is now the least recently used
    model.embed("c")  # evicts b

    model.embed("a")
    model.embed("b")

    assert inner.calls == ["a", "b", "c", "b"]


def test_mutating_a_returned_embedding_never_changes_what_the_cache_serves():
    model = CachingEmbeddingModel(_CountingEmbedder())
    returned = model.embed("x")
    expected = list(returned)
    returned.append(99.0)
    returned[0] = -1.0

    assert model.embed("x") == expected
    cached = model.embed("x")
    cached[0] = -2.0
    assert model.embed("x") == expected


def test_concurrent_callers_all_get_correct_embeddings():
    # CagTier's worker threads and the event loop can call one shared instance.
    inner = _CountingEmbedder()
    model = CachingEmbeddingModel(inner, max_entries=4)
    texts = [f"question {i}" for i in range(10)]
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(50):
                for text in texts:
                    assert model.embed(text) == inner.embed(text)
        except BaseException as exc:  # surfaced below; a thread can't fail the test itself
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []


def test_a_non_positive_capacity_is_rejected():
    with pytest.raises(ValueError):
        CachingEmbeddingModel(_CountingEmbedder(), max_entries=0)
