import threading
from collections import OrderedDict

from src.rag.domain.ports import EmbeddingModel

_DEFAULT_MAX_ENTRIES = 1024


class CachingEmbeddingModel(EmbeddingModel):
    """Remembers recent embeddings by exact text, least recently used first out.

    Built for the orchestration cascade. UnifiedAnswerQuestion embeds each
    question once, and a RAG retriever such as SearchDocuments embeds the same
    text again inside its tier. That second embed is CPU work on the event
    loop, and in PARALLEL routes it blocked the loop long enough to starve the
    CAG tier past its 10ms budget -- measured: 15 of 60 CAG attempts timed out
    with the embedding on the loop, none with it off. Giving both the use case
    and the retriever one shared instance turns the second call into a lookup.

    Thread-safe: CagTier's worker threads and the event loop can both reach it.
    The wrapped model runs outside the lock, so two concurrent misses for the
    same text may both compute it; the result is identical either way.
    Callers get copies, so mutating a returned list never changes the cache.
    """

    def __init__(self, inner: EmbeddingModel, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._inner = inner
        self._max_entries = max_entries
        self._entries: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def embed(self, text: str) -> list[float]:
        with self._lock:
            cached = self._entries.get(text)
            if cached is not None:
                self._entries.move_to_end(text)
                self.hits += 1
                return list(cached)

        embedding = self._inner.embed(text)

        with self._lock:
            self.misses += 1
            self._entries[text] = list(embedding)
            self._entries.move_to_end(text)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return list(embedding)
