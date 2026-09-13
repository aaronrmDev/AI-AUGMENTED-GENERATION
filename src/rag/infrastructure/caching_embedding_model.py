import hashlib
import threading
from collections import OrderedDict

from src.rag.domain.ports import EmbeddingModel

_DEFAULT_MAX_ENTRIES = 1024


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


class CachingEmbeddingModel(EmbeddingModel):
    """Remembers recent embeddings by exact text, least recently used first out.

    Built for the orchestration cascade. UnifiedAnswerQuestion embeds each
    question once, and a RAG retriever such as SearchDocuments embeds the same
    text again inside its tier. That second embed is CPU work on the event
    loop, and in PARALLEL routes it blocked the loop long enough to starve the
    CAG tier past its 10ms budget -- measured: 15 of 60 CAG attempts timed out
    with the embedding on the loop, none with it off. Giving both the use case
    and the retriever one shared instance turns the second call into a lookup.

    One instance serves every tenant in the process. An embedding is a pure
    function of its text, so a cached value reveals nothing a fresh one
    wouldn't, and entries are keyed by a SHA-256 digest, so the cache never
    holds the text anyone asked. A hit is still faster than a miss, which a
    caller able to time requests precisely could read as "someone asked this
    exact text recently". Nothing exposes this path over HTTP yet; weighing
    that signal belongs to the security review of the endpoint that does.

    Thread-safe: UnifiedAnswerQuestion embeds on a worker thread while a
    retriever embeds on the event loop. The wrapped model runs outside the
    lock, so two concurrent misses for the same text may both compute it; the
    result is identical either way. Callers get copies, so mutating a returned
    list never changes the cache.
    """

    def __init__(self, inner: EmbeddingModel, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._inner = inner
        self._max_entries = max_entries
        self._entries: OrderedDict[bytes, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def embed(self, text: str) -> list[float]:
        key = _digest(text)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return list(cached)

        embedding = self._inner.embed(text)

        with self._lock:
            self.misses += 1
            self._entries[key] = list(embedding)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return list(embedding)
