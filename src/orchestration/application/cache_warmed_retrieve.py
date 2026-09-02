import math
import uuid

from src.orchestration.domain.ports import FrozenCache
from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import EmbeddingModel, Retriever


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class CacheWarmedRetrieve(Retriever):
    """Cache-Warmed RAG's query path: check a small in-memory index over
    only the currently-warmed documents first (cheap, since that candidate
    set is small by construction); on a confident match, confirm against
    FrozenCache (the source of truth for what's actually still cached) and
    answer from there with no vector-store round-trip; otherwise fall
    through to the real fallback Retriever. Implements RAG's own Retriever
    port so it's a drop-in wherever a plain retriever is used today.

    Operates at document granularity, not chunk granularity -- CAG's cache
    holds whole documents, not individual chunks -- so a hit's SearchResult
    reuses document_id as chunk_id; there's no finer split to report.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        frozen_cache: FrozenCache,
        fallback_retriever: Retriever,
        similarity_threshold: float,
    ) -> None:
        self._embedder = embedding_model
        self._frozen_cache = frozen_cache
        self._fallback = fallback_retriever
        self._threshold = similarity_threshold
        self._warmed_content: dict[uuid.UUID, str] = {}
        self._warmed_embeddings: dict[uuid.UUID, list[float]] = {}
        self._hits = 0
        self._misses = 0

    def note_warmed(self, document_id: uuid.UUID, content: str) -> None:
        self._warmed_content[document_id] = content
        self._warmed_embeddings[document_id] = self._embedder.embed(content)

    def stats(self) -> tuple[int, int]:
        """(hits, misses) recorded so far."""
        return self._hits, self._misses

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        hit = self._best_warmed_match(query)
        if hit is not None:
            document_id, score = hit
            cached_hit = self._frozen_cache.lookup(document_id)
            if cached_hit is not None:
                self._hits += 1
                return [
                    SearchResult(
                        document_id=document_id,
                        chunk_id=document_id,
                        content=self._warmed_content[document_id],
                        score=score,
                    )
                ]

        self._misses += 1
        return await self._fallback.execute(tenant_id, query, top_k)

    def _best_warmed_match(self, query: str) -> tuple[uuid.UUID, float] | None:
        if not self._warmed_embeddings:
            return None
        query_embedding = self._embedder.embed(query)
        best_document_id, best_score = None, -1.0
        for document_id, embedding in self._warmed_embeddings.items():
            score = _cosine_similarity(query_embedding, embedding)
            if score > best_score:
                best_document_id, best_score = document_id, score
        if best_document_id is None or best_score < self._threshold:
            return None
        return best_document_id, best_score
