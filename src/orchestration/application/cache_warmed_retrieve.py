import math
import uuid

from src.orchestration.domain.ports import FrozenCache
from src.orchestration.domain.sync_mixer import content_hash
from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import EmbeddingModel, Retriever

_Key = tuple[uuid.UUID, uuid.UUID]  # (tenant_id, document_id)


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

    Two review findings shaped the hit path below:

    - FrozenCache staleness: a hit is confirmed only when the cache's own
      real content_hash still matches what note_warmed originally recorded
      locally. Without this check, a document evicted and later
      re-preloaded with DIFFERENT content by TieringPolicy or a later
      WarmCache cycle (both preload directly against FrozenCache, never
      through note_warmed) would be served from this class's own stale
      local memo on what the code itself would record as a "confirmed"
      hit -- exactly the kind of silent staleness the Sync Mixer exists to
      prevent everywhere else in this batch.
    - Tenant isolation: every warmed entry and every candidate match is
      scoped by tenant_id, matching FrozenCache/AccessFrequencyTracker's
      own port contracts. Without this, one shared instance across tenants
      (this project's own established singleton-service DI shape) could
      match tenant B's query against tenant A's warmed content and return
      it, a cross-tenant data leak the moment this module gets wired into
      a real endpoint.
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
        self._warmed_content: dict[_Key, str] = {}
        self._warmed_embeddings: dict[_Key, list[float]] = {}
        self._hits = 0
        self._misses = 0

    def note_warmed(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        key = (tenant_id, document_id)
        self._warmed_content[key] = content
        self._warmed_embeddings[key] = self._embedder.embed(content)

    def stats(self) -> tuple[int, int]:
        """(hits, misses) recorded so far, across all tenants."""
        return self._hits, self._misses

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        hit = self._best_warmed_match(tenant_id, query)
        if hit is not None:
            document_id, score = hit
            local_content = self._warmed_content[(tenant_id, document_id)]
            cached_hit = self._frozen_cache.lookup(tenant_id, document_id)
            if cached_hit is not None and cached_hit.content_hash == content_hash(local_content):
                self._hits += 1
                return [
                    SearchResult(
                        document_id=document_id,
                        chunk_id=document_id,
                        content=local_content,
                        score=score,
                    )
                ]

        self._misses += 1
        return await self._fallback.execute(tenant_id, query, top_k)

    def _best_warmed_match(
        self, tenant_id: uuid.UUID, query: str
    ) -> tuple[uuid.UUID, float] | None:
        candidates = {
            document_id: embedding
            for (tid, document_id), embedding in self._warmed_embeddings.items()
            if tid == tenant_id
        }
        if not candidates:
            return None
        query_embedding = self._embedder.embed(query)
        best_document_id, best_score = None, -1.0
        for document_id, embedding in candidates.items():
            score = _cosine_similarity(query_embedding, embedding)
            if score > best_score:
                best_document_id, best_score = document_id, score
        if best_document_id is None or best_score < self._threshold:
            return None
        return best_document_id, best_score
