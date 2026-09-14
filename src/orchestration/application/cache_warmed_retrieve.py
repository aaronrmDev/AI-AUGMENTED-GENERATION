import uuid

from src.orchestration.domain.ports import FrozenCache
from src.orchestration.domain.similarity import cosine_similarity
from src.orchestration.domain.sync_mixer import content_hash
from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import EmbeddingModel, Retriever

_Warmed = tuple[str, list[float]]  # (content, embedding), stored together


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

    best_warmed_match is public so the orchestration cascade's CagTier can
    reuse the same confirmed matching with its own hit/partial thresholds
    and a query embedding computed once upstream. CagTier calls it on a
    worker thread, so it scans a copy of the tenant's entries: dict.copy()
    is atomic, whereas iterating the live dict raises if note_warmed runs on
    the event loop mid-scan. Entries are indexed by tenant, so a query never
    scans another tenant's warmed set.
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
        self._warmed: dict[uuid.UUID, dict[uuid.UUID, _Warmed]] = {}
        self._hits = 0
        self._misses = 0

    def note_warmed(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        embedding = self._embedder.embed(content)
        self._warmed.setdefault(tenant_id, {})[document_id] = (content, embedding)

    def forget(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        """Drop a document's warmed memo once its cache entry is gone.

        best_warmed_match confirms only its single best candidate, so an evicted
        document left here would keep shadowing every valid warmed document it
        outscores.
        """
        entries = self._warmed.get(tenant_id)
        if entries is not None:
            entries.pop(document_id, None)

    def stats(self) -> tuple[int, int]:
        """(hits, misses) recorded so far, across all tenants."""
        return self._hits, self._misses

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        # Embedding the query only when this tenant has something warmed
        # keeps a cold tenant's queries from paying for a lookup that
        # cannot succeed.
        if tenant_id in self._warmed:
            result = self.best_warmed_match(tenant_id, self._embedder.embed(query))
            if result is not None and result.score >= self._threshold:
                self._hits += 1
                return [result]

        self._misses += 1
        return await self._fallback.execute(tenant_id, query, top_k)

    def best_warmed_match(
        self, tenant_id: uuid.UUID, query_embedding: list[float]
    ) -> SearchResult | None:
        """The closest warmed document for this tenant, scored by similarity
        and confirmed against FrozenCache's real content_hash, with no
        threshold applied -- the caller decides what score counts."""
        entries = self._warmed.get(tenant_id)
        if not entries:
            return None
        best: tuple[uuid.UUID, str, float] | None = None
        for document_id, (content, embedding) in entries.copy().items():
            score = cosine_similarity(query_embedding, embedding)
            if best is None or score > best[2]:
                best = (document_id, content, score)
        if best is None:
            return None

        document_id, content, score = best
        cached_hit = self._frozen_cache.lookup(tenant_id, document_id)
        if cached_hit is None or cached_hit.content_hash != content_hash(content):
            return None
        return SearchResult(
            document_id=document_id, chunk_id=document_id, content=content, score=score
        )
