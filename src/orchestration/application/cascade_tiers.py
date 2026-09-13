import asyncio

from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.domain.entities import (
    ContextItem,
    Paradigm,
    TierOutcome,
    TierRequest,
    TierResult,
)
from src.orchestration.domain.ports import CascadeTier
from src.rag.domain.ports import Retriever


def _check_thresholds(hit_threshold: float, partial_threshold: float) -> None:
    if partial_threshold > hit_threshold:
        raise ValueError("partial_threshold cannot exceed hit_threshold")


def _check_top_k(top_k: int) -> None:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")


class CagTier(CascadeTier):
    """The cascade's first tier: a confirmed match against CAG's warmed set."""

    def __init__(
        self,
        cache_warmed_retrieve: CacheWarmedRetrieve,
        *,
        hit_threshold: float,
        partial_threshold: float,
    ) -> None:
        _check_thresholds(hit_threshold, partial_threshold)
        self._retrieve = cache_warmed_retrieve
        self._hit = hit_threshold
        self._partial = partial_threshold

    @property
    def paradigm(self) -> Paradigm:
        return Paradigm.CAG

    async def attempt(self, request: TierRequest) -> TierResult:
        # Matching is CPU work (a cosine pass over the warmed set plus a
        # FrozenCache lookup). Inline, it would hold the event loop and the
        # cascade's 10ms timeout could never fire.
        match = await asyncio.to_thread(
            self._retrieve.best_warmed_match, request.tenant_id, request.query_embedding
        )
        if match is None:
            return TierResult(TierOutcome.MISS)
        result, score = match
        if score < self._partial:
            return TierResult(TierOutcome.MISS)
        item = ContextItem(Paradigm.CAG, result.content, score, result.document_id)
        outcome = TierOutcome.HIT if score >= self._hit else TierOutcome.PARTIAL
        return TierResult(outcome, [item])


class MagTier(CascadeTier):
    """The cascade's second tier: this user's semantic facts.

    Invalidated and archived facts never arrive here --
    PostgresSemanticMemoryRepository.search_by_similarity filters them --
    which is Concept 5's "if MAG has stale state -> invalidate". Like every
    MAG repository, the one behind find_semantic_facts relies on its caller
    having set the transaction-local tenant context (set_tenant_context).
    """

    def __init__(
        self,
        find_semantic_facts: FindSemanticFacts,
        *,
        hit_threshold: float,
        partial_threshold: float,
        top_k: int = 5,
    ) -> None:
        _check_thresholds(hit_threshold, partial_threshold)
        _check_top_k(top_k)
        self._find = find_semantic_facts
        self._hit = hit_threshold
        self._partial = partial_threshold
        self._top_k = top_k

    @property
    def paradigm(self) -> Paradigm:
        return Paradigm.MAG

    async def attempt(self, request: TierRequest) -> TierResult:
        facts = await self._find.by_similarity(
            request.query_embedding, request.user_id, request.tenant_id, self._top_k
        )
        relevant = [scored for scored in facts if scored.score >= self._partial]
        if not relevant:
            return TierResult(TierOutcome.MISS)
        items = [
            ContextItem(
                Paradigm.MAG,
                f"{scored.fact.fact_key}: {scored.fact.fact_value}",
                scored.score,
                scored.fact.id,
            )
            for scored in relevant
        ]
        best = max(scored.score for scored in relevant)
        return TierResult(TierOutcome.HIT if best >= self._hit else TierOutcome.PARTIAL, items)


class RagTier(CascadeTier):
    """The cascade's last resort: any RAG Retriever. RAG has no partial hit."""

    def __init__(self, retriever: Retriever, *, top_k: int = 5) -> None:
        _check_top_k(top_k)
        self._retriever = retriever
        self._top_k = top_k

    @property
    def paradigm(self) -> Paradigm:
        return Paradigm.RAG

    async def attempt(self, request: TierRequest) -> TierResult:
        results = await self._retriever.execute(request.tenant_id, request.query, self._top_k)
        if not results:
            return TierResult(TierOutcome.MISS)
        return TierResult(
            TierOutcome.HIT,
            [ContextItem(Paradigm.RAG, r.content, r.score, r.document_id) for r in results],
        )
