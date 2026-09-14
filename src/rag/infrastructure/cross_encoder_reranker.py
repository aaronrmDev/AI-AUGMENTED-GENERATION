import asyncio
from collections.abc import Sequence
from typing import Protocol

from sentence_transformers import CrossEncoder

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import Reranker


class PairScoringModel(Protocol):
    """What CrossEncoderReranker needs from a model: one relevance score per
    (query, passage) pair."""

    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]: ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model: PairScoringModel | None = None) -> None:
        # The default is the real ms-marco MiniLM cross-encoder; tests inject a model.
        self._model: PairScoringModel = (
            model if model is not None else CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        )

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []
        # A forward pass per (query, passage) pair is CPU work, so scoring and sorting run
        # on one worker thread. On the event loop they would stall every other coroutine,
        # which in a PARALLEL cascade route means the CAG and MAG tiers' budgets.
        return await asyncio.to_thread(self._rank, query, results, top_k)

    def _rank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        pairs = [(query, r.content) for r in results]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [r for r, _ in ranked[:top_k]]
