from sentence_transformers import CrossEncoder

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(self) -> None:
        self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, r.content) for r in results]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [r for r, _ in ranked[:top_k]]
