import math
import re

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import EmbeddingModel, Reranker

_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


def _jaccard(a: str, b: str) -> float:
    tokens_a = set(_TOKEN.findall(a.lower()))
    tokens_b = set(_TOKEN.findall(b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class BiEncoderRerankReranker(Reranker):
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._embedder = embedding_model

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []
        query_embedding = self._embedder.embed(query)
        scored = []
        for r in results:
            semantic = _cosine(query_embedding, self._embedder.embed(r.content))
            lexical = _jaccard(query, r.content)
            scored.append((r, 0.7 * semantic + 0.3 * lexical))
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return [r for r, _ in ranked[:top_k]]
