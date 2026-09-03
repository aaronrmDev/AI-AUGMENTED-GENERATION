import math
import uuid

import tiktoken

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import EmbeddingModel, Retriever
from src.rag.infrastructure._sentence_splitter import split_sentences


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


class CompressingRetriever(Retriever):
    def __init__(
        self, inner: Retriever, embedding_model: EmbeddingModel, target_tokens: int = 2000
    ) -> None:
        self._inner = inner
        self._embedder = embedding_model
        self._target_tokens = target_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        results = await self._inner.execute(tenant_id=tenant_id, query=query, top_k=top_k)
        if not results:
            return []

        query_embedding = self._embedder.embed(query)

        # Pool every result's sentences together (not per-result) so
        # selection can remove redundancy ACROSS chunks, not just within
        # one -- matching RAG.md's own "removes duplicate chunks...
        # redundancy removal" framing for what compression is supposed to do.
        scored_sentences: list[tuple[float, int, int, str]] = []  # (score, result_idx, order, text)
        for result_idx, result in enumerate(results):
            for order, sentence in enumerate(split_sentences(result.content)):
                score = _cosine(query_embedding, self._embedder.embed(sentence))
                scored_sentences.append((score, result_idx, order, sentence))

        scored_sentences.sort(key=lambda s: s[0], reverse=True)

        kept_tokens = 0
        kept_by_result: dict[int, list[tuple[int, str]]] = {}
        for _score, result_idx, order, sentence in scored_sentences:
            sentence_tokens = len(self._encoding.encode(sentence))
            if kept_tokens + sentence_tokens > self._target_tokens:
                continue
            kept_tokens += sentence_tokens
            kept_by_result.setdefault(result_idx, []).append((order, sentence))

        compressed: list[SearchResult] = []
        for result_idx, result in enumerate(results):
            kept = kept_by_result.get(result_idx)
            if not kept:
                continue
            kept.sort(key=lambda pair: pair[0])  # restore original sentence order
            compressed.append(
                SearchResult(
                    document_id=result.document_id, chunk_id=result.chunk_id,
                    content=" ".join(sentence for _, sentence in kept), score=result.score,
                )
            )
        return compressed
