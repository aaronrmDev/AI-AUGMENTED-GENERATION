import threading
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.bi_encoder_rerank_reranker import BiEncoderRerankReranker
from tests.unit.rag_fakes import ThreadRecordingEmbeddingModel


class _TopicEmbedder:
    # Fakes semantic similarity as shared topic words, so the ranking is predictable
    # without a real model in a unit test.
    def embed(self, text: str) -> list[float]:
        words = set(text.lower().split())
        return [1.0 if topic in words else 0.0 for topic in ("refund", "shipping")]


def _result(content: str) -> SearchResult:
    return SearchResult(
        document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content=content, score=0.5
    )


async def test_results_are_reordered_by_the_blended_semantic_and_lexical_score():
    shipping, refund = _result("shipping takes two days"), _result("a refund within thirty days")
    reranker = BiEncoderRerankReranker(embedding_model=_TopicEmbedder())

    ranked = await reranker.rerank(query="refund policy", results=[shipping, refund], top_k=2)

    assert ranked == [refund, shipping]


async def test_only_the_top_k_results_are_returned():
    results = [_result("refund"), _result("shipping"), _result("refund shipping")]
    reranker = BiEncoderRerankReranker(embedding_model=_TopicEmbedder())

    assert len(await reranker.rerank(query="refund", results=results, top_k=2)) == 2


async def test_no_results_rerank_to_none_without_embedding_anything():
    embedder = ThreadRecordingEmbeddingModel(_TopicEmbedder())
    reranker = BiEncoderRerankReranker(embedding_model=embedder)

    assert await reranker.rerank(query="refund", results=[], top_k=3) == []
    assert embedder.thread_ids == []


async def test_every_embedding_runs_off_the_event_loop():
    embedder = ThreadRecordingEmbeddingModel(_TopicEmbedder())
    reranker = BiEncoderRerankReranker(embedding_model=embedder)

    await reranker.rerank(
        query="refund", results=[_result("refund"), _result("shipping")], top_k=2
    )

    assert len(embedder.thread_ids) == 3  # the query and both results
    assert threading.get_ident() not in embedder.thread_ids
