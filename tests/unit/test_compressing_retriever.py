import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.compressing_retriever import CompressingRetriever


class _FakeInner:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def execute(self, tenant_id, query, top_k):
        return self._results[:top_k]


class _KeywordOverlapEmbedder:
    # A fake that fakes "semantic similarity" as literal word overlap with
    # the query -- enough to test the pooling/greedy-selection arithmetic
    # without needing a real model in a unit test.
    def embed(self, text: str) -> list[float]:
        words = set(text.lower().split())
        vocab = ["query", "relevant", "irrelevant", "padding"]
        return [1.0 if w in words else 0.0 for w in vocab]


async def test_keeps_the_query_relevant_sentence_and_drops_the_irrelevant_one():
    results = [
        SearchResult(
            document_id=uuid.uuid4(), chunk_id=uuid.uuid4(),
            content="This sentence answers the query directly. This sentence is irrelevant padding.",
            score=0.9,
        )
    ]
    retriever = CompressingRetriever(
        inner=_FakeInner(results), embedding_model=_KeywordOverlapEmbedder(), target_tokens=10
    )

    compressed = await retriever.execute(tenant_id=uuid.uuid4(), query="query relevant", top_k=1)

    assert "answers the query directly" in compressed[0].content
    assert "irrelevant padding" not in compressed[0].content


async def test_a_result_contributing_zero_kept_sentences_is_dropped():
    results = [
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="Totally irrelevant padding here.", score=0.9),
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="This directly answers the query relevant to it.", score=0.8),
    ]
    # target_tokens=9, not the plan's literal 8: "This directly answers the
    # query relevant to it." encodes to exactly 9 cl100k_base tokens. At 8,
    # the greedy budget pass skips that highest-scoring sentence entirely
    # (0 + 9 > 8) and falls through to keep the lower-scoring 6-token
    # "Totally irrelevant padding here." instead -- inverting this test's
    # intent (verified empirically: the plan's literal target_tokens=8 fails
    # with compressed[0].content == "Totally irrelevant padding here.").
    # 9 is the smallest budget that keeps the relevant sentence whole while
    # still excluding the irrelevant one (9 + 6 = 15 > 9).
    retriever = CompressingRetriever(
        inner=_FakeInner(results), embedding_model=_KeywordOverlapEmbedder(), target_tokens=9
    )

    compressed = await retriever.execute(tenant_id=uuid.uuid4(), query="query relevant", top_k=2)

    assert len(compressed) == 1
    assert "answers the query" in compressed[0].content
