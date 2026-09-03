import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.hybrid_search_documents import HybridSearchDocuments

_TENANT = uuid.uuid4()


def _result(chunk_id: uuid.UUID, score: float) -> SearchResult:
    return SearchResult(
        document_id=uuid.uuid4(), chunk_id=chunk_id, content=f"chunk {chunk_id}", score=score
    )


class _FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def execute(self, tenant_id, query, top_k):
        return self._results[:top_k]


async def test_a_chunk_ranked_first_by_both_retrievers_wins_the_merge():
    shared_id = uuid.uuid4()
    vector = _FakeRetriever([_result(shared_id, 0.9), _result(uuid.uuid4(), 0.5)])
    keyword = _FakeRetriever([_result(shared_id, 12.0), _result(uuid.uuid4(), 3.0)])
    hybrid = HybridSearchDocuments(
        vector_retriever=vector, keyword_retriever=keyword, candidate_k=10
    )

    results = await hybrid.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert results[0].chunk_id == shared_id


async def test_a_chunk_found_by_only_one_retriever_still_appears():
    only_vector_id = uuid.uuid4()
    vector = _FakeRetriever([_result(only_vector_id, 0.9)])
    keyword = _FakeRetriever([_result(uuid.uuid4(), 5.0)])
    hybrid = HybridSearchDocuments(
        vector_retriever=vector, keyword_retriever=keyword, candidate_k=10
    )

    results = await hybrid.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert any(r.chunk_id == only_vector_id for r in results)


async def test_respects_top_k_after_merging():
    vector = _FakeRetriever([_result(uuid.uuid4(), 1.0) for _ in range(5)])
    keyword = _FakeRetriever([_result(uuid.uuid4(), 1.0) for _ in range(5)])
    hybrid = HybridSearchDocuments(
        vector_retriever=vector, keyword_retriever=keyword, candidate_k=10
    )

    results = await hybrid.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert len(results) == 3


async def test_a_chunk_ranked_second_by_both_beats_a_chunk_ranked_first_by_only_one():
    # The discriminating case a no-op "just return the vector results" stub
    # would fail (and did, silently, until the _NullDocumentRepository bug
    # this regression test exists for was caught): a chunk at rank 1 in
    # BOTH lists (RRF score 2/62 ~= 0.032258) must outscore a chunk at rank
    # 0 in only the vector list (RRF score 1/61 ~= 0.016393), because RRF
    # rewards agreement across both retrievers, not just vector rank.
    shared_id = uuid.uuid4()
    vector_only_id = uuid.uuid4()
    keyword_only_id = uuid.uuid4()
    vector = _FakeRetriever([_result(vector_only_id, 0.9), _result(shared_id, 0.5)])
    keyword = _FakeRetriever([_result(keyword_only_id, 12.0), _result(shared_id, 3.0)])
    hybrid = HybridSearchDocuments(
        vector_retriever=vector, keyword_retriever=keyword, candidate_k=10
    )

    results = await hybrid.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert results[0].chunk_id == shared_id


async def test_requests_candidate_k_from_both_the_vector_and_keyword_retriever():
    class _SpyRetriever:
        def __init__(self, results: list[SearchResult]) -> None:
            self._results = results
            self.last_top_k: int | None = None

        async def execute(self, tenant_id, query, top_k):
            self.last_top_k = top_k
            return self._results[:top_k]

    vector = _SpyRetriever([_result(uuid.uuid4(), 1.0)])
    keyword = _SpyRetriever([_result(uuid.uuid4(), 1.0)])
    hybrid = HybridSearchDocuments(
        vector_retriever=vector, keyword_retriever=keyword, candidate_k=15
    )

    await hybrid.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert vector.last_top_k == 15
    assert keyword.last_top_k == 15
