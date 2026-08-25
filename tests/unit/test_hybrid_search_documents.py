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
