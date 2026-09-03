import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.reranking_retriever import RerankingRetriever

_TENANT = uuid.uuid4()


def _result(cid: uuid.UUID | None = None) -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=cid or uuid.uuid4(), content="c", score=1.0)


class _FakeInner:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_top_k: int | None = None

    async def execute(self, tenant_id, query, top_k):
        self.last_top_k = top_k
        return self._results[:top_k]


class _ReverseReranker:
    async def rerank(self, query, results, top_k):
        return list(reversed(results))[:top_k]


async def test_asks_the_inner_retriever_for_the_wider_candidate_pool():
    inner = _FakeInner([_result() for _ in range(20)])
    retriever = RerankingRetriever(inner=inner, reranker=_ReverseReranker(), candidate_k=15)

    await retriever.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert inner.last_top_k == 15


async def test_returns_the_rerankers_output_truncated_to_top_k():
    first, second, third = _result(), _result(), _result()
    inner = _FakeInner([first, second, third])
    retriever = RerankingRetriever(inner=inner, reranker=_ReverseReranker(), candidate_k=10)

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=2)

    assert results == [third, second]
