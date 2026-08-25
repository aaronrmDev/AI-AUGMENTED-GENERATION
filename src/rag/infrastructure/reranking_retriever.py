import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import Reranker, Retriever


class RerankingRetriever(Retriever):
    def __init__(self, inner: Retriever, reranker: Reranker, candidate_k: int = 20) -> None:
        self._inner = inner
        self._reranker = reranker
        self._candidate_k = candidate_k

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        candidates = await self._inner.execute(
            tenant_id=tenant_id, query=query, top_k=self._candidate_k
        )
        return await self._reranker.rerank(query=query, results=candidates, top_k=top_k)
