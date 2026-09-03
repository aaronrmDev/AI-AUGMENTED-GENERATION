import asyncio
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import Retriever
from src.rag.infrastructure._result_fusion import reciprocal_rank_fusion


class HybridSearchDocuments(Retriever):
    def __init__(
        self, vector_retriever: Retriever, keyword_retriever: Retriever, candidate_k: int = 20
    ) -> None:
        self._vector = vector_retriever
        self._keyword = keyword_retriever
        self._candidate_k = candidate_k

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        vector_results, keyword_results = await asyncio.gather(
            self._vector.execute(tenant_id=tenant_id, query=query, top_k=self._candidate_k),
            self._keyword.execute(tenant_id=tenant_id, query=query, top_k=self._candidate_k),
        )
        return reciprocal_rank_fusion([vector_results, keyword_results], top_k=top_k)
