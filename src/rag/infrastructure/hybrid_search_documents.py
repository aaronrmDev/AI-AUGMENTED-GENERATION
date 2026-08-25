import asyncio
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import Retriever

_RRF_K = 60


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

        rrf_scores: dict[uuid.UUID, float] = {}
        by_id: dict[uuid.UUID, SearchResult] = {}
        for result_list in (vector_results, keyword_results):
            for rank, result in enumerate(result_list):
                rrf_scores[result.chunk_id] = rrf_scores.get(
                    result.chunk_id, 0.0
                ) + 1.0 / (_RRF_K + rank + 1)
                by_id[result.chunk_id] = result

        merged_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        return [
            SearchResult(
                document_id=by_id[cid].document_id,
                chunk_id=cid,
                content=by_id[cid].content,
                score=rrf_scores[cid],
            )
            for cid in merged_ids[:top_k]
        ]
