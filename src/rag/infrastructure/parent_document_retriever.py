import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import DocumentRepository, Retriever


class ParentDocumentRetriever(Retriever):
    def __init__(self, inner: Retriever, document_repository: DocumentRepository) -> None:
        self._inner = inner
        self._documents = document_repository

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        # Requests top_k, not a wider candidate pool: per the design spec's
        # resolved composition order, expansion operates on the
        # already-final ranked set, not on a pool that gets narrowed later.
        child_results = await self._inner.execute(tenant_id=tenant_id, query=query, top_k=top_k)

        expanded: list[SearchResult] = []
        seen_parent_ids: set[uuid.UUID] = set()
        for child in child_results:
            child_chunk = await self._documents.get_chunk_by_id(child.chunk_id)
            if child_chunk is None or child_chunk.parent_id is None:
                continue
            if child_chunk.parent_id in seen_parent_ids:
                continue
            parent_chunk = await self._documents.get_chunk_by_id(child_chunk.parent_id)
            if parent_chunk is None:
                continue
            seen_parent_ids.add(child_chunk.parent_id)
            expanded.append(
                SearchResult(
                    document_id=child.document_id, chunk_id=child_chunk.parent_id,
                    content=parent_chunk.content, score=child.score,
                )
            )
        return expanded
