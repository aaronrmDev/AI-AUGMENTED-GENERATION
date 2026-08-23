import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import EmbeddingModel, VectorStore


class SearchDocuments:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore) -> None:
        self._embedder = embedding_model
        self._vector_store = vector_store

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        query_embedding = self._embedder.embed(query)
        return await self._vector_store.search(query_embedding, tenant_id=tenant_id, top_k=top_k)
