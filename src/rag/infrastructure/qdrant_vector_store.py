import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.rag.domain.entities import Chunk, SearchResult
from src.rag.domain.ports import VectorStore

_COLLECTION_NAME = "documents"
_VECTOR_SIZE = 384


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str) -> None:
        self._client = AsyncQdrantClient(url=url)

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(_COLLECTION_NAME)
        if not exists:
            await self._client.create_collection(
                collection_name=_COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=_VECTOR_SIZE, distance=qmodels.Distance.COSINE
                ),
                hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=128),
            )

    async def upsert(self, chunk: Chunk, tenant_id: uuid.UUID) -> None:
        await self._client.upsert(
            collection_name=_COLLECTION_NAME,
            points=[
                qmodels.PointStruct(
                    id=str(chunk.id),
                    vector=chunk.embedding,
                    payload={
                        "tenant_id": str(tenant_id),
                        "document_id": str(chunk.document_id),
                        "content": chunk.content,
                    },
                )
            ],
        )

    async def search(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[SearchResult]:
        response = await self._client.query_points(
            collection_name=_COLLECTION_NAME,
            query=query_embedding,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value=str(tenant_id)))]
            ),
            limit=top_k,
        )
        return [
            SearchResult(
                document_id=uuid.UUID(point.payload["document_id"]),
                chunk_id=uuid.UUID(str(point.id)),
                content=point.payload["content"],
                score=point.score,
            )
            for point in response.points
        ]
