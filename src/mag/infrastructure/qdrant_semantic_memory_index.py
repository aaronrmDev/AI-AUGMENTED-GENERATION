import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.mag.domain.entities import SemanticMemory

_COLLECTION_NAME = "semantic_memory"
_VECTOR_SIZE = 384


class QdrantSemanticMemoryIndex:
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

    async def upsert(self, fact: SemanticMemory) -> None:
        await self._client.upsert(
            collection_name=_COLLECTION_NAME,
            points=[
                qmodels.PointStruct(
                    id=str(fact.id),
                    vector=fact.embedding,
                    payload={
                        "user_id": str(fact.user_id),
                        "fact_key": fact.fact_key,
                        "fact_value": fact.fact_value,
                    },
                )
            ],
        )

    async def search(
        self, query_embedding: list[float], user_id: uuid.UUID, top_k: int
    ) -> list[SemanticMemory]:
        response = await self._client.query_points(
            collection_name=_COLLECTION_NAME,
            query=query_embedding,
            query_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id", match=qmodels.MatchValue(value=str(user_id))
                    )
                ]
            ),
            limit=top_k,
        )
        results: list[SemanticMemory] = []
        for point in response.points:
            # Same reasoning as QdrantVectorStore.search: a payload-less point
            # can't be one of ours (upsert above always writes one), so it's
            # skipped rather than crashing this user's search.
            payload = point.payload
            if payload is None:
                continue
            results.append(
                SemanticMemory(
                    id=uuid.UUID(str(point.id)),
                    user_id=uuid.UUID(str(payload["user_id"])),
                    fact_key=str(payload["fact_key"]),
                    fact_value=str(payload["fact_value"]),
                    # Qdrant doesn't return stored vectors unless with_vectors
                    # is requested; nothing in this batch's search consumers
                    # needs the vector back, matching how
                    # QdrantVectorStore.search's SearchResult carries no
                    # embedding field either.
                    embedding=[],
                )
            )
        return results
