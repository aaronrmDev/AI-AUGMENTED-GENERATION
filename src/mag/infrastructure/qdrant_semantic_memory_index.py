import uuid
from datetime import datetime
from typing import cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.mag.domain.entities import SemanticMemory
from src.mag.domain.ports import SemanticMemoryIndex

_COLLECTION_NAME = "semantic_memory"
_VECTOR_SIZE = 384


class QdrantSemanticMemoryIndex(SemanticMemoryIndex):
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

    async def upsert(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None:
        await self._client.upsert(
            collection_name=_COLLECTION_NAME,
            points=[
                qmodels.PointStruct(
                    id=str(fact.id),
                    vector=fact.embedding,
                    payload={
                        "tenant_id": str(tenant_id),
                        "user_id": str(fact.user_id),
                        "fact_key": fact.fact_key,
                        "fact_value": fact.fact_value,
                        "confidence": fact.confidence,
                        "source": fact.source,
                        "valid_until": fact.valid_until.isoformat() if fact.valid_until else None,
                    },
                )
            ],
        )

    async def search(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[SemanticMemory]:
        response = await self._client.query_points(
            collection_name=_COLLECTION_NAME,
            query=query_embedding,
            query_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id", match=qmodels.MatchValue(value=str(user_id))
                    ),
                    qmodels.FieldCondition(
                        key="tenant_id", match=qmodels.MatchValue(value=str(tenant_id))
                    ),
                ]
            ),
            limit=top_k,
            with_vectors=True,
        )
        results: list[SemanticMemory] = []
        for point in response.points:
            payload = point.payload
            if payload is None:
                continue
            embedding = cast(list[float], point.vector) if point.vector is not None else []
            valid_until_raw = payload.get("valid_until")
            results.append(
                SemanticMemory(
                    id=uuid.UUID(str(point.id)),
                    user_id=uuid.UUID(str(payload["user_id"])),
                    fact_key=str(payload["fact_key"]),
                    fact_value=str(payload["fact_value"]),
                    embedding=embedding,
                    # .get() with the entity's own defaults, not payload[...]
                    # -- a point upserted before confidence/source were added
                    # to this payload (or by any future schema change) reads
                    # back as "unknown," not a KeyError.
                    confidence=float(payload.get("confidence", 1.0)),
                    source=str(payload.get("source", "")),
                    valid_until=(
                        datetime.fromisoformat(valid_until_raw) if valid_until_raw else None
                    ),
                )
            )
        return results
