import uuid
from datetime import UTC, datetime
from typing import cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.mag.domain.entities import ScoredFact, SemanticMemory
from src.mag.domain.ports import SemanticMemoryIndex

_COLLECTION_NAME = "semantic_memory"
_VECTOR_SIZE = 384


def _epoch(value: datetime | None) -> float | None:
    # Qdrant's Range filter only compares numbers -- the ISO-string
    # payload fields (kept for readability / round-tripping back into a
    # SemanticMemory) can't be range-compared server-side, so every
    # timestamp field this index writes also gets a numeric-epoch mirror
    # search() filters on.
    return value.timestamp() if value is not None else None


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
                        "valid_until_epoch": _epoch(fact.valid_until),
                        "archived_at": (
                            fact.archived_at.isoformat() if fact.archived_at else None
                        ),
                        "archived_at_epoch": _epoch(fact.archived_at),
                    },
                )
            ],
        )

    async def update_status(
        self,
        fact_id: uuid.UUID,
        tenant_id: uuid.UUID,
        valid_until: datetime | None,
        archived_at: datetime | None,
    ) -> None:
        # set_payload merges the given keys into the existing point's
        # payload and leaves the vector (and every other payload field)
        # untouched -- unlike upsert(), which always replaces the whole
        # point. InvalidateMemory/ArchiveMemory only ever have an
        # embedding-less SemanticMemory to work with (find_by_key never
        # returns a real embedding -- Postgres isn't this system's
        # embedding-bearing read path), so going through upsert() here
        # would silently blank out the stored vector.
        await self._client.set_payload(
            collection_name=_COLLECTION_NAME,
            payload={
                "valid_until": valid_until.isoformat() if valid_until else None,
                "valid_until_epoch": _epoch(valid_until),
                "archived_at": archived_at.isoformat() if archived_at else None,
                "archived_at_epoch": _epoch(archived_at),
            },
            points=[str(fact_id)],
        )

    async def search(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredFact]:
        now_epoch = datetime.now(UTC).timestamp()
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
                    # Not archived (MAG Batch F, #64): archived_at_epoch
                    # must be missing or null. IsEmptyCondition, not
                    # IsNullCondition -- it matches both "field explicitly
                    # null" and "field never written" alike, so a point
                    # from before this field existed is treated as
                    # not-archived rather than excluded by an
                    # over-narrow check.
                    qmodels.IsEmptyCondition(
                        is_empty=qmodels.PayloadField(key="archived_at_epoch")
                    ),
                    # Not expired (#63): valid_until_epoch is missing/null,
                    # OR set to a time still in the future.
                    qmodels.Filter(
                        should=[
                            qmodels.IsEmptyCondition(
                                is_empty=qmodels.PayloadField(key="valid_until_epoch")
                            ),
                            qmodels.FieldCondition(
                                key="valid_until_epoch", range=qmodels.Range(gt=now_epoch)
                            ),
                        ]
                    ),
                ]
            ),
            limit=top_k,
            with_vectors=True,
        )
        results: list[ScoredFact] = []
        for point in response.points:
            payload = point.payload
            if payload is None:
                continue
            embedding = cast(list[float], point.vector) if point.vector is not None else []
            valid_until_raw = payload.get("valid_until")
            archived_at_raw = payload.get("archived_at")
            fact = SemanticMemory(
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
                archived_at=(
                    datetime.fromisoformat(archived_at_raw) if archived_at_raw else None
                ),
            )
            results.append(ScoredFact(fact=fact, score=point.score))
        return results
