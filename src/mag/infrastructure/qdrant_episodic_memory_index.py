import uuid
from datetime import datetime
from typing import cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryIndex

_COLLECTION_NAME = "episodic_memory"
_VECTOR_SIZE = 384


class QdrantEpisodicMemoryIndex(EpisodicMemoryIndex):
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

    async def upsert(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None:
        await self._client.upsert(
            collection_name=_COLLECTION_NAME,
            points=[
                qmodels.PointStruct(
                    id=str(episode.id),
                    vector=episode.embedding,
                    payload={
                        "tenant_id": str(tenant_id),
                        "session_id": str(episode.session_id),
                        "content": episode.content,
                        "timestamp": episode.timestamp.isoformat(),
                        "salience_score": episode.salience_score,
                    },
                )
            ],
        )

    async def search(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredEpisode]:
        # Design choice the next batch's retrieval strategies build on: this
        # index returns full EpisodicMemory objects reconstructed from payload
        # + vector (mirroring QdrantVectorStore, which inlines chunk content
        # into SearchResult rather than a bare id/score pair), so a strategy
        # that needs the whole episode never has to round-trip back to
        # Postgres just to get it. A caller that only needs a fast existence
        # or ranking check can ignore the extra fields.
        response = await self._client.query_points(
            collection_name=_COLLECTION_NAME,
            query=query_embedding,
            query_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="tenant_id", match=qmodels.MatchValue(value=str(tenant_id))
                    )
                ]
            ),
            limit=top_k,
            with_vectors=True,
        )
        results: list[ScoredEpisode] = []
        for point in response.points:
            payload = point.payload
            if payload is None:
                continue
            # ensure_collection only ever configures a single, unnamed vector
            # for this collection, so the client's broader named/sparse/multi
            # -vector return type never actually resolves to those other
            # branches here -- narrowing by hand is what that general return
            # type can't do for us.
            embedding = cast(list[float], point.vector) if point.vector is not None else []
            episode = EpisodicMemory(
                id=uuid.UUID(str(point.id)),
                session_id=uuid.UUID(str(payload["session_id"])),
                content=payload["content"],
                embedding=embedding,
                timestamp=datetime.fromisoformat(str(payload["timestamp"])),
                salience_score=float(payload["salience_score"]),
                # consolidated_at defaults to None -- upsert() above
                # never writes it to the payload. See
                # EpisodicMemoryIndex.search's docstring for why.
            )
            # point.score is cosine similarity for this COSINE-distance
            # collection -- the same quantity search_by_similarity's port
            # docstring above documents Postgres computing, on purpose, so
            # fusion across backends isn't mixing two conventions.
            results.append(ScoredEpisode(episode=episode, score=point.score))
        return results
