"""Real-store composition for the freshness router's end-to-end tests:
- Postgres with RLS, and Qdrant;
- MiniLM behind CachingEmbeddingModel;
- a distilgpt2 HFFrozenCache behind ExpiringFrozenCache, on a clock the test moves;
- MAG writes through RecordSemanticFact with Neo4j;
- Batch A's unrouted cascade to ask questions. Concept 5 as drawn trusts a CAG hit,
  which is exactly what stale placement exploits.

Not collected by pytest (no test_ prefix)."""
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from evaluation.scenarios.orchestration_meta_layer_thresholds import (
    CAG_HIT,
    CAG_PARTIAL,
    MAG_HIT,
    MAG_PARTIAL,
)
from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.cascade_tiers import CagTier, MagTier, RagTier
from src.orchestration.application.ingest_data_source import IngestDataSource
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.refresh_cached_sources import RefreshCachedSources
from src.orchestration.application.review_source_freshness import ReviewSourceFreshness
from src.orchestration.domain.entities import (
    CascadeResult,
    DataSourceProfile,
    SourceScope,
    TierRequest,
)
from src.orchestration.infrastructure.chunked_rag_index import ChunkedRagIndex
from src.orchestration.infrastructure.expiring_frozen_cache import ExpiringFrozenCache
from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache
from src.orchestration.infrastructure.postgres_data_source_repository import (
    PostgresDataSourceRepository,
)
from src.orchestration.infrastructure.record_semantic_fact_writer import RecordSemanticFactWriter
from src.orchestration.infrastructure.session_scoped_semantic_fact_search import (
    SessionScopedSemanticFactSearch,
)
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.ports import EmbeddingModel
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)

PRICES = DataSourceProfile("backpack-price", SourceScope.TENANT, HOUR)
PRICE_QUESTION = "How much does the blue hiking backpack cost today?"


def price_text(dollars: int) -> str:
    return f"The blue hiking backpack costs {dollars} dollars today."


POLICY = DataSourceProfile("return-policy", SourceScope.TENANT, 90 * DAY)
POLICY_QUESTION = "What is the return policy for unopened items?"


def policy_text(days: str) -> str:
    return (
        f"Our return policy allows customers to return unopened items within {days} days "
        "of purchase for a full refund."
    )


SIZE = DataSourceProfile("size-preference", SourceScope.USER, 30 * DAY)
SIZE_QUESTION = "What shoe size do I wear?"
SIZE_TEXT = "The user wears size 10 running shoes."

# Generous on purpose: these tests check placement, not latency.
CORRECTNESS_TIMEOUTS = TierTimeouts(cag=5.0, mag=5.0, rag=10.0)


class SimulatedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@dataclass
class FreshnessEnv:
    tenant_id: uuid.UUID
    owner_id: uuid.UUID
    other_user_id: uuid.UUID
    clock: SimulatedClock
    embedder: EmbeddingModel
    ingest: IngestDataSource
    refresh: RefreshCachedSources
    review: ReviewSourceFreshness
    cascade: LatencyCascade

    async def ask(self, question: str, user_id: uuid.UUID | None = None) -> CascadeResult:
        request = TierRequest(
            self.tenant_id, user_id or self.owner_id, uuid.uuid4(), question,
            self.embedder.embed(question),
        )
        return await self.cascade.run(request)


async def create_user(db_session: Any, tenant_id: uuid.UUID) -> uuid.UUID:
    await set_tenant_context(db_session, tenant_id)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id) "
            "VALUES (:id, :email, :hashed_password, :tenant_id)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id,
        },
    )
    await db_session.commit()
    return user_id


@asynccontextmanager
async def freshness_env(
    db_session: Any,
    qdrant_url: str,
    neo4j_url: tuple[str, str, str],
    embedding_model: EmbeddingModel,
    tokenizer: Any,
    model: Any,
) -> AsyncIterator[FreshnessEnv]:
    tenant_id = uuid.uuid4()
    owner_id = await create_user(db_session, tenant_id)
    other_user_id = await create_user(db_session, tenant_id)
    sessionmaker = get_sessionmaker(db_session.bind)
    embedder = CachingEmbeddingModel(embedding_model)
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    semantic_index = QdrantSemanticMemoryIndex(qdrant_url)
    await semantic_index.ensure_collection()
    url, username, password = neo4j_url
    graph = Neo4jMemoryGraphRepository(url, auth=(username, password))
    await graph.ensure_schema()

    clock = SimulatedClock(T0)
    cache = ExpiringFrozenCache(HFFrozenCache(tokenizer=tokenizer, model=model), clock)
    search = SearchDocuments(embedder, vector_store)
    warmed = CacheWarmedRetrieve(embedder, cache, search, similarity_threshold=CAG_HIT)
    repository = PostgresDataSourceRepository(sessionmaker)
    rag_index = ChunkedRagIndex(sessionmaker, vector_store, FixedSizeChunker(), embedder)
    writer = RecordSemanticFactWriter(sessionmaker, semantic_index, embedder, graph)
    cascade = LatencyCascade(
        [
            CagTier(warmed, hit_threshold=CAG_HIT, partial_threshold=CAG_PARTIAL),
            MagTier(
                SessionScopedSemanticFactSearch(sessionmaker),
                hit_threshold=MAG_HIT,
                partial_threshold=MAG_PARTIAL,
            ),
            RagTier(search, top_k=2),
        ],
        CORRECTNESS_TIMEOUTS,
    )
    try:
        yield FreshnessEnv(
            tenant_id=tenant_id,
            owner_id=owner_id,
            other_user_id=other_user_id,
            clock=clock,
            embedder=embedder,
            ingest=IngestDataSource(repository, rag_index, cache, warmed, writer),
            refresh=RefreshCachedSources(repository, cache, warmed),
            review=ReviewSourceFreshness(repository, cache, warmed),
            cascade=cascade,
        )
    finally:
        await cascade.drain()
        await graph.close()
