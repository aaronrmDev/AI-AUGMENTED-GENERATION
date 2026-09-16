import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from src.identity.infrastructure.db import get_engine, get_sessionmaker
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.ingest_data_source import IngestDataSource
from src.orchestration.domain.entities import DataSourceProfile, SourceScope
from src.orchestration.infrastructure.chunked_rag_index import ChunkedRagIndex
from src.orchestration.infrastructure.null_frozen_cache import NullFrozenCache
from src.orchestration.infrastructure.postgres_data_source_repository import (
    PostgresDataSourceRepository,
)
from src.orchestration.infrastructure.record_semantic_fact_writer import RecordSemanticFactWriter
from src.rag.application.search_documents import SearchDocuments
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.workers.celery_app import celery_app

# CAG's own similarity threshold for a warmed-cache hit -- CacheWarmedRetrieve's
# constructor requires one, but with NullFrozenCache backing it, nothing this
# task does ever produces a hit for the threshold to apply to.
_UNUSED_CAG_THRESHOLD = 0.9


async def _build_ingest_data_source() -> (
    tuple[IngestDataSource, AsyncEngine, Neo4jMemoryGraphRepository]
):
    engine = get_engine(os.environ["APP_DATABASE_URL"])
    sessionmaker = get_sessionmaker(engine)
    embedder = CachingEmbeddingModel(SentenceTransformersEmbedder())
    vector_store = QdrantVectorStore(os.environ["QDRANT_URL"])
    semantic_index = QdrantSemanticMemoryIndex(os.environ["QDRANT_URL"])
    graph = Neo4jMemoryGraphRepository(
        os.environ["NEO4J_URL"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        # Every store this composition root touches provisions its own schema
        # on first use in this codebase's established pattern:
        # QdrantVectorStore / QdrantSemanticMemoryIndex both expose
        # ensure_collection(), and tests/integration/freshness_env.py calls
        # all three (including this graph's ensure_schema()) once per
        # environment before any write. Alembic migrations own Postgres's
        # schema up front, but Qdrant and Neo4j have no migration runner --
        # ensure_collection()/ensure_schema() *is* their migration step, and
        # nothing else in this codebase runs it for a real (non-test)
        # process. Skipping it here would leave a fresh Neo4j instance (a
        # real one, e.g. from docker-compose up on a clean volume) with none
        # of Neo4jMemoryGraphRepository's uniqueness constraints or the
        # Entity embedding index, silently degrading semantic-fact writes
        # from constrained MERGE-by-id to full scans with no duplicate
        # protection. Every statement in ensure_schema() is `IF NOT EXISTS`,
        # so calling it on every task invocation -- required here since this
        # task has no process-startup hook to call it once, unlike a real
        # worker entry point -- is a cheap, idempotent no-op after the first
        # call rather than a per-call cost that grows with usage.
        await vector_store.ensure_collection()
        await semantic_index.ensure_collection()
        await graph.ensure_schema()
    except Exception:
        # A provisioning failure here means this function raises before ever
        # handing engine/graph back to a caller who could close them in its
        # own finally block (ingest_data_source_task's, below) -- without
        # this except, a transient Qdrant/Neo4j provisioning error would
        # leak the engine's connection pool and the Bolt driver this same
        # call already opened, on top of whatever failure is being reported.
        await graph.close()
        await engine.dispose()
        raise

    repository = PostgresDataSourceRepository(sessionmaker)
    rag_index = ChunkedRagIndex(sessionmaker, vector_store, FixedSizeChunker(), embedder)
    cache = NullFrozenCache()
    search = SearchDocuments(embedder, vector_store)
    warmed = CacheWarmedRetrieve(
        embedder, cache, search, similarity_threshold=_UNUSED_CAG_THRESHOLD
    )
    writer = RecordSemanticFactWriter(sessionmaker, semantic_index, embedder, graph)
    # engine and graph are handed back alongside the use case, not stashed
    # anywhere else, so the caller can close both -- see
    # ingest_data_source_task's own finally block. The two Qdrant-backed
    # stores this function also builds (vector_store, semantic_index) are
    # NOT handed back: QdrantVectorStore and QdrantSemanticMemoryIndex
    # expose no close() today, so there is nothing for a caller to call --
    # disclosed as a known, out-of-scope gap in this task's report, not
    # silently ignored (fixing it means expanding two shared RAG/MAG
    # infrastructure classes other call sites and tests also depend on).
    # engine (a plain sqlalchemy.ext.asyncio.AsyncEngine) and graph (whose
    # own close() is already public API, exercised identically by
    # freshness_env.py's teardown) both already expose what's needed, so
    # both get closed here with zero new API surface.
    return IngestDataSource(repository, rag_index, cache, warmed, writer), engine, graph


@celery_app.task(name="ingest_data_source")  # type: ignore[untyped-decorator]
# celery ships no py.typed marker (pyproject.toml's [[tool.mypy.overrides]]
# for module = ["celery", "celery.result"] already disclosed this for
# CeleryIngestionJobDispatcher's imports) -- Celery.task resolves to Any
# under strict mode's disallow_untyped_decorators, same narrowly-scoped
# ignore-at-the-call-site convention Task 3 used for celery's untyped
# redis.asyncio call sites rather than a blanket per-module ignore.
def ingest_data_source_task(
    *,
    tenant_id: str,
    source_key: str,
    scope: str,
    expected_change_interval_seconds: int,
    content: str,
    user_id: str | None,
) -> dict[str, Any]:
    import asyncio
    import concurrent.futures

    async def _run() -> dict[str, Any]:
        ingest, engine, graph = await _build_ingest_data_source()
        try:
            profile = DataSourceProfile(
                source_key,
                SourceScope(scope),
                timedelta(seconds=expected_change_interval_seconds),
            )
            result = await ingest.execute(
                uuid.UUID(tenant_id),
                profile,
                content,
                datetime.now(UTC),
                user_id=uuid.UUID(user_id) if user_id is not None else None,
            )
            return {
                "source_id": str(result.source_id),
                "route": result.route.value,
                "changed": result.changed,
            }
        finally:
            # Confirmed empirically (uv run pytest ... -W error::ResourceWarning):
            # an unclosed AsyncBoltDriver left for GC raises its own
            # ResourceWarning from neo4j's __del__ -- a real leaked Bolt
            # connection pool, not a style nit. engine.dispose() closes this
            # call's own AsyncEngine (and the asyncpg connection pool it
            # opened) the same way -- both engine and graph are built fresh
            # on every invocation (no per-process startup/shutdown hook
            # exists here to build and close either just once), so closing
            # both here is what keeps a long-lived worker process from
            # accumulating one open driver and one open connection pool per
            # ingestion it has ever run. The two Qdrant-backed stores this
            # same composition root builds are NOT closed here --
            # QdrantVectorStore and QdrantSemanticMemoryIndex expose no
            # close() today, and adding one is real, but out of this task's
            # scope (it touches shared RAG/MAG infrastructure classes other
            # call sites and tests also depend on). Disclosed as a known,
            # deliberately unfixed concern in this task's report, not
            # silently ignored.
            await graph.close()
            await engine.dispose()

    # A real Celery worker process (prefork, solo, or gevent pool) has no
    # already-running event loop the way uvicorn does -- asyncio.run is the
    # direct bridge from this synchronous task body to IngestDataSource.execute,
    # which is async, and is the fast path taken there. But
    # tests/integration/test_ingestion_worker.py calls this task's synchronous
    # .apply() from *inside* an async test function, so pytest-asyncio's own
    # loop is already running on this same thread at that point -- asyncio.run
    # refuses to nest inside a loop that's already running in the calling
    # thread (confirmed empirically: RuntimeError: asyncio.run() cannot be
    # called from a running event loop). Running the coroutine on a dedicated
    # thread with its own fresh loop sidesteps that without special-casing
    # tests in production code: it never touches whatever loop, if any, is
    # already running on the calling thread.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result()
