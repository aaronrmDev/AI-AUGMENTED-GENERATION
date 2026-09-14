"""Freshness-Aware Data Router measurements, on a simulated clock against real stores.

Parts (select with --parts):
- spectrum: Concept 9's eight data types, declared as profiles and routed.
- ablation: 30 simulated days, six sources, four placement arms. Every source is
  ingested hourly; refresh (and review, in the freshness-aware arm) runs at
  midnight; every source is probed every 3 hours through Batch A's unrouted
  cascade, which trusts a CAG hit exactly as Concept 5 draws it.
- ttl: a cached source changed in RAG behind the router's back while its feed has
  stalled, with a TTL and without one.

Starts its own Postgres, Qdrant, and Neo4j containers. No LLM: staleness is judged
on the assembled context, using version-specific markers. Before any arm runs, the
questions are checked against the tier thresholds, so a wording problem stops the
run instead of skewing it. Not pytest-collected.

Usage (from the repository root):
    PYTHONPATH=. python evaluation/scenarios/freshness-router/run_freshness_measurements.py
"""
import argparse
import asyncio
import io
import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.qdrant import QdrantContainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from alembic import command
from evaluation.domain.freshness_metrics import (
    MigrationObservation,
    PreloadTracker,
    ProbeObservation,
    RouteRow,
    tally_probes,
)
from evaluation.infrastructure.freshness_report import render_freshness_measurements
from evaluation.scenarios.orchestration_meta_layer_thresholds import (
    CAG_HIT,
    CAG_PARTIAL,
    MAG_HIT,
    MAG_PARTIAL,
)
from src.identity.infrastructure.db import get_engine, get_sessionmaker, set_tenant_context
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.cascade_tiers import CagTier, MagTier, RagTier
from src.orchestration.application.ingest_data_source import IngestDataSource, RoutePolicy
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.refresh_cached_sources import RefreshCachedSources
from src.orchestration.application.review_source_freshness import ReviewSourceFreshness
from src.orchestration.domain.entities import (
    DataSourceProfile,
    FreshnessPolicy,
    IngestionRoute,
    Paradigm,
    SourceScope,
    TierRequest,
)
from src.orchestration.domain.freshness_router import route_for, source_id_for
from src.orchestration.domain.similarity import cosine_similarity
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
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder

_REPORT = Path("evaluation/reports/freshness-router-measurements.md")
_PARTS = ("spectrum", "ablation", "ttl")
_APP_DB_PASSWORD = "evaluation-only-app-user-password"
_SEED_PASSWORD_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
T0 = datetime(2026, 1, 1, tzinfo=UTC)  # day 1, 00:00
HOUR, DAY = timedelta(hours=1), timedelta(days=1)
HORIZON_HOURS = 30 * 24
PROBE_EVERY_HOURS = 3
_TIMEOUTS = TierTimeouts(cag=5.0, mag=5.0, rag=10.0)  # placement, not latency, is measured


def _hours(now: datetime) -> int:
    return int((now - T0) / HOUR)


@dataclass(frozen=True)
class Feed:
    profile: DataSourceProfile
    question: str
    version_at: Callable[[datetime], int]
    text: Callable[[int], str]
    marker: Callable[[int], str]


def _catalog_version(now: datetime) -> int:
    hours = _hours(now)  # daily until day 10 (hour 216), hourly after
    return hours // 24 if hours < 216 else 9 + (hours - 216)


_POLICY_DAYS = ("forty-five", "sixty")
FEEDS = [
    Feed(
        DataSourceProfile("backpack-price", SourceScope.TENANT, HOUR),
        "How much does the blue hiking backpack cost today?",
        _hours,
        lambda n: f"The blue hiking backpack costs {40 + n} dollars today.",
        lambda n: f"costs {40 + n} dollars",
    ),
    Feed(
        DataSourceProfile("return-policy", SourceScope.TENANT, 90 * DAY),
        "What is the return policy for unopened items?",
        lambda now: int(now >= T0 + 11 * DAY + 9 * HOUR),  # day 12, 09:00
        lambda n: (
            "Our return policy allows customers to return unopened items within "
            f"{_POLICY_DAYS[n]} days of purchase for a full refund."
        ),
        lambda n: f"within {_POLICY_DAYS[n]} days",
    ),
    Feed(
        # Not shoes: a shoe catalog scored 0.49 against the size-preference question.
        DataSourceProfile("blender-catalog", SourceScope.TENANT, 7 * DAY),
        "How many kitchen blender models does the catalog list?",
        _catalog_version,
        lambda n: f"The kitchen blender catalog lists {100 + n} models this season.",
        lambda n: f"lists {100 + n} models",
    ),
    Feed(
        DataSourceProfile("shipping-guide", SourceScope.TENANT, 365 * DAY),
        "How long does standard shipping take?",
        lambda now: 0,
        lambda n: "Standard shipping takes five to seven business days.",
        lambda n: "five to seven",
    ),
    Feed(
        DataSourceProfile("flash-sale", SourceScope.TENANT, HOUR),
        "Which code gives the flash sale discount on garden hoses?",
        lambda now: min(_hours(now), 96),  # hourly until day 5 (hour 96), then quiet
        lambda n: f"Flash sale code SALE{n} takes 20 percent off every garden hose.",
        lambda n: f"SALE{n} ",
    ),
]
SIZE_FEED = Feed(
    DataSourceProfile("size-preference", SourceScope.USER, 30 * DAY),
    "What shoe size do I wear?",
    lambda now: int(now >= T0 + 14 * DAY + 9 * HOUR),  # day 15, 09:00
    lambda n: f"The user wears size {10 + n} running shoes.",
    lambda n: f"size {10 + n} ",
)
_WARRANTY_YEARS = ("two", "three")
WARRANTY = Feed(
    DataSourceProfile("warranty-terms", SourceScope.TENANT, 7 * DAY),
    # Not "product": that wording scored 0.36 against the return-policy text.
    "How long is the bicycle frame warranty?",
    lambda now: int(now >= T0 + 7 * DAY),  # day 8: changed in RAG behind the router's back
    lambda n: f"Every bicycle carries a {_WARRANTY_YEARS[n]}-year limited frame warranty.",
    lambda n: f"{_WARRANTY_YEARS[n]}-year",
)
# When each drifting source's real behaviour changed, for the migration-lag table.
SHIFTS = {"blender-catalog": T0 + 9 * DAY, "flash-sale": T0 + 4 * DAY}
SPECTRUM = [
    ("Stock prices, live scores", "seconds", SourceScope.TENANT, timedelta(seconds=1), "RAG"),
    ("News, social media", "minutes-hours", SourceScope.TENANT, HOUR, "RAG"),
    ("User session state", "every turn", SourceScope.USER, timedelta(minutes=1), "MAG"),
    ("Product catalog", "daily-weekly", SourceScope.TENANT, 3 * DAY, "CAG + RAG hybrid"),
    ("Company policies, manuals", "monthly-quarterly", SourceScope.TENANT, 60 * DAY, "CAG"),
    ("Textbooks, reference", "never", SourceScope.TENANT, 3650 * DAY, "CAG"),
    ("Code repositories", "per commit", SourceScope.TENANT, 4 * HOUR, "CAG + RAG"),
    ("User long-term preferences", "gradually", SourceScope.USER, 30 * DAY, "MAG"),
]
# Spec decision 2: the source's CAG rows are the stable route, which keeps RAG as backup.
_SOURCE_ROUTE = {
    "RAG": IngestionRoute.RAG_ONLY,
    "MAG": IngestionRoute.MAG,
    "CAG": IngestionRoute.CAG_WITH_RAG_BACKUP,
    "CAG + RAG hybrid": IngestionRoute.CAG_WITH_RAG_BACKUP,
    "CAG + RAG": IngestionRoute.CAG_WITH_RAG_BACKUP,
}


def _always_cag(profile: DataSourceProfile, policy: FreshnessPolicy) -> IngestionRoute:
    return IngestionRoute.CAG_WITH_RAG_BACKUP


def _always_rag(profile: DataSourceProfile, policy: FreshnessPolicy) -> IngestionRoute:
    return IngestionRoute.RAG_ONLY


NAIVE = "cache everything, batch refresh only"
ARMS: list[tuple[str, RoutePolicy, bool]] = [
    (NAIVE, _always_cag, False),
    ("cache everything, invalidate on change", _always_cag, False),
    ("RAG only", _always_rag, False),
    ("freshness-aware", route_for, True),
]


class SimulatedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@dataclass
class Stores:
    sessionmaker: async_sessionmaker[AsyncSession]
    vector_store: QdrantVectorStore
    semantic_index: QdrantSemanticMemoryIndex
    graph: Neo4jMemoryGraphRepository
    embedder: CachingEmbeddingModel
    tokenizer: Any
    model: Any


@dataclass
class Rig:
    name: str
    tenant_id: uuid.UUID
    owner_id: uuid.UUID
    other_id: uuid.UUID
    clock: SimulatedClock
    cache: ExpiringFrozenCache
    warmed: CacheWarmedRetrieve
    rag_index: ChunkedRagIndex
    cascade: LatencyCascade
    ingest: IngestDataSource
    refresh: RefreshCachedSources
    review: ReviewSourceFreshness | None


async def _build_rig(
    stores: Stores,
    name: str,
    route_policy: RoutePolicy,
    review: bool,
    policy: FreshnessPolicy | None = None,
) -> Rig:
    tenant_id, owner_id, other_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with stores.sessionmaker() as session, session.begin():
        await set_tenant_context(session, tenant_id)
        for user_id in (owner_id, other_id):
            await session.execute(
                text(
                    "INSERT INTO users (id, email, hashed_password, tenant_id) "
                    "VALUES (:id, :email, :hashed_password, :tenant_id)"
                ),
                {
                    "id": user_id, "email": f"{user_id}@example.com",
                    "hashed_password": _SEED_PASSWORD_HASH, "tenant_id": tenant_id,
                },
            )
    effective = policy or FreshnessPolicy()
    clock = SimulatedClock(T0)
    cache = ExpiringFrozenCache(
        HFFrozenCache(tokenizer=stores.tokenizer, model=stores.model), clock
    )
    search = SearchDocuments(stores.embedder, stores.vector_store)
    warmed = CacheWarmedRetrieve(stores.embedder, cache, search, similarity_threshold=CAG_HIT)
    repository = PostgresDataSourceRepository(stores.sessionmaker)
    rag_index = ChunkedRagIndex(
        stores.sessionmaker, stores.vector_store, FixedSizeChunker(), stores.embedder
    )
    writer = RecordSemanticFactWriter(
        stores.sessionmaker, stores.semantic_index, stores.embedder, stores.graph
    )
    cascade = LatencyCascade(
        [
            CagTier(warmed, hit_threshold=CAG_HIT, partial_threshold=CAG_PARTIAL),
            MagTier(
                SessionScopedSemanticFactSearch(stores.sessionmaker),
                hit_threshold=MAG_HIT,
                partial_threshold=MAG_PARTIAL,
            ),
            RagTier(search, top_k=2),
        ],
        _TIMEOUTS,
    )
    return Rig(
        name=name,
        tenant_id=tenant_id,
        owner_id=owner_id,
        other_id=other_id,
        clock=clock,
        cache=cache,
        warmed=warmed,
        rag_index=rag_index,
        cascade=cascade,
        ingest=IngestDataSource(
            repository, rag_index, cache, warmed, writer, policy=effective,
            route_policy=route_policy,
        ),
        refresh=RefreshCachedSources(repository, cache, warmed, policy=effective),
        review=(
            ReviewSourceFreshness(repository, cache, warmed, policy=effective) if review else None
        ),
    )


class NaivePlacement:
    """The batch-refresh-only baseline: every source's latest text goes to RAG at once,
    and into CAG only at the nightly re-warm, with no invalidation in between."""

    def __init__(self, rig: Rig) -> None:
        self._rig = rig
        self._latest: dict[uuid.UUID, tuple[str, str]] = {}

    async def ingest(self, feed: Feed, content: str, user_id: uuid.UUID | None) -> None:
        key = feed.profile.source_key
        source_id = source_id_for(self._rig.tenant_id, key, user_id)
        if self._latest.get(source_id, ("", ""))[1] != content:
            await self._rig.rag_index.replace(self._rig.tenant_id, source_id, key, content)
            self._latest[source_id] = (key, content)

    def rewarm(self) -> list[str]:
        for source_id, (_, content) in self._latest.items():
            self._rig.cache.preload(self._rig.tenant_id, source_id, content)
            self._rig.warmed.note_warmed(self._rig.tenant_id, source_id, content)
        return [key for key, _ in self._latest.values()]


def _probe_rows(source_key: str) -> list[str]:
    if source_key == SIZE_FEED.profile.source_key:
        return [f"{source_key} (owner)", f"{source_key} (other user)"]
    return [source_key]


async def _probe(
    stores: Stores,
    rig: Rig,
    feed: Feed,
    now: datetime,
    user_id: uuid.UUID,
    row: str,
    foreign: tuple[str, ...] = (),
) -> ProbeObservation:
    version = feed.version_at(now)
    request = TierRequest(
        rig.tenant_id, user_id, uuid.uuid4(), feed.question, stores.embedder.embed(feed.question)
    )
    result = await rig.cascade.run(request)
    return ProbeObservation(
        arm=rig.name,
        source_key=row,
        context="\n".join(item.content for item in result.items),
        current_marker=feed.marker(version),
        superseded_markers=tuple(feed.marker(v) for v in range(version)),
        served_from_cag=any(item.paradigm is Paradigm.CAG for item in result.items),
        foreign_markers=foreign,
    )


async def _run_ablation_arm(
    stores: Stores,
    arm: tuple[str, RoutePolicy, bool],
    tracker: PreloadTracker,
    migrations: list[MigrationObservation],
) -> list[ProbeObservation]:
    name, route_policy, review = arm
    rig = await _build_rig(stores, name, route_policy, review)
    naive = NaivePlacement(rig) if name == NAIVE else None
    size_markers = (SIZE_FEED.marker(0), SIZE_FEED.marker(1))
    owner_row, other_row = _probe_rows(SIZE_FEED.profile.source_key)
    observations: list[ProbeObservation] = []
    for tick in range(HORIZON_HOURS + 1):
        now = T0 + tick * HOUR
        rig.clock.now = now
        feeds: list[tuple[Feed, uuid.UUID | None]] = [(feed, None) for feed in FEEDS]
        feeds.append((SIZE_FEED, rig.owner_id))
        for feed, user_id in feeds:
            content = feed.text(feed.version_at(now))
            if naive is not None:
                await naive.ingest(feed, content, user_id)
            else:
                await rig.ingest.execute(
                    rig.tenant_id, feed.profile, content, now, user_id=user_id
                )
        if now.hour == 0:
            if rig.review is not None:
                for m in await rig.review.run(rig.tenant_id, now):
                    migrations.append(
                        MigrationObservation(
                            m.source_key, m.from_route.value, m.to_route.value,
                            SHIFTS.get(m.source_key, now), now,
                        )
                    )
            if naive is not None:
                keys = naive.rewarm()
            else:
                keys = await rig.refresh.run(rig.tenant_id, now)
            for key in keys:
                for row in _probe_rows(key):
                    tracker.preloaded(name, row)
        if tick % PROBE_EVERY_HOURS == 0:
            probes: list[ProbeObservation] = []
            for feed in FEEDS:
                probes.append(
                    await _probe(stores, rig, feed, now, rig.owner_id, feed.profile.source_key)
                )
            probes.append(await _probe(stores, rig, SIZE_FEED, now, rig.owner_id, owner_row))
            probes.append(
                await _probe(stores, rig, SIZE_FEED, now, rig.other_id, other_row, size_markers)
            )
            for observation in probes:
                if observation.served_from_cag:
                    tracker.served(name, observation.source_key)
            observations.extend(probes)
    await rig.cascade.drain()
    return observations


async def _run_ttl_arm(
    stores: Stores, name: str, policy: FreshnessPolicy
) -> list[ProbeObservation]:
    rig = await _build_rig(stores, name, route_for, review=False, policy=policy)
    source_id = source_id_for(rig.tenant_id, WARRANTY.profile.source_key, None)
    observations: list[ProbeObservation] = []
    for tick in range(HORIZON_HOURS + 1):
        now = T0 + tick * HOUR
        rig.clock.now = now
        if now.hour == 0 and now <= T0 + 6 * DAY:  # confirmed daily through day 7, then stalls
            await rig.ingest.execute(rig.tenant_id, WARRANTY.profile, WARRANTY.text(0), now)
        if now == T0 + 7 * DAY:  # day 8: the text changes in RAG behind the router's back
            await rig.rag_index.replace(
                rig.tenant_id, source_id, WARRANTY.profile.source_key, WARRANTY.text(1)
            )
        if now.hour == 0:
            await rig.refresh.run(rig.tenant_id, now)
        if tick % PROBE_EVERY_HOURS == 0:
            observations.append(
                await _probe(stores, rig, WARRANTY, now, rig.owner_id, WARRANTY.profile.source_key)
            )
    await rig.cascade.drain()
    return observations


def _spectrum() -> list[RouteRow]:
    policy = FreshnessPolicy()
    rows: list[RouteRow] = []
    for data_type, change, scope, interval, paradigm in SPECTRUM:
        route = route_for(DataSourceProfile(data_type, scope, interval), policy)
        rows.append(
            RouteRow(
                data_type, change, scope.value, paradigm, route.value,
                route is _SOURCE_ROUTE[paradigm],
            )
        )
    return rows


def _calibrate(embedder: CachingEmbeddingModel) -> str:
    """Each question must reach its own source's text at both tiers' hit thresholds,
    and stay below the partial thresholds against every other source."""
    feeds = [*FEEDS, SIZE_FEED, WARRANTY]
    texts = {f.profile.source_key: embedder.embed(f.text(0)) for f in feeds}
    hit, partial = max(CAG_HIT, MAG_HIT), min(CAG_PARTIAL, MAG_PARTIAL)
    lines: list[str] = []
    failed = False
    for feed in feeds:
        question = embedder.embed(feed.question)
        scores = {key: cosine_similarity(question, vector) for key, vector in texts.items()}
        own = scores.pop(feed.profile.source_key)
        best_other = max(scores.values())
        failed = failed or own < hit or best_other >= partial
        lines.append(f"{feed.profile.source_key}: own {own:.2f}, best other {best_other:.2f}")
    if failed:
        raise SystemExit(
            f"question wording fails calibration (need own >= {hit}, other < {partial}):\n"
            + "\n".join(lines)
        )
    return "; ".join(lines)


async def _measure(
    app_url: str, qdrant_url: str, neo4j: tuple[str, str, str], parts: frozenset[str]
) -> None:
    engine = get_engine(app_url)
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    semantic_index = QdrantSemanticMemoryIndex(qdrant_url)
    await semantic_index.ensure_collection()
    graph = Neo4jMemoryGraphRepository(neo4j[0], auth=(neo4j[1], neo4j[2]))
    await graph.ensure_schema()
    stores = Stores(
        sessionmaker=get_sessionmaker(engine),
        vector_store=vector_store,
        semantic_index=semantic_index,
        graph=graph,
        embedder=CachingEmbeddingModel(SentenceTransformersEmbedder()),
        tokenizer=AutoTokenizer.from_pretrained("distilgpt2"),
        model=AutoModelForCausalLM.from_pretrained("distilgpt2"),
    )
    calibration = _calibrate(stores.embedder)
    print(f"calibration: {calibration}")
    routes = _spectrum() if "spectrum" in parts else []
    tracker = PreloadTracker()
    migrations: list[MigrationObservation] = []
    observations: list[ProbeObservation] = []
    if "ablation" in parts:
        for arm in ARMS:
            observations.extend(await _run_ablation_arm(stores, arm, tracker, migrations))
            print(f"arm done: {arm[0]}")
    tallies = tally_probes(observations)
    ttl_observations: list[ProbeObservation] = []
    if "ttl" in parts:
        ttl_observations += await _run_ttl_arm(stores, "TTL factor 0.5", FreshnessPolicy())
        ttl_observations += await _run_ttl_arm(stores, "no TTL", FreshnessPolicy(ttl_factor=None))
    notes = (
        f"Simulated clock from {T0:%Y-%m-%d} (day 1) for 30 days. Every source is ingested "
        f"hourly. Refresh runs at midnight, and review too in the freshness-aware arm. Every "
        f"source is probed every {PROBE_EVERY_HOURS} hours through Batch A's unrouted cascade "
        f"(CAG -> MAG -> RAG, first hit wins), with tier thresholds carried over unchanged: CAG "
        f"{CAG_HIT}/{CAG_PARTIAL}, MAG {MAG_HIT}/{MAG_PARTIAL}. Staleness is read from the "
        "assembled context using version-specific markers; no LLM runs. Schedules: prices "
        "change hourly; the return policy changes once, day 12 09:00; the catalog changes "
        "daily and hourly from day 10; the shipping guide never changes; the flash sale "
        "changes hourly until day 5 and then goes quiet; the owner's size preference changes "
        "day 15 09:00 and is probed by its owner and by another user of the same tenant. TTL "
        "part: warranty terms declared at 7 days, confirmed daily through day 7, changed "
        "directly in RAG on day 8. Question calibration: " + calibration + ". The change "
        "schedules are synthetic, one author wrote them and the rules, the cache is a CPU "
        "distilgpt2 proxy, and tier latency is cited from Batch A rather than re-measured "
        "here."
    )
    preloads = {(t.arm, t.source_key): tracker.counts(t.arm, t.source_key) for t in tallies}
    _REPORT.write_text(
        render_freshness_measurements(
            routes, tallies, preloads, migrations, tally_probes(ttl_observations), notes
        ),
        encoding="utf-8",
    )
    print(_REPORT.read_text(encoding="utf-8"))
    await graph.close()
    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freshness-Aware Data Router measurements.")
    parser.add_argument(
        "--parts", default=",".join(_PARTS), help="comma-separated subset of: " + ", ".join(_PARTS)
    )
    parts = frozenset(
        part.strip() for part in parser.parse_args().parts.split(",") if part.strip()
    )
    unknown = parts - set(_PARTS)
    if unknown:
        parser.error(f"unknown parts: {sorted(unknown)}")
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # the report contains non-cp1252 characters
    with (
        PostgresContainer("pgvector/pgvector:pg16") as postgres,
        QdrantContainer("qdrant/qdrant:v1.16.2") as qdrant,
        Neo4jContainer("neo4j:5-community") as neo4j,
    ):
        # 127.0.0.1, not "localhost": asyncpg via "localhost" stalls on this Windows host.
        database_url = (
            make_url(postgres.get_connection_url())
            .set(drivername="postgresql+asyncpg", host="127.0.0.1")
            .render_as_string(hide_password=False)
        )
        os.environ["DATABASE_URL"] = database_url
        os.environ["APP_DB_PASSWORD"] = _APP_DB_PASSWORD
        command.upgrade(Config("alembic.ini"), "head")
        app_url = (
            make_url(database_url)
            .set(username="app_user", password=_APP_DB_PASSWORD)
            .render_as_string(hide_password=False)
        )
        neo4j_url = f"bolt://127.0.0.1:{neo4j.get_exposed_port(neo4j.port)}"
        asyncio.run(
            _measure(
                app_url,
                f"http://127.0.0.1:{qdrant.get_exposed_port(6333)}",
                (neo4j_url, neo4j.username, neo4j.password),
                parts,
            )
        )


if __name__ == "__main__":
    main()
