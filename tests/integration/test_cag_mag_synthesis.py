"""Real, live-measured validation of CAG+MAG synthesis -- the Combined
Capability Validation Story (#14/#30) plus the real numbers each mechanism
Story's DoD asks for. Real Postgres/Qdrant/Neo4j (testcontainers), real
sentence-transformers embeddings, real distilgpt2 (CPU) -- no GPU/vLLM.

Each test mints its own fresh tenant_id/user_id, the same per-test-
isolation discipline the two sibling cross-paradigm integration test
files already established.
"""
import time
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.application.commands.update_memory import UpdateMemory
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.orchestration.application.cag_mag_sync_cycle import CagMagSyncCycle
from src.orchestration.application.cag_mag_tiering_policy import CagMagTieringPolicy
from src.orchestration.domain import cag_mag_keys
from src.orchestration.domain.entities import TierDecision
from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache
from src.orchestration.infrastructure.in_memory_user_scoped_access_tracker import (
    InMemoryUserScopedAccessFrequencyTracker,
)

_WINDOW = timedelta(hours=1)
_NOW = datetime(2026, 9, 2, 12, 0, 0)
_VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
_FACT_KEY = "preferred_visualization_library"
_ORIGINAL_VALUE = "strongly prefers matplotlib for all data visualization tasks"


async def _create_user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    await set_tenant_context(db_session, tenant_id)
    now = datetime.now(UTC)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": _VALID_HASH,
            "tenant_id": tenant_id, "created_at": now, "updated_at": now,
        },
    )
    await db_session.commit()
    return user_id


async def _memory_graph_repository(neo4j_url) -> Neo4jMemoryGraphRepository:
    url, username, password = neo4j_url
    repository = Neo4jMemoryGraphRepository(url, auth=(username, password))
    await repository.ensure_schema()
    return repository


async def test_hot_warm_tiering_promotes_and_demotes_a_real_mag_fact_with_real_ttft_win(
    db_session, qdrant_url, embedding_model, neo4j_url, distilgpt2_model, distilgpt2_tokenizer
):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)

    semantic_repo = PostgresSemanticMemoryRepository(db_session)
    semantic_index = QdrantSemanticMemoryIndex(qdrant_url)
    await semantic_index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    record_fact = RecordSemanticFact(semantic_repo, semantic_index, embedding_model, graph)

    await set_tenant_context(db_session, tenant_id)
    await record_fact.execute(tenant_id, user_id, _FACT_KEY, _ORIGINAL_VALUE)
    await db_session.commit()

    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    policy = CagMagTieringPolicy(tracker, cache)

    def content_provider(fact_key: str) -> str:
        return _ORIGINAL_VALUE

    for _ in range(10):
        tracker.record_access(tenant_id, user_id, cag_mag_keys.tracker_key(_FACT_KEY), _NOW)

    promote_decision = policy.evaluate(
        tenant_id, user_id, _FACT_KEY, content_provider,
        promote_threshold=5, demote_threshold=2, window=_WINDOW, now=_NOW,
    )
    print(f"real tiering decision after heavy traffic: {promote_decision}")
    assert promote_decision == TierDecision.PROMOTED

    cache_id = cag_mag_keys.cache_key(user_id, _FACT_KEY)
    hit = cache.lookup(tenant_id, cache_id)
    assert hit is not None
    assert hit.kv_cache is not None  # a real KV cache, not a stub

    # Real "near-zero TTFT on a hit" for MAG-sourced content specifically --
    # the same mechanism Batch D measured for RAG documents, now measured
    # for a real promoted MAG fact, not assumed to transfer.
    trials = [cache.prefill_latency_ms(tenant_id, cache_id, _ORIGINAL_VALUE) for _ in range(3)]
    cold_ms = sorted(t[0] for t in trials)[1]
    warm_ms = sorted(t[1] for t in trials)[1]
    print(f"real cold-vs-warm prefill latency for promoted MAG content (ms): {trials}")
    print(f"median cold: {cold_ms:.1f}ms, median warm: {warm_ms:.1f}ms")
    assert cold_ms > 1.5 * warm_ms

    later = _NOW + timedelta(hours=2)
    demote_decision = policy.evaluate(
        tenant_id, user_id, _FACT_KEY, content_provider,
        promote_threshold=5, demote_threshold=2, window=_WINDOW, now=later,
    )
    print(f"real tiering decision after traffic dried up: {demote_decision}")
    assert demote_decision == TierDecision.DEMOTED
    assert cache.lookup(tenant_id, cache_id) is None


async def test_a_second_users_traffic_never_promotes_into_the_first_users_hot_tier(
    db_session, qdrant_url, embedding_model, distilgpt2_model, distilgpt2_tokenizer
):
    tenant_id = uuid.uuid4()
    user_a = await _create_user(db_session, tenant_id)
    user_b = await _create_user(db_session, tenant_id)

    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    tracker = InMemoryUserScopedAccessFrequencyTracker()
    policy = CagMagTieringPolicy(tracker, cache)

    def content_provider(fact_key: str) -> str:
        return _ORIGINAL_VALUE

    await set_tenant_context(db_session, tenant_id)
    for _ in range(10):
        tracker.record_access(tenant_id, user_a, cag_mag_keys.tracker_key(_FACT_KEY), _NOW)

    policy.evaluate(
        tenant_id, user_a, _FACT_KEY, content_provider,
        promote_threshold=5, demote_threshold=2, window=_WINDOW, now=_NOW,
    )
    decision_b = policy.evaluate(
        tenant_id, user_b, _FACT_KEY, content_provider,
        promote_threshold=5, demote_threshold=2, window=_WINDOW, now=_NOW,
    )

    assert decision_b == TierDecision.UNCHANGED
    assert cache.lookup(tenant_id, cag_mag_keys.cache_key(user_a, _FACT_KEY)) is not None
    assert cache.lookup(tenant_id, cag_mag_keys.cache_key(user_b, _FACT_KEY)) is None


async def test_sync_cycle_detects_a_real_mag_update_and_evicts_the_stale_cag_copy(
    db_session, qdrant_url, embedding_model, neo4j_url, distilgpt2_model, distilgpt2_tokenizer
):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)

    semantic_repo = PostgresSemanticMemoryRepository(db_session)
    semantic_index = QdrantSemanticMemoryIndex(qdrant_url)
    await semantic_index.ensure_collection()
    graph = await _memory_graph_repository(neo4j_url)
    record_fact = RecordSemanticFact(semantic_repo, semantic_index, embedding_model, graph)
    update_memory = UpdateMemory(semantic_repo, record_fact)

    await set_tenant_context(db_session, tenant_id)
    await record_fact.execute(tenant_id, user_id, _FACT_KEY, _ORIGINAL_VALUE)
    await db_session.commit()

    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    cache_id = cag_mag_keys.cache_key(user_id, _FACT_KEY)
    cache.preload(tenant_id, cache_id, _ORIGINAL_VALUE)
    sync_cycle = CagMagSyncCycle(cache)

    # A real, live change to MAG's own data -- this batch's own tiering
    # mechanism is what put a point-in-time copy into CAG in the first
    # place, and MAG's normal write path (UpdateMemory, unmodified) is
    # what makes it go stale, exactly as the design spec's investigation
    # concluded.
    new_value = "strongly prefers seaborn for all data visualization tasks"
    change_at = time.perf_counter()
    await set_tenant_context(db_session, tenant_id)
    await update_memory.execute(tenant_id, user_id, _FACT_KEY, new_value)
    await db_session.commit()

    async def authoritative_content(fact_key: str) -> str:
        await set_tenant_context(db_session, tenant_id)
        fact = await semantic_repo.find_by_key(user_id, fact_key, tenant_id)
        assert fact is not None
        return fact.fact_value

    detected_at = None
    for _ in range(20):
        current_value = await authoritative_content(_FACT_KEY)
        conflicts = sync_cycle.run(
            tenant_id, user_id, [_FACT_KEY], lambda key, v=current_value: v
        )
        if conflicts:
            detected_at = time.perf_counter()
            break
        time.sleep(0.05)

    assert detected_at is not None, "CagMagSyncCycle never detected the real MAG update"
    latency_ms = (detected_at - change_at) * 1000
    print(
        f"real measured CAG-vs-MAG detection-to-eviction latency: {latency_ms:.1f}ms "
        f"(OVERVIEW.md's illustrative figure for the analogous RAG-vs-CAG case: 300000ms / "
        f"5 minutes -- a mechanism-latency measurement, not a claim about production batch "
        f"cadence, the same disclosure Batch D and Batch E's own reports already established)"
    )
    assert cache.lookup(tenant_id, cache_id) is None
