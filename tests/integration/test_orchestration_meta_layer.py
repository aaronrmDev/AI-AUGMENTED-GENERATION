"""Real validation of the orchestration meta-layer's cascade against real
Postgres, Qdrant, MiniLM, and a distilgpt2 frozen cache. No LLM: generation
is ContextEchoChatModel, so this file needs only Docker."""
import asyncio
import json
import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.application.cascade_tiers import MagTier, RagTier
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.orchestration.domain.entities import (
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierOutcome,
)
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)
from tests.integration.orchestration_env import (
    FACT_KEY,
    FACT_VALUE,
    FRESHNESS_QUERY,
    MAG_HIT,
    MAG_PARTIAL,
    POLICY_QUERY,
    PREFERENCE_QUERY,
    ContextEchoChatModel,
    FixedScoresClassifier,
    build_env,
)

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


def _route(*paradigms: Paradigm) -> RoutingDecision:
    return RoutingDecision(frozenset(paradigms), RoutingMode.CASCADE, {})


def _outcomes(result):
    return [(attempt.paradigm, attempt.outcome) for attempt in result.attempts]


async def test_routing_to_rag_keeps_a_superseded_cag_document_out_of_a_freshness_answer(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    cascade = env.cascade()

    unrouted = await cascade.run(env.request(FRESHNESS_QUERY, embedding_model))
    print(f"unrouted: {_outcomes(unrouted)} -> {[i.content[:60] for i in unrouted.items]}")
    assert _outcomes(unrouted) == [(CAG, TierOutcome.HIT)]
    assert any("thirty days" in item.content for item in unrouted.items)
    assert not any("forty-five" in item.content for item in unrouted.items)

    routed = await cascade.run(env.request(FRESHNESS_QUERY, embedding_model), _route(RAG))
    print(f"routed: {_outcomes(routed)} -> {[i.content[:60] for i in routed.items]}")
    assert [attempt.paradigm for attempt in routed.attempts] == [RAG]
    assert any("forty-five days" in item.content for item in routed.items)


async def test_a_real_cag_hit_short_circuits_before_mag_and_rag(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    result = await env.cascade().run(env.request(POLICY_QUERY, embedding_model), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.HIT)]
    assert result.items[0].source_id == env.policy_document_id


async def test_a_real_mag_fact_answers_a_question_about_the_users_own_preference(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    result = await env.cascade().run(
        env.request(PREFERENCE_QUERY, embedding_model), _route(MAG)
    )
    assert _outcomes(result) == [(MAG, TierOutcome.HIT)]
    assert result.items[0].content == f"{FACT_KEY}: {FACT_VALUE}"


async def test_another_users_fact_never_reaches_the_mag_tier(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    stranger = uuid.uuid4()
    result = await env.cascade().run(
        env.request(PREFERENCE_QUERY, embedding_model, user_id=stranger), _route(MAG)
    )
    assert _outcomes(result) == [(MAG, TierOutcome.MISS), (RAG, TierOutcome.HIT)]
    assert not any(FACT_VALUE in item.content for item in result.items)


async def test_the_unified_pipeline_answers_from_rag_and_records_the_real_budget(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    use_case = UnifiedAnswerQuestion(
        embedding_model,
        FixedScoresClassifier({CAG: 0.0, MAG: 0.0, RAG: 1.0}),
        env.cascade(),
        ContextEchoChatModel(),
        budget_recorder=PostgresSessionBudgetRecorder(get_sessionmaker(db_session.bind)),
    )

    result = await use_case.execute(env.tenant_id, env.user_id, env.session_id, FRESHNESS_QUERY)

    assert "forty-five days" in result.answer
    assert "thirty days of purchase" not in result.answer
    await set_tenant_context(db_session, env.tenant_id)
    stored = (
        await db_session.execute(
            text("SELECT context_budget FROM sessions WHERE id = :id"), {"id": env.session_id}
        )
    ).scalar_one()
    stored = json.loads(stored) if isinstance(stored, str) else stored
    assert stored["contributing"] == ["rag"]
    assert stored["slices"]["rag"] == result.allocation.rag == 108_800


class _SlowSemanticMemoryRepository(PostgresSemanticMemoryRepository):
    # A real server-side delay on the real session, so the MAG timeout
    # cancels an in-flight database round-trip rather than a task that
    # never started.
    async def search_by_similarity(self, query_embedding, user_id, tenant_id, top_k):
        await self._session.execute(text("SELECT pg_sleep(0.5)"))
        return await super().search_by_similarity(query_embedding, user_id, tenant_id, top_k)


def _slow_mag_cascade(env, db_session) -> LatencyCascade:
    slow = MagTier(
        FindSemanticFacts(_SlowSemanticMemoryRepository(db_session)),
        hit_threshold=MAG_HIT,
        partial_threshold=MAG_PARTIAL,
    )
    return LatencyCascade([slow, RagTier(env.search)], TierTimeouts(cag=5.0, mag=0.05, rag=10.0))


async def test_a_mag_query_cancelled_mid_flight_leaves_the_session_usable_after_invalidate(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    # Measured, not assumed: SQLAlchemy treats the cascade's CancelledError as
    # a disconnect and terminates the asyncpg connection, after which rollback()
    # and close() on this session both raise InterfaceError ("the underlying
    # connection is closed"). invalidate() is the recovery that works.
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    await db_session.commit()  # keep the seeded fact across the invalidation below
    await set_tenant_context(db_session, env.tenant_id)

    first = await _slow_mag_cascade(env, db_session).run(
        env.request(PREFERENCE_QUERY, embedding_model), _route(MAG)
    )
    assert _outcomes(first)[0] == (MAG, TierOutcome.TIMEOUT)

    await asyncio.sleep(0.6)
    await db_session.invalidate()
    await set_tenant_context(db_session, env.tenant_id)
    second = await env.cascade().run(env.request(PREFERENCE_QUERY, embedding_model), _route(MAG))
    assert _outcomes(second) == [(MAG, TierOutcome.HIT)]


async def test_a_mag_timeout_does_not_break_recording_the_turns_budget(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    await db_session.commit()
    await set_tenant_context(db_session, env.tenant_id)
    use_case = UnifiedAnswerQuestion(
        embedding_model,
        FixedScoresClassifier({CAG: 0.0, MAG: 1.0, RAG: 0.0}),
        _slow_mag_cascade(env, db_session),
        ContextEchoChatModel(),
        budget_recorder=PostgresSessionBudgetRecorder(get_sessionmaker(db_session.bind)),
    )

    result = await use_case.execute(env.tenant_id, env.user_id, env.session_id, PREFERENCE_QUERY)

    assert (MAG, TierOutcome.TIMEOUT) in _outcomes(result)
    await asyncio.sleep(0.6)
    await db_session.invalidate()
    await set_tenant_context(db_session, env.tenant_id)
    stored = (
        await db_session.execute(
            text("SELECT context_budget FROM sessions WHERE id = :id"), {"id": env.session_id}
        )
    ).scalar_one()
    stored = json.loads(stored) if isinstance(stored, str) else stored
    assert stored["contributing"] == ["rag"]
