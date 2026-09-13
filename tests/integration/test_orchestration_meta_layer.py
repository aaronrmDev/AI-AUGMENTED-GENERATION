"""Real validation of the orchestration meta-layer's cascade against real
Postgres, Qdrant, MiniLM, and a distilgpt2 frozen cache. No LLM: generation
is ContextEchoChatModel, so this file needs only Docker."""
import json
import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.application.cascade_tiers import MagTier, RagTier
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierOutcome,
)
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)
from src.orchestration.infrastructure.session_scoped_semantic_fact_search import (
    SessionScopedSemanticFactSearch,
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


async def _stored_budget(db_session, env):
    await set_tenant_context(db_session, env.tenant_id)
    stored = (
        await db_session.execute(
            text("SELECT context_budget FROM sessions WHERE id = :id"), {"id": env.session_id}
        )
    ).scalar_one()
    return json.loads(stored) if isinstance(stored, str) else stored


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


async def test_parallel_routing_runs_every_real_tier_without_an_error(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    everything = RoutingDecision(frozenset(PARADIGM_ORDER), RoutingMode.PARALLEL, {})

    result = await env.cascade().run(env.request(PREFERENCE_QUERY, embedding_model), everything)

    assert [attempt.paradigm for attempt in result.attempts] == [CAG, MAG, RAG]
    assert not any(
        outcome in (TierOutcome.ERROR, TierOutcome.TIMEOUT) for _, outcome in _outcomes(result)
    )
    assert any(item.content == f"{FACT_KEY}: {FACT_VALUE}" for item in result.items)


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
        budget_recorder=PostgresSessionBudgetRecorder(env.sessionmaker),
    )

    result = await use_case.execute(env.tenant_id, env.user_id, env.session_id, FRESHNESS_QUERY)

    assert "forty-five days" in result.answer
    assert "thirty days of purchase" not in result.answer
    stored = await _stored_budget(db_session, env)
    assert stored["contributing"] == ["rag"]
    assert stored["slices"]["rag"] == result.allocation.rag == 108_800


class _SlowSemanticMemoryRepository(PostgresSemanticMemoryRepository):
    # A real server-side delay, so the MAG timeout cancels an in-flight
    # database round-trip rather than a task that never started.
    async def search_by_similarity(self, query_embedding, user_id, tenant_id, top_k):
        await self._session.execute(text("SELECT pg_sleep(0.5)"))
        return await super().search_by_similarity(query_embedding, user_id, tenant_id, top_k)


def _slow_mag_cascade(env) -> LatencyCascade:
    slow = MagTier(
        SessionScopedSemanticFactSearch(
            env.sessionmaker, repository_factory=_SlowSemanticMemoryRepository
        ),
        hit_threshold=MAG_HIT,
        partial_threshold=MAG_PARTIAL,
    )
    return LatencyCascade([slow, RagTier(env.search)], TierTimeouts(cag=5.0, mag=0.05, rag=10.0))


async def test_a_mag_query_cancelled_mid_flight_leaves_the_next_request_and_the_callers_session_working(  # noqa: E501
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    # Measured earlier in this batch: a cancelled query makes SQLAlchemy terminate
    # its connection, and rollback()/close() on that session then raise. Because
    # the MAG tier's search owns its session, that termination never reaches
    # another request or the caller's session -- no sleep, no recovery call.
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )

    first = await _slow_mag_cascade(env).run(
        env.request(PREFERENCE_QUERY, embedding_model), _route(MAG)
    )
    assert _outcomes(first) == [(MAG, TierOutcome.TIMEOUT), (RAG, TierOutcome.HIT)]

    second = await env.cascade().run(env.request(PREFERENCE_QUERY, embedding_model), _route(MAG))
    assert _outcomes(second) == [(MAG, TierOutcome.HIT)]
    assert (await db_session.execute(text("SELECT 1"))).scalar_one() == 1


async def test_a_mag_timeout_does_not_break_recording_the_turns_budget(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    use_case = UnifiedAnswerQuestion(
        embedding_model,
        FixedScoresClassifier({CAG: 0.0, MAG: 1.0, RAG: 0.0}),
        _slow_mag_cascade(env),
        ContextEchoChatModel(),
        budget_recorder=PostgresSessionBudgetRecorder(env.sessionmaker),
    )

    result = await use_case.execute(env.tenant_id, env.user_id, env.session_id, PREFERENCE_QUERY)

    assert (MAG, TierOutcome.TIMEOUT) in _outcomes(result)
    assert (await _stored_budget(db_session, env))["contributing"] == ["rag"]
