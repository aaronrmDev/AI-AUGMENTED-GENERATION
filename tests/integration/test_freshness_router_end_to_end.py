from src.orchestration.domain.entities import IngestionRoute, Paradigm, TierOutcome
from tests.integration.freshness_env import (
    DAY,
    HOUR,
    POLICY,
    POLICY_QUESTION,
    PRICE_QUESTION,
    PRICES,
    SIZE,
    SIZE_QUESTION,
    SIZE_TEXT,
    T0,
    freshness_env,
    policy_text,
    price_text,
)

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


def _context(result) -> str:
    return "\n".join(item.content for item in result.items)


def _paradigms(result) -> set[Paradigm]:
    return {item.paradigm for item in result.items}


async def test_a_volatile_source_is_answered_from_rag_and_never_cached(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        await env.ingest.execute(env.tenant_id, PRICES, price_text(41), T0)
        assert await env.refresh.run(env.tenant_id, T0) == []
        first = await env.ask(PRICE_QUESTION)
        await env.ingest.execute(env.tenant_id, PRICES, price_text(43), T0 + HOUR)
        second = await env.ask(PRICE_QUESTION)

    assert CAG not in _paradigms(first)
    assert "41 dollars" in _context(first)
    assert "43 dollars" in _context(second)
    assert "41 dollars" not in _context(second)


async def test_a_stable_source_is_served_from_cag_once_the_batch_has_preloaded_it(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        result = await env.ingest.execute(env.tenant_id, POLICY, policy_text("forty-five"), T0)
        before = await env.ask(POLICY_QUESTION)
        assert await env.refresh.run(env.tenant_id, T0) == ["return-policy"]
        after = await env.ask(POLICY_QUESTION)

    assert result.route is IngestionRoute.CAG_WITH_RAG_BACKUP
    assert CAG not in _paradigms(before)
    assert [(a.paradigm, a.outcome) for a in after.attempts] == [(CAG, TierOutcome.HIT)]
    assert "forty-five days" in _context(after)


async def test_a_changed_cached_source_is_served_from_rag_until_the_next_refresh(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        await env.ingest.execute(env.tenant_id, POLICY, policy_text("forty-five"), T0)
        await env.refresh.run(env.tenant_id, T0)
        env.clock.now = T0 + DAY
        await env.ingest.execute(env.tenant_id, POLICY, policy_text("sixty"), T0 + DAY)
        between = await env.ask(POLICY_QUESTION)
        await env.refresh.run(env.tenant_id, T0 + DAY)
        refreshed = await env.ask(POLICY_QUESTION)

    assert CAG not in _paradigms(between)
    assert "sixty days" in _context(between)
    assert "forty-five days" not in _context(between)
    assert CAG in _paradigms(refreshed)
    assert "sixty days" in _context(refreshed)
    assert "forty-five days" not in _context(refreshed)


async def test_an_expired_entry_falls_back_to_rag_and_is_not_preloaded_unconfirmed(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        await env.ingest.execute(env.tenant_id, POLICY, policy_text("forty-five"), T0)
        await env.refresh.run(env.tenant_id, T0)
        fresh = await env.ask(POLICY_QUESTION)
        env.clock.now = T0 + 45 * DAY  # 90-day interval x 0.5: exactly one TTL later
        expired = await env.ask(POLICY_QUESTION)
        preloaded = await env.refresh.run(env.tenant_id, env.clock.now)

    assert CAG in _paradigms(fresh)
    assert CAG not in _paradigms(expired)
    assert RAG in _paradigms(expired)
    assert preloaded == []


async def test_a_user_scoped_source_reaches_its_owner_through_mag_and_no_one_else(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        result = await env.ingest.execute(
            env.tenant_id, SIZE, SIZE_TEXT, T0, user_id=env.owner_id
        )
        owner = await env.ask(SIZE_QUESTION, env.owner_id)
        other = await env.ask(SIZE_QUESTION, env.other_user_id)

    assert result.route is IngestionRoute.MAG
    assert MAG in _paradigms(owner)
    assert "size 10" in _context(owner)
    assert "size 10" not in _context(other)


async def test_review_demotes_a_cached_source_that_starts_changing_hourly(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        await env.ingest.execute(env.tenant_id, POLICY, policy_text("45"), T0)
        await env.refresh.run(env.tenant_id, T0)
        for hour, days in ((1, "46"), (2, "47"), (3, "48")):
            await env.ingest.execute(env.tenant_id, POLICY, policy_text(days), T0 + hour * HOUR)
        env.clock.now = T0 + 4 * HOUR
        migrations = await env.review.run(env.tenant_id, env.clock.now)
        preloaded = await env.refresh.run(env.tenant_id, env.clock.now)
        answer = await env.ask(POLICY_QUESTION)

    assert [(m.source_key, m.to_route) for m in migrations] == [
        ("return-policy", IngestionRoute.RAG_ONLY)
    ]
    assert migrations[0].observed_interval == HOUR
    assert preloaded == []
    assert CAG not in _paradigms(answer)
    assert "48 days" in _context(answer)
