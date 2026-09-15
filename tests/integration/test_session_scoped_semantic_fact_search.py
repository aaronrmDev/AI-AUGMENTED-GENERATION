"""SessionScopedSemanticFactSearch against real Postgres: MAG's cascade tier
runs its query in a session it owns, so a timeout that cancels the query
mid-flight can only ever terminate that session's own connection."""
import asyncio
import uuid

import pytest
from sqlalchemy import text

from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.infrastructure.session_scoped_semantic_fact_search import (
    SessionScopedSemanticFactSearch,
)
from tests.integration.orchestration_env import FACT_KEY, FACT_VALUE, create_user_and_session


async def _seed_fact(db_session, embedding_model) -> tuple[uuid.UUID, uuid.UUID]:
    tenant_id = uuid.uuid4()
    user_id, _ = await create_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)
    fact = SemanticMemory(
        id=uuid.uuid4(),
        user_id=user_id,
        fact_key=FACT_KEY,
        fact_value=FACT_VALUE,
        embedding=embedding_model.embed(FACT_VALUE),
    )
    await PostgresSemanticMemoryRepository(db_session).save(fact, tenant_id)
    await db_session.commit()
    return tenant_id, user_id


class _SlowRepository(PostgresSemanticMemoryRepository):
    # A real server-side delay, so cancellation lands on an in-flight round-trip.
    async def search_by_similarity(self, query_embedding, user_id, tenant_id, top_k):
        await self._session.execute(text("SELECT pg_sleep(0.5)"))
        return await super().search_by_similarity(query_embedding, user_id, tenant_id, top_k)


async def test_search_finds_the_users_fact_and_sets_its_own_tenant_context(
    db_session, embedding_model
):
    tenant_id, user_id = await _seed_fact(db_session, embedding_model)
    search = SessionScopedSemanticFactSearch(get_sessionmaker(db_session.bind))

    facts = await search.search(tenant_id, user_id, embedding_model.embed(FACT_VALUE), 3)

    assert [scored.fact.fact_key for scored in facts] == [FACT_KEY]


async def test_search_under_another_tenant_finds_nothing(db_session, embedding_model):
    _, user_id = await _seed_fact(db_session, embedding_model)
    search = SessionScopedSemanticFactSearch(get_sessionmaker(db_session.bind))

    facts = await search.search(uuid.uuid4(), user_id, embedding_model.embed(FACT_VALUE), 3)

    assert facts == []


async def test_a_search_cancelled_mid_flight_leaves_nothing_broken_behind(
    db_session, embedding_model
):
    tenant_id, user_id = await _seed_fact(db_session, embedding_model)
    sessionmaker = get_sessionmaker(db_session.bind)
    embedding = embedding_model.embed(FACT_VALUE)
    slow = SessionScopedSemanticFactSearch(sessionmaker, repository_factory=_SlowRepository)
    pool = db_session.bind.sync_engine.pool
    baseline = pool.checkedout()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(slow.search(tenant_id, user_id, embedding, 3), 0.05)
    # wait_for has awaited the cancelled search's cleanup: no connection leaked.
    assert pool.checkedout() == baseline

    # No sleep and no recovery call: the next search and the caller's own
    # session both work immediately.
    facts = await SessionScopedSemanticFactSearch(sessionmaker).search(
        tenant_id, user_id, embedding, 3
    )
    assert [scored.fact.fact_key for scored in facts] == [FACT_KEY]
    assert (await db_session.execute(text("SELECT 1"))).scalar_one() == 1
