import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _create_user(db_session: AsyncSession) -> uuid.UUID:
    # semantic_memory.user_id is a foreign key to users, and semantic_memory
    # carries no RLS of its own (migration 0003 comment: no user_id-keyed RLS
    # pattern exists yet) -- so this insert needs no tenant context, matching
    # how test_rls_tenant_isolation.py inserts users ahead of its RLS-scoped
    # sessions rows.
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id) "
            "VALUES (:id, :email, :hashed_password, :tenant_id)"
        ),
        {
            "id": user_id,
            "email": f"{user_id}@example.com",
            "hashed_password": VALID_HASH,
            "tenant_id": uuid.uuid4(),
        },
    )
    await db_session.commit()
    return user_id


def _fact(
    user_id: uuid.UUID, fact_key: str, fact_value: str, embedding: list[float]
) -> SemanticMemory:
    return SemanticMemory(
        id=uuid.uuid4(), user_id=user_id, fact_key=fact_key, fact_value=fact_value,
        embedding=embedding,
    )


async def test_save_then_find_by_key_round_trips(db_session, embedding_model):
    user_id = await _create_user(db_session)
    repo = PostgresSemanticMemoryRepository(db_session)

    embedding = embedding_model.embed("blue")
    fact = _fact(user_id, "favorite_color", "blue", embedding)
    await repo.save(fact)
    await db_session.commit()

    found = await repo.find_by_key(user_id, "favorite_color")

    assert found is not None
    assert found.id == fact.id
    assert found.user_id == user_id
    assert found.fact_key == "favorite_color"
    assert found.fact_value == "blue"
    assert found.confidence == 1.0
    assert found.source == ""
    assert found.valid_until is None
    assert len(found.embedding) == 384
    assert found.embedding == pytest.approx(embedding, rel=1e-3)


async def test_find_by_key_returns_none_for_an_unknown_key(db_session):
    user_id = await _create_user(db_session)
    repo = PostgresSemanticMemoryRepository(db_session)

    assert await repo.find_by_key(user_id, "unknown_key") is None


async def test_a_different_users_fact_never_leaks_into_find_by_key(db_session, embedding_model):
    user_a = await _create_user(db_session)
    user_b = await _create_user(db_session)
    repo = PostgresSemanticMemoryRepository(db_session)

    embedding = embedding_model.embed("blue")
    await repo.save(_fact(user_a, "favorite_color", "blue", embedding))
    await db_session.commit()

    assert await repo.find_by_key(user_b, "favorite_color") is None


async def test_search_by_similarity_returns_real_nearest_neighbor_ordering(
    db_session, embedding_model
):
    user_id = await _create_user(db_session)
    repo = PostgresSemanticMemoryRepository(db_session)

    close_fact = _fact(
        user_id, "pet", "The user owns a small domesticated cat.",
        embedding_model.embed("The user owns a small domesticated cat."),
    )
    far_fact = _fact(
        user_id, "job", "FastAPI background tasks run after the response is returned.",
        embedding_model.embed("FastAPI background tasks run after the response is returned."),
    )
    await repo.save(close_fact)
    await repo.save(far_fact)
    await db_session.commit()

    query_embedding = embedding_model.embed("What kind of pet does the user have?")
    results = await repo.search_by_similarity(query_embedding, user_id=user_id, top_k=2)

    assert len(results) == 2
    assert results[0].fact_key == "pet"
    assert results[1].fact_key == "job"


async def test_search_by_similarity_never_leaks_another_users_facts(db_session, embedding_model):
    user_a = await _create_user(db_session)
    user_b = await _create_user(db_session)
    repo = PostgresSemanticMemoryRepository(db_session)

    embedding = embedding_model.embed("blue")
    fact_a = _fact(user_a, "favorite_color", "blue", embedding)
    fact_b = _fact(user_b, "favorite_color", "blue", embedding)
    await repo.save(fact_a)
    await repo.save(fact_b)
    await db_session.commit()

    results = await repo.search_by_similarity(embedding, user_id=user_a, top_k=10)

    fact_ids = {fact.id for fact in results}
    assert fact_a.id in fact_ids
    assert fact_b.id not in fact_ids
