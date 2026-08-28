import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.identity.infrastructure.db import set_tenant_context
from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _create_user(db_session: AsyncSession, tenant_id: uuid.UUID) -> uuid.UUID:
    # set_tenant_context's setting is transaction-LOCAL (see its own comment
    # in src/identity/infrastructure/db.py) and this helper commits -- so
    # every caller must re-set the context after calling this, right before
    # its own RLS-sensitive operation. Mirrors
    # test_postgres_episodic_memory_repository.py's identical pattern.
    await set_tenant_context(db_session, tenant_id)
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
            "tenant_id": tenant_id,
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
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repo = PostgresSemanticMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_id)
    embedding = embedding_model.embed("blue")
    fact = _fact(user_id, "favorite_color", "blue", embedding)
    await repo.save(fact, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    found = await repo.find_by_key(user_id, "favorite_color", tenant_id)

    assert found is not None
    assert found.id == fact.id
    assert found.user_id == user_id
    assert found.fact_key == "favorite_color"
    assert found.fact_value == "blue"
    assert found.confidence == 1.0
    assert found.source == ""
    assert found.valid_until is None
    # Never read back from Postgres -- see postgres_semantic_memory_repository
    # .py's _row_to_fact; the real-embedding round trip is covered in
    # test_qdrant_semantic_memory_index.py instead.
    assert found.embedding == []


async def test_saving_the_same_key_twice_updates_instead_of_duplicating(
    db_session, embedding_model
):
    # Regression test: migration 0003's uq_semantic_memory_user_id_fact_key
    # constraint plus save()'s ON CONFLICT DO UPDATE must mean a second
    # RecordSemanticFact for the same (user_id, fact_key) replaces the first
    # row rather than creating a second one resolved nondeterministically.
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repo = PostgresSemanticMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_id)
    await repo.save(
        _fact(user_id, "favorite_color", "blue", embedding_model.embed("blue")), tenant_id
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    await repo.save(
        _fact(user_id, "favorite_color", "red", embedding_model.embed("red")), tenant_id
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    count_result = await db_session.execute(
        text(
            "SELECT count(*) FROM semantic_memory WHERE user_id = :user_id AND fact_key = :key"
        ),
        {"user_id": user_id, "key": "favorite_color"},
    )
    assert count_result.scalar_one() == 1

    found = await repo.find_by_key(user_id, "favorite_color", tenant_id)
    assert found is not None
    assert found.fact_value == "red"


async def test_find_by_key_returns_none_for_an_unknown_key(db_session):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repo = PostgresSemanticMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_id)
    assert await repo.find_by_key(user_id, "unknown_key", tenant_id) is None


async def test_a_different_users_fact_never_leaks_into_find_by_key(db_session, embedding_model):
    tenant_id = uuid.uuid4()
    user_a = await _create_user(db_session, tenant_id)
    user_b = await _create_user(db_session, tenant_id)
    repo = PostgresSemanticMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_id)
    embedding = embedding_model.embed("blue")
    await repo.save(_fact(user_a, "favorite_color", "blue", embedding), tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    assert await repo.find_by_key(user_b, "favorite_color", tenant_id) is None


async def test_a_different_tenants_fact_never_leaks_into_find_by_key(db_session, embedding_model):
    # user_id alone used to be the only scoping key; a fact recorded for the
    # same user_id under a different tenant_id must not be visible either --
    # this is exactly the gap migration 0003's added tenant_id/RLS closes.
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_a)
    repo = PostgresSemanticMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_a)
    embedding = embedding_model.embed("blue")
    await repo.save(_fact(user_id, "favorite_color", "blue", embedding), tenant_a)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    assert await repo.find_by_key(user_id, "favorite_color", tenant_b) is None


async def test_search_by_similarity_returns_real_nearest_neighbor_ordering(
    db_session, embedding_model
):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repo = PostgresSemanticMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_id)
    close_fact = _fact(
        user_id, "pet", "The user owns a small domesticated cat.",
        embedding_model.embed("The user owns a small domesticated cat."),
    )
    far_fact = _fact(
        user_id, "job", "FastAPI background tasks run after the response is returned.",
        embedding_model.embed("FastAPI background tasks run after the response is returned."),
    )
    await repo.save(close_fact, tenant_id)
    await repo.save(far_fact, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    query_embedding = embedding_model.embed("What kind of pet does the user have?")
    results = await repo.search_by_similarity(
        query_embedding, user_id=user_id, tenant_id=tenant_id, top_k=2
    )

    assert len(results) == 2
    assert results[0].fact_key == "pet"
    assert results[1].fact_key == "job"


async def test_search_by_similarity_never_leaks_another_users_facts(db_session, embedding_model):
    tenant_id = uuid.uuid4()
    user_a = await _create_user(db_session, tenant_id)
    user_b = await _create_user(db_session, tenant_id)
    repo = PostgresSemanticMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_id)
    embedding = embedding_model.embed("blue")
    fact_a = _fact(user_a, "favorite_color", "blue", embedding)
    fact_b = _fact(user_b, "favorite_color", "blue", embedding)
    await repo.save(fact_a, tenant_id)
    await repo.save(fact_b, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    results = await repo.search_by_similarity(
        embedding, user_id=user_a, tenant_id=tenant_id, top_k=10
    )

    fact_ids = {fact.id for fact in results}
    assert fact_a.id in fact_ids
    assert fact_b.id not in fact_ids
