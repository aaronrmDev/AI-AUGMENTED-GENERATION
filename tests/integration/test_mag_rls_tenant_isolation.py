import uuid
from datetime import UTC, datetime

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.domain.entities import EpisodicMemory, SemanticMemory
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _create_user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
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
    return user_id


async def _create_session(db_session, tenant_id: uuid.UUID, user_id: uuid.UUID) -> uuid.UUID:
    session_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "t"},
    )
    return session_id


async def test_episodic_memory_rls_returns_zero_cross_tenant_rows_without_an_app_level_filter(
    db_session, embedding_model
):
    # Same shape as test_rag_rls_tenant_isolation.py's chunks test: prove the
    # DATABASE ITSELF refuses another tenant's rows, not that the
    # repository's own WHERE clause happens to filter correctly -- every
    # functional episodic test elsewhere passes tenant_id explicitly and
    # would keep passing even if this policy were silently dropped.
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    repo = PostgresEpisodicMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_a)
    user_a = await _create_user(db_session, tenant_a)
    session_a = await _create_session(db_session, tenant_a, user_a)
    await repo.save(
        EpisodicMemory(
            id=uuid.uuid4(), session_id=session_a, content={"who": "a"},
            embedding=embedding_model.embed("tenant a episode"), timestamp=datetime.now(UTC),
        ),
        tenant_a,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_b)
    user_b = await _create_user(db_session, tenant_b)
    session_b = await _create_session(db_session, tenant_b, user_b)
    await repo.save(
        EpisodicMemory(
            id=uuid.uuid4(), session_id=session_b, content={"who": "b"},
            embedding=embedding_model.embed("tenant b episode"), timestamp=datetime.now(UTC),
        ),
        tenant_b,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    # Deliberately no WHERE tenant_id = ... -- RLS alone must do the filtering.
    result = await db_session.execute(text("SELECT content FROM episodic_memory"))
    who_values = {row.content["who"] for row in result}

    assert who_values == {"a"}
    assert "b" not in who_values


async def test_semantic_memory_rls_returns_zero_cross_tenant_rows_without_an_app_level_filter(
    db_session, embedding_model
):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    repo = PostgresSemanticMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_a)
    user_a = await _create_user(db_session, tenant_a)
    await repo.save(
        SemanticMemory(
            id=uuid.uuid4(), user_id=user_a, fact_key="favorite_color",
            fact_value="tenant a's fact", embedding=embedding_model.embed("blue"),
        ),
        tenant_a,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_b)
    user_b = await _create_user(db_session, tenant_b)
    await repo.save(
        SemanticMemory(
            id=uuid.uuid4(), user_id=user_b, fact_key="favorite_color",
            fact_value="tenant b's fact", embedding=embedding_model.embed("red"),
        ),
        tenant_b,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    # Deliberately no WHERE tenant_id = ... -- RLS alone must do the filtering.
    result = await db_session.execute(text("SELECT fact_value FROM semantic_memory"))
    fact_values = {row.fact_value for row in result}

    assert fact_values == {"tenant a's fact"}
    assert "tenant b's fact" not in fact_values
