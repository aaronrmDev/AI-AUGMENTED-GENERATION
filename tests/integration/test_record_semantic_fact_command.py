import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _create_user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
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


async def test_execute_dual_writes_to_real_postgres_and_qdrant(
    db_session, qdrant_url, embedding_model
):
    # This is the real seam CaptureEpisode/RecordSemanticFact's own unit
    # tests can't reach -- they inject fakes for both stores, so a bug in
    # how the two real writes actually interact (see the regression test
    # below) is invisible at the unit level. This test constructs the real
    # command against real infrastructure end to end.
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    command = RecordSemanticFact(
        semantic_memory_repository=repository, semantic_memory_index=index,
        embedding_model=embedding_model,
    )

    await set_tenant_context(db_session, tenant_id)
    fact = await command.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="favorite_color", fact_value="blue",
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    from_postgres = await repository.find_by_key(user_id, "favorite_color", tenant_id)
    assert from_postgres is not None
    assert from_postgres.id == fact.id
    assert from_postgres.fact_value == "blue"

    from_qdrant = await index.search(
        query_embedding=embedding_model.embed("blue"), user_id=user_id, tenant_id=tenant_id,
        top_k=5,
    )
    assert [r.fact.id for r in from_qdrant] == [fact.id]


async def test_recording_the_same_key_twice_does_not_orphan_a_stale_qdrant_point(
    db_session, qdrant_url, embedding_model
):
    # Regression test: an earlier version minted a fresh uuid4() per call.
    # Postgres upserts by (user_id, fact_key) (migration 0003's unique
    # constraint) and ends up with one row under the NEW id -- but Qdrant
    # has no equivalent "overwrite by fact_key," only "overwrite a point
    # with this exact id." A fresh id every call meant the OLD id's point
    # was never touched, so it stayed in the index forever, orphaned:
    # Postgres had no row referencing it, but a similarity search would
    # still surface it as if it were current. Fixed by deriving the fact's
    # id deterministically from (user_id, fact_key), so a re-record
    # overwrites the same Qdrant point instead of leaving a second one.
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    command = RecordSemanticFact(
        semantic_memory_repository=repository, semantic_memory_index=index,
        embedding_model=embedding_model,
    )

    await set_tenant_context(db_session, tenant_id)
    await command.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="favorite_color", fact_value="blue",
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    await command.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="favorite_color", fact_value="red",
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    postgres_count = await db_session.execute(
        text(
            "SELECT count(*) FROM semantic_memory WHERE user_id = :user_id AND fact_key = :key"
        ),
        {"user_id": user_id, "key": "favorite_color"},
    )
    assert postgres_count.scalar_one() == 1

    # top_k high enough that a real orphaned point (which this bug would
    # produce) is not accidentally excluded by the query embedding ranking
    # only the current fact into range.
    from_qdrant = await index.search(
        query_embedding=embedding_model.embed("favorite color"), user_id=user_id,
        tenant_id=tenant_id, top_k=10,
    )
    assert len(from_qdrant) == 1
    assert from_qdrant[0].fact.fact_value == "red"
