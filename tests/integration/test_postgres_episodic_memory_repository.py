import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.domain.entities import EpisodicMemory
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _insert_user_and_session(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    # episodic_memory.session_id is a foreign key to sessions.id, and sessions
    # itself is FORCE ROW LEVEL SECURITY -- so a real row for each is needed
    # before an episode can be saved at all, following the same pattern
    # test_rls_tenant_isolation.py uses for the sessions table.
    await set_tenant_context(db_session, tenant_id)
    now = datetime.now(UTC)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id,
            "email": f"{user_id}@example.com",
            "hashed_password": VALID_HASH,
            "tenant_id": tenant_id,
            "created_at": now,
            "updated_at": now,
        },
    )
    session_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "test session"},
    )
    await db_session.commit()
    return session_id


def _episode(
    session_id: uuid.UUID,
    content: dict,
    embedding: list[float],
    timestamp: datetime | None = None,
    salience_score: float = 0.0,
) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id,
        content=content,
        embedding=embedding,
        timestamp=timestamp or datetime.now(UTC),
        salience_score=salience_score,
    )


async def test_save_then_get_by_session_round_trips(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    episode = _episode(
        session_id,
        content={"input": "hi", "output": "hello", "tool_calls": []},
        embedding=[0.1] * 384,
        salience_score=0.75,
    )
    await repo.save(episode, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    result = await repo.get_by_session(session_id, tenant_id)

    assert len(result) == 1
    assert result[0].id == episode.id
    assert result[0].session_id == session_id
    assert result[0].content == episode.content
    assert result[0].salience_score == episode.salience_score


async def test_get_by_session_orders_by_timestamp_ascending(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    base = datetime.now(UTC)
    later = _episode(session_id, {"n": 2}, [0.1] * 384, timestamp=base + timedelta(seconds=30))
    earlier = _episode(session_id, {"n": 1}, [0.1] * 384, timestamp=base)
    await repo.save(later, tenant_id)
    await repo.save(earlier, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    result = await repo.get_by_session(session_id, tenant_id)

    assert [e.id for e in result] == [earlier.id, later.id]


async def test_get_by_session_does_not_leak_another_tenants_episode_for_the_same_session(
    db_session,
):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    # One real session, owned by tenant_a via the FK -- episodic_memory still
    # carries its own independent tenant_id column (mirroring chunks), so a
    # row can be written against this same session_id under tenant_b's own
    # context too. get_by_session(session_id, tenant_a) must not return it.
    session_id = await _insert_user_and_session(db_session, tenant_a)

    repo = PostgresEpisodicMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_a)
    episode_a = _episode(session_id, {"who": "a"}, [0.1] * 384)
    await repo.save(episode_a, tenant_a)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_b)
    episode_b = _episode(session_id, {"who": "b"}, [0.1] * 384)
    await repo.save(episode_b, tenant_b)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    result = await repo.get_by_session(session_id, tenant_a)

    assert [e.id for e in result] == [episode_a.id]


async def test_search_by_similarity_orders_by_nearest_neighbor(db_session, embedding_model):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    about_cats = _episode(
        session_id, {"text": "cats"}, embedding_model.embed("the cat sat on the warm mat")
    )
    about_finance = _episode(
        session_id,
        {"text": "finance"},
        embedding_model.embed("quarterly earnings report and stock prices"),
    )
    await repo.save(about_cats, tenant_id)
    await repo.save(about_finance, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    query_embedding = embedding_model.embed("a small kitten playing with a ball of yarn")
    result = await repo.search_by_similarity(query_embedding, tenant_id, top_k=2)

    assert len(result) == 2
    assert result[0].id == about_cats.id


async def test_search_by_similarity_only_returns_the_given_tenants_episodes(
    db_session, embedding_model
):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    session_a = await _insert_user_and_session(db_session, tenant_a)
    session_b = await _insert_user_and_session(db_session, tenant_b)

    repo = PostgresEpisodicMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_a)
    episode_a = _episode(session_a, {"text": "a"}, embedding_model.embed("alpha content"))
    await repo.save(episode_a, tenant_a)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_b)
    episode_b = _episode(session_b, {"text": "b"}, embedding_model.embed("alpha content"))
    await repo.save(episode_b, tenant_b)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    result = await repo.search_by_similarity(
        embedding_model.embed("alpha content"), tenant_a, top_k=10
    )

    assert {e.id for e in result} == {episode_a.id}
