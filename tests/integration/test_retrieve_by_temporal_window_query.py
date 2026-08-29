import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.queries.retrieve_by_temporal_window import TemporalRetrieval
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


async def test_within_window_returns_only_episodes_inside_it_newest_first(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    base = datetime.now(UTC)
    inside_early = _episode(session_id, {"n": 1}, [0.1] * 384, timestamp=base)
    inside_late = _episode(
        session_id, {"n": 2}, [0.1] * 384, timestamp=base + timedelta(minutes=5)
    )
    outside = _episode(session_id, {"n": 3}, [0.1] * 384, timestamp=base + timedelta(days=1))
    await repo.save(inside_early, tenant_id)
    await repo.save(inside_late, tenant_id)
    await repo.save(outside, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    result = await TemporalRetrieval(repo).execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=10,
        within=(base, base + timedelta(minutes=30)),
    )

    assert [s.episode.id for s in result] == [inside_late.id, inside_early.id]
    assert all(s.score == 1.0 for s in result)


async def test_within_window_truncates_to_top_k(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    base = datetime.now(UTC)
    episodes = []
    for i in range(5):
        e = _episode(session_id, {"n": i}, [0.1] * 384, timestamp=base + timedelta(minutes=i))
        episodes.append(e)
        await repo.save(e, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    result = await TemporalRetrieval(repo).execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=2,
        within=(base, base + timedelta(hours=1)),
    )

    assert [s.episode.id for s in result] == [episodes[4].id, episodes[3].id]


async def test_recency_fallback_orders_newest_first_with_descending_scores(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    base = datetime.now(UTC)
    episodes = []
    for i in range(4):
        e = _episode(session_id, {"n": i}, [0.1] * 384, timestamp=base + timedelta(minutes=i))
        episodes.append(e)
        await repo.save(e, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    result = await TemporalRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, top_k=10
    )

    assert [s.episode.id for s in result] == [
        episodes[3].id,
        episodes[2].id,
        episodes[1].id,
        episodes[0].id,
    ]
    scores = [s.score for s in result]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0
    assert all(0.0 < score <= 1.0 for score in scores)


async def test_recency_fallback_truncates_to_top_k(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    base = datetime.now(UTC)
    episodes = []
    for i in range(5):
        e = _episode(session_id, {"n": i}, [0.1] * 384, timestamp=base + timedelta(minutes=i))
        episodes.append(e)
        await repo.save(e, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    result = await TemporalRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, top_k=2
    )

    assert [s.episode.id for s in result] == [episodes[4].id, episodes[3].id]
    assert result[0].score == 1.0
    assert 0.0 < result[1].score < 1.0


async def test_empty_session_returns_empty_list_in_both_modes(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    query = TemporalRetrieval(repo)
    base = datetime.now(UTC)

    windowed = await query.execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=10,
        within=(base - timedelta(days=1), base + timedelta(days=1)),
    )
    recent = await query.execute(tenant_id=tenant_id, session_id=session_id, top_k=10)

    assert windowed == []
    assert recent == []


async def test_another_tenants_episodes_never_leak_into_either_mode(db_session):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    # One real session, owned by tenant_a via the FK -- episodic_memory still
    # carries its own independent tenant_id column (mirroring chunks), so a
    # row can be written against this same session_id under tenant_b's own
    # context too. Neither mode of TemporalRetrieval scoped to tenant_a may
    # return tenant_b's row.
    session_id = await _insert_user_and_session(db_session, tenant_a)

    repo = PostgresEpisodicMemoryRepository(db_session)
    base = datetime.now(UTC)

    await set_tenant_context(db_session, tenant_a)
    episode_a = _episode(session_id, {"who": "a"}, [0.1] * 384, timestamp=base)
    await repo.save(episode_a, tenant_a)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_b)
    episode_b = _episode(session_id, {"who": "b"}, [0.1] * 384, timestamp=base)
    await repo.save(episode_b, tenant_b)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    query = TemporalRetrieval(repo)
    windowed = await query.execute(
        tenant_id=tenant_a,
        session_id=session_id,
        top_k=10,
        within=(base - timedelta(minutes=1), base + timedelta(minutes=1)),
    )
    recent = await query.execute(tenant_id=tenant_a, session_id=session_id, top_k=10)

    assert [s.episode.id for s in windowed] == [episode_a.id]
    assert [s.episode.id for s in recent] == [episode_a.id]
