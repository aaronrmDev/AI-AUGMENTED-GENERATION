import uuid
from datetime import UTC, datetime

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.queries.retrieve_by_entity import EntityRetrieval
from src.mag.domain.entities import EpisodicMemory
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _insert_user_and_session(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    # episodic_memory.session_id is a foreign key to sessions.id, and sessions
    # itself is FORCE ROW LEVEL SECURITY -- so a real row for each is needed
    # before an episode can be saved at all, following the same pattern
    # test_postgres_episodic_memory_repository.py uses.
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
) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id,
        content=content,
        embedding=embedding,
        timestamp=timestamp or datetime.now(UTC),
    )


async def test_matches_via_structured_entities_jsonb_containment(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    matching = _episode(
        session_id, {"entities": ["Acme Corp", "Bob"], "output": "unrelated text"}, [0.1] * 384
    )
    # entities is populated but doesn't contain the queried entity -- proves
    # the containment check isn't secretly falling back to a substring match
    # on its own, since "Acme Corp" doesn't appear anywhere in this episode's
    # content at all.
    non_matching = _episode(session_id, {"entities": ["Bob"], "output": "unrelated"}, [0.1] * 384)
    await repo.save(matching, tenant_id)
    await repo.save(non_matching, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    result = await EntityRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, entity="Acme Corp", top_k=10
    )

    assert [s.episode.id for s in result] == [matching.id]
    assert result[0].score == 1.0


async def test_matches_via_ilike_substring_fallback_when_entities_not_populated(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    matching = _episode(
        session_id, {"output": "We discussed Acme Corp's quarterly earnings"}, [0.1] * 384
    )
    non_matching = _episode(session_id, {"output": "no relevant mention here"}, [0.1] * 384)
    await repo.save(matching, tenant_id)
    await repo.save(non_matching, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    result = await EntityRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, entity="Acme Corp", top_k=10
    )

    assert [s.episode.id for s in result] == [matching.id]
    assert result[0].score == 1.0


async def test_does_not_leak_another_tenants_matching_episode(db_session):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    # One real session, owned by tenant_a via the FK -- episodic_memory still
    # carries its own independent tenant_id column (mirroring chunks), so a
    # row can be written against this same session_id under tenant_b's own
    # context too -- same setup as
    # test_get_by_session_does_not_leak_another_tenants_episode_for_the_same_session
    # in test_postgres_episodic_memory_repository.py.
    session_id = await _insert_user_and_session(db_session, tenant_a)

    repo = PostgresEpisodicMemoryRepository(db_session)

    await set_tenant_context(db_session, tenant_a)
    episode_a = _episode(session_id, {"entities": ["Acme Corp"]}, [0.1] * 384)
    await repo.save(episode_a, tenant_a)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_b)
    episode_b = _episode(session_id, {"entities": ["Acme Corp"]}, [0.1] * 384)
    await repo.save(episode_b, tenant_b)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    result = await EntityRetrieval(repo).execute(
        tenant_id=tenant_a, session_id=session_id, entity="Acme Corp", top_k=10
    )

    assert [s.episode.id for s in result] == [episode_a.id]
