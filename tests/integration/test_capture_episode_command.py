import uuid
from datetime import UTC, datetime

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.capture_episode import CaptureEpisode
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)
from src.mag.infrastructure.qdrant_episodic_memory_index import QdrantEpisodicMemoryIndex

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _create_user_and_session(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    await set_tenant_context(db_session, tenant_id)
    now = datetime.now(UTC)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id, "created_at": now, "updated_at": now,
        },
    )
    session_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "t"},
    )
    await db_session.commit()
    return session_id


async def test_execute_dual_writes_to_real_postgres_and_qdrant(
    db_session, qdrant_url, embedding_model
):
    # Same seam as test_record_semantic_fact_command.py -- CaptureEpisode's
    # own unit tests inject fakes for both stores, so this is the only test
    # that actually exercises the real dual write together.
    tenant_id = uuid.uuid4()
    session_id = await _create_user_and_session(db_session, tenant_id)
    repository = PostgresEpisodicMemoryRepository(db_session)
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()
    command = CaptureEpisode(
        episodic_memory_repository=repository, episodic_memory_index=index,
        embedding_model=embedding_model,
    )

    await set_tenant_context(db_session, tenant_id)
    content = {"input": "what's the weather", "output": "sunny", "tool_calls": []}
    episode = await command.execute(tenant_id=tenant_id, session_id=session_id, content=content)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    from_postgres = await repository.get_by_session(session_id, tenant_id)
    assert [e.id for e in from_postgres] == [episode.id]
    assert from_postgres[0].content == content

    from_qdrant = await index.search(
        query_embedding=episode.embedding, tenant_id=tenant_id, top_k=5
    )
    assert [e.id for e in from_qdrant] == [episode.id]
