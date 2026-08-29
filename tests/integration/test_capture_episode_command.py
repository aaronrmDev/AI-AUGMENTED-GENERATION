import uuid
from datetime import UTC, datetime

import ollama
from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.capture_episode import CaptureEpisode
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)
from src.mag.infrastructure.qdrant_episodic_memory_index import QdrantEpisodicMemoryIndex
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
_MODEL_ID = "qwen3.5"


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
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    command = CaptureEpisode(
        episodic_memory_repository=repository, episodic_memory_index=index,
        embedding_model=embedding_model, chat_model=chat_model,
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
    assert [s.episode.id for s in from_qdrant] == [episode.id]
    # Not a dedicated live-measurement of the salience score itself (see
    # test_execute_scores_a_failure_episode_more_salient_than_a_routine_one
    # below for that) -- just confirming the real chat_model call this test
    # now depends on actually ran and produced something in the documented
    # [0.0, 1.0] range, not that execute() silently fell back to the
    # parse-failure default every time.
    assert 0.0 <= episode.salience_score <= 1.0


async def test_execute_scores_a_failure_episode_more_salient_than_a_routine_one(
    db_session, qdrant_url, embedding_model
):
    # Real Ollama model (not a fake), same reasoning as
    # test_consolidate_episodes_command.py's live reflection test: a fake
    # chat model can prove the retry/parsing logic works against a SCRIPTED
    # response, but not that a real model's judgment on real content
    # actually differentiates a critical failure from a routine turn the
    # way #74's own language ("weights critical decisions or failures more
    # heavily than routine turns") describes.
    tenant_id = uuid.uuid4()
    session_id = await _create_user_and_session(db_session, tenant_id)
    repository = PostgresEpisodicMemoryRepository(db_session)
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    command = CaptureEpisode(
        episodic_memory_repository=repository, episodic_memory_index=index,
        embedding_model=embedding_model, chat_model=chat_model,
    )

    await set_tenant_context(db_session, tenant_id)
    failure_episode = await command.execute(
        tenant_id=tenant_id,
        session_id=session_id,
        content={
            "input": "deploy the payments service to production",
            "reasoning": "ran the deploy script without checking the health endpoint first",
            "tool_calls": [{"name": "deploy", "args": {"service": "payments"}}],
            "output": "Traceback (most recent call last): ConnectionRefusedError: "
            "database connection failed",
            "outcome": "failure",
        },
    )
    await db_session.commit()
    await set_tenant_context(db_session, tenant_id)
    routine_episode = await command.execute(
        tenant_id=tenant_id,
        session_id=session_id,
        content={
            "input": "what's the weather like today",
            "output": "It's sunny with a high of 72F.",
            "outcome": "success",
        },
    )
    await db_session.commit()

    print(
        f"\nSalience scoring (real Ollama, {_MODEL_ID}): "
        f"failure episode={failure_episode.salience_score}, "
        f"routine episode={routine_episode.salience_score}"
    )
    assert failure_episode.salience_score > routine_episode.salience_score
