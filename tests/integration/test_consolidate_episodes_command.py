"""Real end-to-end test for ConsolidateEpisodes: real Postgres, real Qdrant,
and a real Ollama model (not a fake) for the reflection call. A fake chat
model can prove the retry/parsing logic works against a SCRIPTED response,
but it can't answer the actual empirical question this project's testing
discipline cares about: does a real model's JSON output, under real
prompting, actually get parsed correctly by that logic. Requires Ollama
running locally with qwen3.5 pulled (same model this project's RAG batches
already depend on for live measurement).
"""
import uuid
from datetime import UTC, datetime

import ollama
from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.consolidate_episodes import ConsolidateEpisodes
from src.mag.domain.entities import EpisodicMemory
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
_MODEL_ID = "qwen3.5"


async def _insert_user_and_session(db_session, tenant_id: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
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
    return user_id, session_id


async def test_execute_consolidates_real_episodes_with_a_real_ollama_model(
    db_session, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    user_id, session_id = await _insert_user_and_session(db_session, tenant_id)

    episodic_repo = PostgresEpisodicMemoryRepository(db_session)
    semantic_repo = PostgresSemanticMemoryRepository(db_session)
    semantic_index = QdrantSemanticMemoryIndex(qdrant_url)
    await semantic_index.ensure_collection()
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)

    await set_tenant_context(db_session, tenant_id)
    now = datetime.now(UTC)
    # Mirrors MAG.md's own worked example shape (three Python mentions, one
    # Go mention) closely enough that a reasonable reflection should surface
    # a primary-language fact -- this is a real prompt to a real model, so
    # the assertions below check for a plausible, non-empty result rather
    # than an exact string match, which would be asserting the model's
    # specific wording rather than the pipeline's correctness.
    episodes = [
        EpisodicMemory(
            id=uuid.uuid4(), session_id=session_id,
            content={
                "input": "How do I read a CSV file?",
                "output": "Use pandas.read_csv() in Python.",
            },
            embedding=[0.0] * 384, timestamp=now,
        ),
        EpisodicMemory(
            id=uuid.uuid4(), session_id=session_id,
            content={
                "input": "What's a good way to write unit tests?",
                "output": "pytest is the standard choice for Python testing.",
            },
            embedding=[0.0] * 384, timestamp=now,
        ),
        EpisodicMemory(
            id=uuid.uuid4(), session_id=session_id,
            content={
                "input": "How do I format a Python string?",
                "output": "Use an f-string, e.g. f'{value}'.",
            },
            embedding=[0.0] * 384, timestamp=now,
        ),
    ]
    for episode in episodes:
        await episodic_repo.save(episode, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    command = ConsolidateEpisodes(
        episodic_memory_repository=episodic_repo,
        semantic_memory_repository=semantic_repo,
        semantic_memory_index=semantic_index,
        embedding_model=embedding_model,
        chat_model=chat_model,
    )
    result = await command.execute(tenant_id=tenant_id, user_id=user_id, session_id=session_id)
    await db_session.commit()

    print(f"\nConsolidation extracted {len(result)} fact(s) from a real Ollama reflection:")
    for fact in result:
        print(f"  {fact.fact_key!r} = {fact.fact_value!r} (confidence={fact.confidence})")

    # All three source episodes must be marked consolidated regardless of
    # what the model extracted -- checked against the real database, not
    # just the in-process return value.
    await set_tenant_context(db_session, tenant_id)
    remaining = await episodic_repo.get_unconsolidated_by_session(session_id, tenant_id, limit=10)
    assert remaining == []

    # Three real, distinct, Python-flavored episodes reflected on by a real
    # model should produce at least one fact -- if this ever starts failing,
    # that's a real regression in either the prompt or the model's behavior
    # worth knowing about, not a silently-skipped verification. Without this,
    # the store-verification loop below would pass vacuously on an empty
    # result and prove nothing.
    assert result, "expected at least one consolidated fact from three Python-related episodes"

    # Whatever WAS extracted must have genuinely reached both real stores,
    # WITH THE SAME ID -- not merely a row/point that happens to carry the
    # same fact_key or fact_value. find_by_key alone can't distinguish "the
    # row this command wrote" from "any row carrying that key," which is
    # exactly the gap that would have hidden a regression back to a random
    # (non-deterministic) id.
    for fact in result:
        found = await semantic_repo.find_by_key(user_id, fact.fact_key, tenant_id)
        assert found is not None
        assert found.id == fact.id
        assert found.fact_value == fact.fact_value
        from_index = await semantic_index.search(
            query_embedding=embedding_model.embed(fact.fact_value), user_id=user_id,
            tenant_id=tenant_id, top_k=5,
        )
        assert fact.id in {r.fact.id for r in from_index}
