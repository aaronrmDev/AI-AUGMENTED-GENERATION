"""Real end-to-end test for RecencyDecayFusionRetrieval: real Postgres, real
Qdrant, and a real Ollama model (not a fake) for the causal-scoring leg --
same reasoning as test_consolidate_episodes_command.py and
test_retrieve_by_causal_relevance_query.py's live tests. Requires Ollama
running locally with qwen3.5 pulled.
"""
import uuid
from datetime import UTC, datetime, timedelta

import ollama
from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.queries.retrieve_with_recency_decay_fusion import (
    RecencyDecayFusionRetrieval,
)
from src.mag.domain.entities import EpisodicMemory
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)
from src.mag.infrastructure.qdrant_episodic_memory_index import QdrantEpisodicMemoryIndex
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
_MODEL_ID = "qwen3.5"


async def _insert_user_and_session(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
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


def _episode(
    session_id: uuid.UUID,
    content: dict,
    embedding: list[float],
    timestamp: datetime,
    salience_score: float = 0.0,
) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id,
        content=content,
        embedding=embedding,
        timestamp=timestamp,
        salience_score=salience_score,
    )


async def test_a_real_fusion_pass_ranks_a_causally_relevant_recent_episode_above_a_routine_one(
    db_session, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()
    now = datetime.now(UTC)

    failure_content = {
        "input": "deploy the payments service to production",
        "reasoning": "ran the deploy script without checking the health endpoint first",
        "output": "Traceback (most recent call last): ConnectionRefusedError: "
        "database connection failed",
        "outcome": "failure",
        "entities": ["payments-service"],
    }
    failure_text = "deployment failure database connection refused payments service"
    failure_episode = _episode(
        session_id, failure_content, embedding_model.embed(failure_text),
        timestamp=now, salience_score=0.9,
    )

    routine_content = {
        "input": "what's the weather like today", "output": "sunny", "outcome": "success",
    }
    routine_episode = _episode(
        session_id, routine_content, embedding_model.embed("weather is sunny today"),
        timestamp=now - timedelta(days=3), salience_score=0.1,
    )

    await repo.save(failure_episode, tenant_id)
    await repo.save(routine_episode, tenant_id)
    await db_session.commit()
    await index.upsert(failure_episode, tenant_id)
    await index.upsert(routine_episode, tenant_id)

    await set_tenant_context(db_session, tenant_id)
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    fusion = RecencyDecayFusionRetrieval(
        episodic_memory_repository=repo, episodic_memory_index=index, chat_model=chat_model,
    )

    result = await fusion.execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=2,
        query_embedding=embedding_model.embed(failure_text),
        causal_query="why did the deployment fail",
        entity="payments-service",
        now=now,
    )

    print(
        f"\nFusion (real Postgres+Qdrant+Ollama, {_MODEL_ID}): "
        + ", ".join(f"{s.episode.content.get('outcome')}={s.score:.4f}" for s in result)
    )
    assert len(result) == 2
    assert result[0].episode.id == failure_episode.id
    assert result[0].score > result[1].score


async def test_a_real_fusion_pass_isolates_the_causal_leg_via_weights(
    db_session, qdrant_url, embedding_model
):
    # The test above proves the full pipeline agrees on the obvious answer,
    # but a Batch C review caught that it can't actually tell whether the
    # real Ollama causal call is doing meaningful work: salience, recency,
    # semantic similarity, and entity match ALL independently favor the same
    # episode by a wide margin there, so causal's leg could silently fall
    # back to its flat 0.0-for-everyone failure mode (see
    # retrieve_by_causal_relevance.py's exhausted-retry path) and the test
    # would still pass. This test isolates causal as the ONLY leg that can
    # possibly differentiate the two episodes: identical salience_score,
    # identical timestamp (so decay is identical too), and no
    # query_embedding/entity given at all (so semantic/entity never run).
    # weights zeroes out temporal and salience explicitly, so even though
    # they still execute, only causal's real Ollama-produced score
    # determines the fused ranking.
    tenant_id = uuid.uuid4()
    session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()
    now = datetime.now(UTC)

    causal_content = {
        "input": "deploy the payments service to production",
        "reasoning": "ran the deploy script without checking the health endpoint first",
        "output": "Traceback (most recent call last): ConnectionRefusedError: "
        "database connection failed",
        "outcome": "failure",
    }
    causal_episode = _episode(
        session_id, causal_content, embedding_model.embed("irrelevant"),
        timestamp=now, salience_score=0.5,
    )

    unrelated_content = {"input": "what are some good names for a goldfish", "output": "Bubbles"}
    unrelated_episode = _episode(
        session_id, unrelated_content, embedding_model.embed("irrelevant2"),
        timestamp=now, salience_score=0.5,
    )

    await repo.save(causal_episode, tenant_id)
    await repo.save(unrelated_episode, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    fusion = RecencyDecayFusionRetrieval(
        episodic_memory_repository=repo, episodic_memory_index=index, chat_model=chat_model,
    )

    result = await fusion.execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=2,
        causal_query="why did it fail",
        now=now,
        weights={"temporal": 0.0, "salience": 0.0, "causal": 1.0},
    )

    print(
        f"\nFusion causal-isolated (real Postgres+Ollama, {_MODEL_ID}): "
        + ", ".join(f"{s.episode.content.get('outcome')}={s.score:.4f}" for s in result)
    )
    assert len(result) == 2
    assert result[0].episode.id == causal_episode.id
    assert result[0].score > result[1].score
