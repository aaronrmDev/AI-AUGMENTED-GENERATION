"""Two-part real-dependency test for CausalRetrieval.

The first test exercises the real Postgres candidate-fetch path (real
PostgresEpisodicMemoryRepository, testcontainers, no mocks) against a
FakeChatModel with a scripted response -- it's about the Postgres
get_by_session round trip feeding correctly into the scoring/ranking logic,
not about real LLM judgment.

The second test uses a real Ollama model (not a fake) for the causal scoring
call itself, mirroring test_consolidate_episodes_command.py's reasoning: a
fake chat model can prove the retry/parsing logic works against a SCRIPTED
response, but it can't answer the actual empirical question this project's
testing discipline cares about -- does a real model, given a query and a mix
of causal and unrelated episodes, actually score the causal one higher.
Requires Ollama running locally with qwen3.5 pulled (same model this
project's other live MAG/RAG tests already depend on).
"""
import json
import uuid
from datetime import UTC, datetime

import ollama
from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.queries.retrieve_by_causal_relevance import CausalRetrieval
from src.mag.domain.entities import EpisodicMemory
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from tests.unit.rag_fakes import FakeChatModel

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


def _episode(session_id: uuid.UUID, content: dict, timestamp: datetime) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(), session_id=session_id, content=content,
        embedding=[0.0] * 384, timestamp=timestamp,
    )


async def test_execute_ranks_real_postgres_episodes_using_a_scripted_scorer(db_session):
    tenant_id = uuid.uuid4()
    _, session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    now = datetime.now(UTC)
    low = _episode(session_id, {"input": "what's the weather"}, now)
    high = _episode(session_id, {"input": "deploy failed", "output": "root cause: bad config"}, now)
    await repo.save(low, tenant_id)
    await repo.save(high, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    # get_by_session returns oldest-first (save order here): low is episode
    # 1, high is episode 2 -- the scripted response scores them accordingly.
    scripted = json.dumps(
        {"scores": [{"episode_index": 1, "score": 0.1}, {"episode_index": 2, "score": 0.9}]}
    )
    chat_model = FakeChatModel(response=scripted)
    use_case = CausalRetrieval(episodic_memory_repository=repo, chat_model=chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did the deployment fail", top_k=2
    )

    assert [r.episode.id for r in result] == [high.id, low.id]
    assert [r.score for r in result] == [0.9, 0.1]


async def test_execute_scores_a_causal_episode_higher_than_unrelated_with_real_ollama(
    db_session,
):
    tenant_id = uuid.uuid4()
    _, session_id = await _insert_user_and_session(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresEpisodicMemoryRepository(db_session)
    now = datetime.now(UTC)
    causal = _episode(
        session_id,
        {
            "input": "The deployment is failing, can you help?",
            "reasoning_trace": (
                "Checked the logs: Traceback (most recent call last):\n"
                '  File "deploy.py", line 42, in <module>\n'
                "    config = load_config(path)\n"
                "KeyError: 'DATABASE_URL'\n"
                "Root cause: the DATABASE_URL environment variable was never set "
                "in the production deployment manifest."
            ),
            "output": (
                "Found it -- the deploy script crashed with a KeyError because "
                "DATABASE_URL was missing from the environment. Fix: added "
                "DATABASE_URL to the production manifest and redeployed "
                "successfully."
            ),
        },
        now,
    )
    unrelated = _episode(
        session_id,
        {
            "input": "What's a good name for a pet goldfish?",
            "output": "Some popular goldfish names are Bubbles, Nemo, and Finn.",
        },
        now,
    )
    await repo.save(unrelated, tenant_id)
    await repo.save(causal, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    chat_model = OllamaChatModel(client=ollama.AsyncClient(), model_id=_MODEL_ID)
    use_case = CausalRetrieval(episodic_memory_repository=repo, chat_model=chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=2
    )

    by_id = {r.episode.id: r.score for r in result}
    causal_score = by_id[causal.id]
    unrelated_score = by_id[unrelated.id]
    print(
        f"\nCausal relevance scores from a real Ollama model for query 'why did it fail':\n"
        f"  causal episode (traceback + root cause + fix): {causal_score}\n"
        f"  unrelated episode (goldfish names): {unrelated_score}"
    )

    assert causal_score > unrelated_score
