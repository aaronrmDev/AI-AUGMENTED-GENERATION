"""Real end-to-end test for GateMemories: real Postgres-backed episodic and
semantic retrieval (Batch A/C infrastructure) and a real embedding model
feeding the gating pipeline -- reproducing MAG.md's own worked example
structurally (more candidates and tokens than the budget allows; gating
narrows to what fits and orders facts ahead of episodes), not with
hand-built fixtures alone.
"""
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.gating.gate_memories import GateMemories
from src.mag.domain.entities import EpisodicMemory, SemanticMemory
from src.mag.infrastructure.postgres_episodic_memory_repository import (
    PostgresEpisodicMemoryRepository,
)
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.shared.tokenization import count_tokens

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


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


async def test_gate_memories_narrows_real_retrieval_output_to_fit_a_tight_budget(
    db_session, embedding_model
):
    tenant_id = uuid.uuid4()
    user_id, session_id = await _insert_user_and_session(db_session, tenant_id)
    # _insert_user_and_session commits internally, which resets
    # set_tenant_context's transaction-local scoping (SET LOCAL) -- must be
    # re-applied before any further RLS-sensitive write, or the INSERT
    # below fails with "invalid input syntax for type uuid: ''" (confirmed
    # empirically, not assumed -- this exact failure mode is why every
    # other MAG integration test file in this project follows the same
    # re-call-after-every-commit pattern).
    await set_tenant_context(db_session, tenant_id)
    episode_repo = PostgresEpisodicMemoryRepository(db_session)
    fact_repo = PostgresSemanticMemoryRepository(db_session)
    now = datetime.now(UTC)

    # Five real episodes, three real facts -- real text, real embeddings,
    # more content than a tight token budget can hold, mirroring MAG.md's
    # own "50 retrieved memories totaling 15K tokens" oversupply framing at
    # a scale a real Postgres round trip can run quickly.
    episode_texts = [
        "The user asked how to read a CSV file in Python using pandas.",
        "The user asked about writing unit tests with pytest fixtures.",
        "The user asked how to format a string using an f-string in Python.",
        "The user asked about async/await syntax for coroutines in Python.",
        "The user asked how to raise a custom exception class in Python.",
    ]
    for i, text_ in enumerate(episode_texts):
        episode = EpisodicMemory(
            id=uuid.uuid4(), session_id=session_id, content={"input": text_},
            embedding=embedding_model.embed(text_), timestamp=now - timedelta(minutes=i),
        )
        await episode_repo.save(episode, tenant_id)
    await db_session.commit()
    await set_tenant_context(db_session, tenant_id)

    fact_values = [
        "The user's preferred programming language is Python.",
        "The user prefers pytest over unittest for testing.",
        "The user is currently working on an async data pipeline.",
    ]
    for i, fact_value in enumerate(fact_values):
        fact = SemanticMemory(
            id=uuid.uuid4(), user_id=user_id, fact_key=f"preference_{i}",
            fact_value=fact_value, embedding=embedding_model.embed(fact_value),
        )
        await fact_repo.save(fact, tenant_id)
    await db_session.commit()
    await set_tenant_context(db_session, tenant_id)

    query_embedding = embedding_model.embed("What does the user know about Python?")
    scored_episodes = await episode_repo.search_by_similarity(query_embedding, tenant_id, top_k=10)
    scored_facts = await fact_repo.search_by_similarity(
        query_embedding, user_id, tenant_id, top_k=10
    )
    assert len(scored_episodes) == 5
    assert len(scored_facts) == 3

    # A budget tight enough that not everything retrieved can fit -- real
    # tiktoken counts, not estimated: the full untrimmed pool costs more
    # tokens than this budget allows.
    full_pool_tokens = sum(count_tokens(s.episode.content["input"]) for s in scored_episodes) + sum(
        count_tokens(s.fact.fact_value) for s in scored_facts
    )
    tight_budget = full_pool_tokens // 2
    assert tight_budget > 0

    result = await GateMemories().execute(
        episodes=scored_episodes,
        facts=scored_facts,
        graph_nodes=[],
        token_budget=tight_budget,
    )

    assert len(result) < len(scored_episodes) + len(scored_facts)
    assert sum(count_tokens(c.content_text) for c in result) <= tight_budget

    # Hierarchical assembly ran last -- every fact in the final assembly
    # must be ordered ahead of every episode, matching MAG.md's own worked
    # example ("places user preferences ... facts first, adds supporting
    # episodic context after").
    source_types = [c.source_type for c in result]
    fact_indices = [i for i, t in enumerate(source_types) if t == "fact"]
    episode_indices = [i for i, t in enumerate(source_types) if t == "episode"]
    if fact_indices and episode_indices:
        assert max(fact_indices) < min(episode_indices)
