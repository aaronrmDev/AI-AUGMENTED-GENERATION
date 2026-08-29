import uuid
from datetime import UTC, datetime

import pytest

from src.mag.domain.entities import EpisodicMemory
from src.mag.infrastructure.qdrant_episodic_memory_index import QdrantEpisodicMemoryIndex


def _episode(
    content: dict, embedding: list[float], session_id: uuid.UUID | None = None
) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id or uuid.uuid4(),
        content=content,
        embedding=embedding,
        timestamp=datetime.now(UTC),
        salience_score=0.5,
    )


async def test_upsert_then_search_finds_the_episode(qdrant_url):
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()

    tenant_id = uuid.uuid4()
    episode = _episode({"input": "hello", "output": "hi there"}, [0.1] * 384)
    await index.upsert(episode, tenant_id)

    results = await index.search(
        query_embedding=[0.1] * 384,
        tenant_id=tenant_id,
        session_id=episode.session_id,
        top_k=5,
    )

    assert len(results) == 1
    found = results[0].episode
    assert found.id == episode.id
    assert found.session_id == episode.session_id
    assert found.content == episode.content
    assert found.salience_score == episode.salience_score
    # Identical vectors, so cosine similarity is 1.0 -- the same COSINE-
    # distance-collection scale search_by_similarity's port docstring
    # documents for the Postgres side of this same score-carrying contract.
    assert results[0].score == pytest.approx(1.0)


async def test_search_returns_an_l2_normalized_embedding_not_the_raw_upserted_one(qdrant_url):
    # Symmetric with the semantic index's identical regression test
    # (test_qdrant_semantic_memory_index.py) -- this collection is also
    # configured for COSINE distance, so it normalizes every vector to unit
    # length on storage the same way; the port's own docstring
    # (EpisodicMemoryIndex.search) makes this claim and it needs the same
    # verification the sibling index already has, not just the same prose.
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()

    tenant_id = uuid.uuid4()
    raw_embedding = [0.3] * 384
    episode = _episode({"input": "hello"}, raw_embedding)
    await index.upsert(episode, tenant_id)

    results = await index.search(
        query_embedding=raw_embedding,
        tenant_id=tenant_id,
        session_id=episode.session_id,
        top_k=1,
    )

    assert len(results) == 1
    norm = sum(x * x for x in raw_embedding) ** 0.5
    expected_normalized = [x / norm for x in raw_embedding]
    assert results[0].episode.embedding == pytest.approx(expected_normalized, rel=1e-4)


async def test_search_never_returns_another_tenants_episodes(qdrant_url):
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()

    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    # Same session_id under both tenants, isolating tenant_id as the only
    # variable this test probes -- session isolation is a separate test below.
    session_id = uuid.uuid4()
    episode_a = _episode({"who": "a"}, [0.2] * 384, session_id=session_id)
    episode_b = _episode({"who": "b"}, [0.2] * 384, session_id=session_id)
    await index.upsert(episode_a, tenant_a)
    await index.upsert(episode_b, tenant_b)

    results = await index.search(
        query_embedding=[0.2] * 384, tenant_id=tenant_a, session_id=session_id, top_k=10
    )

    ids = {s.episode.id for s in results}
    assert episode_a.id in ids
    assert episode_b.id not in ids


async def test_search_never_returns_another_sessions_episodes_within_the_same_tenant(qdrant_url):
    # Regression test: an earlier version of search() filtered only by
    # tenant_id, so SemanticSimilarityRetrieval (the sole caller of this
    # method) silently returned another session's episodes within the same
    # tenant -- a real cross-session leak a Batch C review caught, directly
    # contradicting this batch's own "no cross-session retrieval" design.
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()

    tenant_id = uuid.uuid4()
    session_a = uuid.uuid4()
    session_b = uuid.uuid4()
    episode_a = _episode({"who": "a"}, [0.4] * 384, session_id=session_a)
    episode_b = _episode({"who": "b"}, [0.4] * 384, session_id=session_b)
    await index.upsert(episode_a, tenant_id)
    await index.upsert(episode_b, tenant_id)

    results = await index.search(
        query_embedding=[0.4] * 384, tenant_id=tenant_id, session_id=session_a, top_k=10
    )

    ids = {s.episode.id for s in results}
    assert episode_a.id in ids
    assert episode_b.id not in ids
