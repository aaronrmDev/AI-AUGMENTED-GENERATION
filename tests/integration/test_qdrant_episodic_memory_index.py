import uuid
from datetime import UTC, datetime

import pytest

from src.mag.domain.entities import EpisodicMemory
from src.mag.infrastructure.qdrant_episodic_memory_index import QdrantEpisodicMemoryIndex


def _episode(content: dict, embedding: list[float]) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
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

    results = await index.search(query_embedding=[0.1] * 384, tenant_id=tenant_id, top_k=5)

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

    results = await index.search(query_embedding=raw_embedding, tenant_id=tenant_id, top_k=1)

    assert len(results) == 1
    norm = sum(x * x for x in raw_embedding) ** 0.5
    expected_normalized = [x / norm for x in raw_embedding]
    assert results[0].episode.embedding == pytest.approx(expected_normalized, rel=1e-4)


async def test_search_never_returns_another_tenants_episodes(qdrant_url):
    index = QdrantEpisodicMemoryIndex(qdrant_url)
    await index.ensure_collection()

    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    episode_a = _episode({"who": "a"}, [0.2] * 384)
    episode_b = _episode({"who": "b"}, [0.2] * 384)
    await index.upsert(episode_a, tenant_a)
    await index.upsert(episode_b, tenant_b)

    results = await index.search(query_embedding=[0.2] * 384, tenant_id=tenant_a, top_k=10)

    ids = {s.episode.id for s in results}
    assert episode_a.id in ids
    assert episode_b.id not in ids
