import uuid
from datetime import UTC, datetime

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
    assert results[0].id == episode.id
    assert results[0].session_id == episode.session_id
    assert results[0].content == episode.content
    assert results[0].salience_score == episode.salience_score


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

    ids = {e.id for e in results}
    assert episode_a.id in ids
    assert episode_b.id not in ids
