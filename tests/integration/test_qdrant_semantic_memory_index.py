import uuid

from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex


async def test_upsert_then_search_finds_the_fact(qdrant_url):
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()

    user_id = uuid.uuid4()
    fact = SemanticMemory(
        id=uuid.uuid4(), user_id=user_id, fact_key="favorite_color", fact_value="blue",
        embedding=[0.1] * 384,
    )
    await index.upsert(fact)

    results = await index.search(query_embedding=[0.1] * 384, user_id=user_id, top_k=5)
    assert len(results) == 1
    assert results[0].id == fact.id
    assert results[0].fact_key == "favorite_color"
    assert results[0].fact_value == "blue"


async def test_search_never_returns_another_users_facts(qdrant_url):
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()

    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    fact_a = SemanticMemory(
        id=uuid.uuid4(), user_id=user_a, fact_key="favorite_color", fact_value="user a's fact",
        embedding=[0.2] * 384,
    )
    fact_b = SemanticMemory(
        id=uuid.uuid4(), user_id=user_b, fact_key="favorite_color", fact_value="user b's fact",
        embedding=[0.2] * 384,
    )
    await index.upsert(fact_a)
    await index.upsert(fact_b)

    results = await index.search(query_embedding=[0.2] * 384, user_id=user_a, top_k=10)
    fact_ids = {r.id for r in results}
    assert fact_a.id in fact_ids
    assert fact_b.id not in fact_ids
