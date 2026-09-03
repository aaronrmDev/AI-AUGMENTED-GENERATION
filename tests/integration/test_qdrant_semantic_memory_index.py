import uuid
from datetime import UTC, datetime

import pytest

from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex


async def test_upsert_then_search_finds_the_fact(qdrant_url):
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()

    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    fact = SemanticMemory(
        id=uuid.uuid4(), user_id=user_id, fact_key="favorite_color", fact_value="blue",
        embedding=[0.1] * 384,
    )
    await index.upsert(fact, tenant_id)

    results = await index.search(
        query_embedding=[0.1] * 384, user_id=user_id, tenant_id=tenant_id, top_k=5
    )
    assert len(results) == 1
    found = results[0].fact
    assert found.id == fact.id
    assert found.fact_key == "favorite_color"
    assert found.fact_value == "blue"
    assert results[0].score == pytest.approx(1.0)


async def test_search_returns_the_real_embedding_confidence_source_and_valid_until(qdrant_url):
    # Regression test: an earlier version of this index only stored
    # user_id/fact_key/fact_value in the payload, so a search-based read
    # silently returned confidence=1.0, source="", valid_until=None for
    # every fact regardless of what was actually recorded -- the dataclass
    # defaults masking the missing data rather than raising.
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()

    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    valid_until = datetime(2030, 1, 1, tzinfo=UTC)
    fact = SemanticMemory(
        id=uuid.uuid4(), user_id=user_id, fact_key="employment_status",
        fact_value="unemployed as of last conversation", embedding=[0.3] * 384,
        confidence=0.4, source="inferred", valid_until=valid_until,
    )
    await index.upsert(fact, tenant_id)

    results = await index.search(
        query_embedding=[0.3] * 384, user_id=user_id, tenant_id=tenant_id, top_k=1
    )

    assert len(results) == 1
    found = results[0].fact
    assert found.confidence == pytest.approx(0.4)
    assert found.source == "inferred"
    assert found.valid_until == valid_until
    assert len(found.embedding) == 384
    # NOT bit-for-bit equal to what was upserted: a COSINE-distance
    # collection (this one is, per ensure_collection's Distance.COSINE)
    # normalizes every vector to unit length on storage, so with_vectors=True
    # hands back the normalized vector, not the original -- confirmed
    # empirically (a [0.3]*384 input search-round-trips to
    # [1/sqrt(384)]*384, exactly unit-length in the same direction), not
    # assumed. Same-direction, unit-length is what search correctness
    # actually depends on; downstream consumers of this embedding (Memory
    # Graphs writing it into Neo4j's Entity.embedding index) need to know
    # they're getting the normalized form, not the literal original.
    norm = sum(x * x for x in fact.embedding) ** 0.5
    expected_normalized = [x / norm for x in fact.embedding]
    assert found.embedding == pytest.approx(expected_normalized, rel=1e-4)


async def test_search_returns_none_for_valid_until_when_the_fact_never_had_one(qdrant_url):
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()

    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    fact = SemanticMemory(
        id=uuid.uuid4(), user_id=user_id, fact_key="favorite_color", fact_value="blue",
        embedding=[0.15] * 384, valid_until=None,
    )
    await index.upsert(fact, tenant_id)

    results = await index.search(
        query_embedding=[0.15] * 384, user_id=user_id, tenant_id=tenant_id, top_k=1
    )

    assert results[0].fact.valid_until is None


async def test_search_never_returns_another_users_facts(qdrant_url):
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()

    tenant_id = uuid.uuid4()
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
    await index.upsert(fact_a, tenant_id)
    await index.upsert(fact_b, tenant_id)

    results = await index.search(
        query_embedding=[0.2] * 384, user_id=user_a, tenant_id=tenant_id, top_k=10
    )
    fact_ids = {r.fact.id for r in results}
    assert fact_a.id in fact_ids
    assert fact_b.id not in fact_ids


async def test_search_never_returns_another_tenants_facts(qdrant_url):
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()

    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    user_id = uuid.uuid4()
    fact_a = SemanticMemory(
        id=uuid.uuid4(), user_id=user_id, fact_key="favorite_color", fact_value="tenant a's fact",
        embedding=[0.25] * 384,
    )
    fact_b = SemanticMemory(
        id=uuid.uuid4(), user_id=user_id, fact_key="favorite_color", fact_value="tenant b's fact",
        embedding=[0.25] * 384,
    )
    await index.upsert(fact_a, tenant_a)
    await index.upsert(fact_b, tenant_b)

    results = await index.search(
        query_embedding=[0.25] * 384, user_id=user_id, tenant_id=tenant_a, top_k=10
    )
    fact_ids = {r.fact.id for r in results}
    assert fact_a.id in fact_ids
    assert fact_b.id not in fact_ids
