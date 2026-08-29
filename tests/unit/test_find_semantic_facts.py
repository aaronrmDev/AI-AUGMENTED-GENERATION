import uuid

from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
from src.mag.domain.entities import ScoredFact, SemanticMemory
from tests.unit.mag_fakes import FakeSemanticMemoryRepository


def _fact(user_id: uuid.UUID, fact_key: str = "favorite_color") -> SemanticMemory:
    return SemanticMemory(
        id=uuid.uuid4(),
        user_id=user_id,
        fact_key=fact_key,
        fact_value="blue",
        embedding=[0.1] * 384,
    )


async def test_by_key_delegates_to_find_by_key():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    fact = _fact(user_id)
    await repository.save(fact, tenant_id)

    query = FindSemanticFacts(semantic_memory_repository=repository)
    result = await query.by_key(user_id, "favorite_color", tenant_id)

    assert result == fact


async def test_by_key_returns_none_for_an_unknown_key():
    query = FindSemanticFacts(semantic_memory_repository=FakeSemanticMemoryRepository())

    result = await query.by_key(uuid.uuid4(), "unknown_key", uuid.uuid4())

    assert result is None


async def test_by_similarity_delegates_to_search_by_similarity():
    repository = FakeSemanticMemoryRepository()
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    facts = [_fact(user_id, fact_key=f"key_{i}") for i in range(3)]
    repository.set_search_results(facts)

    query = FindSemanticFacts(semantic_memory_repository=repository)
    result = await query.by_similarity(
        query_embedding=[0.1] * 384, user_id=user_id, tenant_id=tenant_id, top_k=2
    )

    # All three fixtures share the query's own embedding, so every score
    # ties at cosine similarity 1.0 and a stable sort preserves insertion
    # order -- this asserts delegation and top_k truncation, not ranking.
    assert result == [ScoredFact(fact=f, score=1.0) for f in facts[:2]]
