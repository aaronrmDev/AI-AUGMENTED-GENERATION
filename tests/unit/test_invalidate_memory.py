import uuid
from datetime import UTC, datetime

import pytest

from src.mag.application.commands.invalidate_memory import InvalidateMemory
from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from tests.unit.mag_fakes import (
    FakeMemoryGraphRepository,
    FakeSemanticMemoryIndex,
    FakeSemanticMemoryRepository,
)
from tests.unit.rag_fakes import FakeEmbeddingModel


def _wire(
    repository: FakeSemanticMemoryRepository,
    index: FakeSemanticMemoryIndex,
    graph: FakeMemoryGraphRepository,
) -> tuple[InvalidateMemory, RecordSemanticFact]:
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=graph,
    )
    invalidate = InvalidateMemory(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        memory_graph_repository=graph,
    )
    return invalidate, record


async def test_execute_raises_when_no_existing_fact_for_the_key():
    invalidate, _ = _wire(
        FakeSemanticMemoryRepository(), FakeSemanticMemoryIndex(), FakeMemoryGraphRepository()
    )

    with pytest.raises(ValueError, match="no existing fact"):
        await invalidate.execute(
            tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), fact_key="never_recorded"
        )


async def test_execute_sets_valid_until_and_leaves_fact_value_unchanged():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    invalidate, record = _wire(repository, FakeSemanticMemoryIndex(), FakeMemoryGraphRepository())
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="pet", fact_value="has a pet named Rex"
    )

    invalidated_at = datetime(2026, 1, 1, tzinfo=UTC)
    result = await invalidate.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="pet", invalidated_at=invalidated_at
    )

    assert result.valid_until == invalidated_at
    assert result.fact_value == "has a pet named Rex"
    found = await repository.find_by_key(user_id, "pet", tenant_id)
    assert found is not None
    assert found.valid_until == invalidated_at


async def test_execute_defaults_invalidated_at_to_now():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    invalidate, record = _wire(repository, FakeSemanticMemoryIndex(), FakeMemoryGraphRepository())
    await record.execute(tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="v")
    before = datetime.now(UTC)

    result = await invalidate.execute(tenant_id=tenant_id, user_id=user_id, fact_key="k")

    assert result.valid_until is not None
    assert result.valid_until >= before


async def test_execute_invalidated_fact_is_excluded_from_search_by_similarity():
    # set_search_results is this fake's own decoupled "what a search would
    # currently see" control point (independent of what's saved by key) --
    # so this test seeds it with the POST-invalidation entity execute()
    # actually returns, then proves the repository's own filter (the same
    # filter the real Postgres WHERE clause applies) drops it.
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    embedder = FakeEmbeddingModel()
    invalidate, record = _wire(repository, FakeSemanticMemoryIndex(), FakeMemoryGraphRepository())
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="pet", fact_value="has a pet named Rex"
    )
    invalidated_at = datetime(2026, 1, 1, tzinfo=UTC)

    invalidated = await invalidate.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="pet", invalidated_at=invalidated_at
    )
    repository.set_search_results([invalidated])

    results = await repository.search_by_similarity(
        embedder.embed("has a pet named Rex"), user_id, tenant_id, top_k=10
    )
    assert results == []


async def test_execute_syncs_status_to_the_index_without_touching_the_vector():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    index = FakeSemanticMemoryIndex()
    invalidate, record = _wire(repository, index, FakeMemoryGraphRepository())
    fact = await record.execute(tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="v")
    invalidated_at = datetime(2026, 1, 1, tzinfo=UTC)

    await invalidate.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", invalidated_at=invalidated_at
    )

    assert index.status_updates == [(fact.id, tenant_id, invalidated_at, None)]
    # update_status was called, not upsert() again -- upsert() would have
    # replaced the point's vector with the embedding-less entity find_by_key
    # returns.
    assert len(index.upserted) == 1  # only RecordSemanticFact's original upsert


async def test_execute_best_effort_syncs_the_graph_fact_node():
    graph = FakeMemoryGraphRepository()
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    invalidate, record = _wire(repository, FakeSemanticMemoryIndex(), graph)
    await record.execute(tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="v")
    invalidated_at = datetime(2026, 1, 1, tzinfo=UTC)

    result = await invalidate.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", invalidated_at=invalidated_at
    )

    assert graph.upserted_facts[-1] == (result, tenant_id)
    assert graph.upserted_facts[-1][0].valid_until == invalidated_at
