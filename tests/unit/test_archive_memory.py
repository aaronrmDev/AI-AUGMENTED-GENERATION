import uuid
from datetime import UTC, datetime

import pytest

from src.mag.application.commands.archive_memory import ArchiveMemory
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
) -> tuple[ArchiveMemory, RecordSemanticFact]:
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=graph,
    )
    archive = ArchiveMemory(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        memory_graph_repository=graph,
    )
    return archive, record


async def test_execute_raises_when_no_existing_fact_for_the_key():
    archive, _ = _wire(
        FakeSemanticMemoryRepository(), FakeSemanticMemoryIndex(), FakeMemoryGraphRepository()
    )

    with pytest.raises(ValueError, match="no existing fact"):
        await archive.execute(
            tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), fact_key="never_recorded"
        )


async def test_execute_sets_archived_at_and_leaves_fact_value_unchanged():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    archive, record = _wire(repository, FakeSemanticMemoryIndex(), FakeMemoryGraphRepository())
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="rarely needed but true"
    )

    archived_at = datetime(2026, 1, 1, tzinfo=UTC)
    result = await archive.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", archived_at=archived_at
    )

    assert result.archived_at == archived_at
    assert result.fact_value == "rarely needed but true"
    found = await repository.find_by_key(user_id, "k", tenant_id)
    assert found is not None
    assert found.archived_at == archived_at


async def test_execute_defaults_archived_at_to_now():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    archive, record = _wire(repository, FakeSemanticMemoryIndex(), FakeMemoryGraphRepository())
    await record.execute(tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="v")
    before = datetime.now(UTC)

    result = await archive.execute(tenant_id=tenant_id, user_id=user_id, fact_key="k")

    assert result.archived_at is not None
    assert result.archived_at >= before


async def test_execute_archived_fact_is_excluded_from_search_but_still_findable_by_key():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    embedder = FakeEmbeddingModel()
    archive, record = _wire(repository, FakeSemanticMemoryIndex(), FakeMemoryGraphRepository())
    await record.execute(tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="v")

    archived = await archive.execute(tenant_id=tenant_id, user_id=user_id, fact_key="k")
    repository.set_search_results([archived])

    search_results = await repository.search_by_similarity(
        embedder.embed("v"), user_id, tenant_id, top_k=10
    )
    assert search_results == []
    # "Move to cold storage while keeping it available for reference" --
    # unlike invalidated facts, an archived one is still meant to be
    # reachable by a direct, keyed lookup.
    direct = await repository.find_by_key(user_id, "k", tenant_id)
    assert direct is not None
    assert direct.archived_at is not None


async def test_execute_syncs_status_to_the_index_without_touching_the_vector():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    index = FakeSemanticMemoryIndex()
    archive, record = _wire(repository, index, FakeMemoryGraphRepository())
    fact = await record.execute(tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="v")
    archived_at = datetime(2026, 1, 1, tzinfo=UTC)

    await archive.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", archived_at=archived_at
    )

    assert index.status_updates == [(fact.id, tenant_id, None, archived_at)]
    assert len(index.upserted) == 1


async def test_execute_best_effort_syncs_the_graph_fact_node():
    graph = FakeMemoryGraphRepository()
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    archive, record = _wire(repository, FakeSemanticMemoryIndex(), graph)
    await record.execute(tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="v")
    archived_at = datetime(2026, 1, 1, tzinfo=UTC)

    result = await archive.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", archived_at=archived_at
    )

    assert graph.upserted_facts[-1] == (result, tenant_id)
    assert graph.upserted_facts[-1][0].archived_at == archived_at
