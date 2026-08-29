import uuid

import pytest

from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.application.commands.update_memory import UpdateMemory
from tests.unit.mag_fakes import (
    FakeMemoryGraphRepository,
    FakeSemanticMemoryIndex,
    FakeSemanticMemoryRepository,
)
from tests.unit.rag_fakes import FakeEmbeddingModel


def _wire(
    repository: FakeSemanticMemoryRepository,
) -> tuple[UpdateMemory, RecordSemanticFact]:
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=FakeMemoryGraphRepository(),
    )
    update = UpdateMemory(semantic_memory_repository=repository, record_semantic_fact=record)
    return update, record


async def test_execute_raises_when_no_existing_fact_for_the_key():
    update, _ = _wire(FakeSemanticMemoryRepository())

    with pytest.raises(ValueError, match="no existing fact"):
        await update.execute(
            tenant_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            fact_key="never_recorded",
            new_fact_value="anything",
        )


async def test_execute_overwrites_the_fact_value():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    update, record = _wire(repository)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="location", fact_value="New York"
    )

    updated = await update.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="location", new_fact_value="Berlin"
    )

    assert updated.fact_value == "Berlin"
    found = await repository.find_by_key(user_id, "location", tenant_id)
    assert found is not None
    assert found.fact_value == "Berlin"


async def test_execute_preserves_the_deterministic_id_across_the_overwrite():
    # The id is deterministic (uuid5 of user_id+fact_key) -- an update must
    # not accidentally mint a new one, or Qdrant would end up with two
    # points for what's supposed to be one evolving fact.
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    update, record = _wire(repository)
    original = await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="location", fact_value="New York"
    )

    updated = await update.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="location", new_fact_value="Berlin"
    )

    assert updated.id == original.id


async def test_execute_writes_the_old_value_to_history_before_overwriting():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    update, record = _wire(repository)
    await record.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="location",
        fact_value="New York",
        confidence=0.8,
        source="conversation",
    )

    await update.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="location", new_fact_value="Berlin"
    )

    history = await repository.find_history(user_id, "location", tenant_id)
    assert len(history) == 1
    entry = history[0]
    assert entry.fact_value == "New York"
    assert entry.confidence == 0.8
    assert entry.source == "conversation"
    assert entry.operation == "update"


async def test_execute_defaults_confidence_and_source_for_the_new_value():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    update, record = _wire(repository)
    await record.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="location",
        fact_value="New York",
        confidence=0.8,
        source="conversation",
    )

    updated = await update.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="location", new_fact_value="Berlin"
    )

    # UpdateMemory's own confidence/source parameters, not carried over
    # from the value being replaced -- a genuine correction is fresh
    # information, not an extension of the old, possibly-wrong confidence.
    assert updated.confidence == 1.0
    assert updated.source == ""
