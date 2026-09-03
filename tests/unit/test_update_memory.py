import uuid
from datetime import UTC, datetime

import pytest

from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.application.commands.update_memory import UpdateMemory
from src.mag.domain.entities import SemanticMemoryHistoryEntry
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


async def test_execute_preserves_a_previously_set_valid_until():
    # Complements the integration-level regression test for archived_at
    # (test_update_on_a_previously_archived_fact_preserves_its_archived_
    # status) -- that test alone can't prove valid_until forwarding works
    # too, since its fixture only ever archives (never invalidates)
    # first, leaving valid_until at None throughout (indistinguishable
    # from the pre-fix default). This exercises the OTHER field, at the
    # unit level, closing a coverage gap review flagged: only 1 of the 4
    # (UpdateMemory/RefineMemory x valid_until/archived_at) combinations
    # had any test at all.
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    update, record = _wire(repository)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="pet", fact_value="has a pet named Rex"
    )
    invalidated_at = datetime(2026, 1, 1, tzinfo=UTC)
    await repository.invalidate(user_id, "pet", tenant_id, invalidated_at)

    updated = await update.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="pet", new_fact_value="got a new pet"
    )

    assert updated.valid_until == invalidated_at
    assert updated.fact_value == "got a new pet"


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


async def test_find_history_orders_multiple_entries_newest_first():
    # find_history's port docstring documents "newest first" -- every
    # existing test only ever produced a single history row, which can't
    # actually prove ordering (any order of one element is trivially
    # correct), and a fact genuinely updated more than once is normal,
    # real usage. Explicit, manually-differentiated superseded_at values
    # (rather than two back-to-back datetime.now(UTC) calls through
    # UpdateMemory, which risks landing on the identical microsecond on
    # a fast machine) make this deterministic rather than timing-flaky.
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    original_fact_id = uuid.uuid4()
    older = SemanticMemoryHistoryEntry(
        id=uuid.uuid4(), original_fact_id=original_fact_id, user_id=user_id,
        fact_key="location", fact_value="New York", confidence=1.0, source="",
        operation="update", superseded_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer = SemanticMemoryHistoryEntry(
        id=uuid.uuid4(), original_fact_id=original_fact_id, user_id=user_id,
        fact_key="location", fact_value="Chicago", confidence=1.0, source="",
        operation="update", superseded_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    await repository.save_history_entry(older, tenant_id)
    await repository.save_history_entry(newer, tenant_id)

    history = await repository.find_history(user_id, "location", tenant_id)

    assert [entry.fact_value for entry in history] == ["Chicago", "New York"]


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
