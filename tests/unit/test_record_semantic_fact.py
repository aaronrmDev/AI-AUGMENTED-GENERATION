import uuid
from datetime import UTC, datetime

from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from tests.unit.mag_fakes import (
    FakeMemoryGraphRepository,
    FakeSemanticMemoryIndex,
    FakeSemanticMemoryRepository,
)
from tests.unit.rag_fakes import FakeEmbeddingModel


async def test_execute_saves_to_both_the_repository_and_the_index():
    repository = FakeSemanticMemoryRepository()
    index = FakeSemanticMemoryIndex()
    command = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=FakeMemoryGraphRepository(),
    )

    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    fact = await command.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="favorite_color", fact_value="blue"
    )

    assert repository.saved == [(fact, tenant_id)]
    assert index.upserted == [(fact, tenant_id)]


async def test_execute_embeds_the_fact_value_not_the_fact_key():
    repository = FakeSemanticMemoryRepository()
    index = FakeSemanticMemoryIndex()
    embedder = FakeEmbeddingModel()
    command = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=embedder,
        memory_graph_repository=FakeMemoryGraphRepository(),
    )

    fact = await command.execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        fact_key="k",
        fact_value="a much longer fact value string",
    )

    assert fact.embedding == embedder.embed("a much longer fact value string")
    assert fact.embedding != embedder.embed("k")


async def test_execute_derives_the_same_id_for_the_same_user_and_fact_key():
    # The id is deterministic (uuid5 of user_id+fact_key), not random --
    # this is the actual invariant that keeps a re-recorded fact from
    # orphaning a stale point in Qdrant (see the comment in
    # record_semantic_fact.py and the integration-level regression test
    # test_recording_the_same_key_twice_does_not_orphan_a_stale_qdrant_point).
    # A unit test asserting only "two DIFFERENT keys get different ids"
    # can't tell a correct deterministic scheme apart from a reverted
    # uuid4() -- this test is the fast, Docker-free guard for the specific
    # property that matters.
    command = RecordSemanticFact(
        semantic_memory_repository=FakeSemanticMemoryRepository(),
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=FakeMemoryGraphRepository(),
    )
    user_id = uuid.uuid4()

    first = await command.execute(
        tenant_id=uuid.uuid4(), user_id=user_id, fact_key="favorite_color", fact_value="blue"
    )
    second = await command.execute(
        tenant_id=uuid.uuid4(), user_id=user_id, fact_key="favorite_color", fact_value="red"
    )
    different_key = await command.execute(
        tenant_id=uuid.uuid4(), user_id=user_id, fact_key="favorite_language", fact_value="v"
    )
    different_user = await command.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), fact_key="favorite_color", fact_value="v"
    )

    assert isinstance(first.id, uuid.UUID)
    assert first.id == second.id
    assert first.id != different_key.id
    assert first.id != different_user.id


async def test_execute_defaults_confidence_to_one():
    command = RecordSemanticFact(
        semantic_memory_repository=FakeSemanticMemoryRepository(),
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=FakeMemoryGraphRepository(),
    )

    fact = await command.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), fact_key="k", fact_value="v"
    )

    assert fact.confidence == 1.0


async def test_execute_stores_the_given_user_id_fact_key_and_fact_value():
    command = RecordSemanticFact(
        semantic_memory_repository=FakeSemanticMemoryRepository(),
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=FakeMemoryGraphRepository(),
    )

    user_id = uuid.uuid4()
    fact = await command.execute(
        tenant_id=uuid.uuid4(),
        user_id=user_id,
        fact_key="favorite_color",
        fact_value="blue",
        confidence=0.7,
        source="conversation",
        valid_until=datetime(2030, 1, 1, tzinfo=UTC),
    )

    assert fact.user_id == user_id
    assert fact.fact_key == "favorite_color"
    assert fact.fact_value == "blue"
    assert fact.confidence == 0.7
    assert fact.source == "conversation"
    assert fact.valid_until == datetime(2030, 1, 1, tzinfo=UTC)


async def test_execute_recording_the_same_key_twice_updates_the_fake_not_duplicates():
    # Mirrors the real repository's ON CONFLICT (user_id, fact_key) DO UPDATE
    # -- this fake's upsert semantics (see mag_fakes.py) must agree with it,
    # or a unit test here proves nothing about the real integration behavior.
    repository = FakeSemanticMemoryRepository()
    command = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=FakeMemoryGraphRepository(),
    )
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()

    await command.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="favorite_color", fact_value="blue"
    )
    await command.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="favorite_color", fact_value="red"
    )

    found = await repository.find_by_key(user_id, "favorite_color", tenant_id)
    assert found is not None
    assert found.fact_value == "red"


async def test_execute_upserts_a_fact_node_into_the_memory_graph():
    graph = FakeMemoryGraphRepository()
    command = RecordSemanticFact(
        semantic_memory_repository=FakeSemanticMemoryRepository(),
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=graph,
    )
    tenant_id = uuid.uuid4()

    fact = await command.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), fact_key="k", fact_value="v"
    )

    assert graph.upserted_facts == [(fact, tenant_id)]
