import uuid
from datetime import UTC, datetime

from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.domain.entities import SemanticMemory
from tests.unit.mag_fakes import FakeSemanticMemoryRepository
from tests.unit.rag_fakes import FakeEmbeddingModel


class FakeSemanticMemoryIndex:
    def __init__(self) -> None:
        self.upserted: list[SemanticMemory] = []

    async def upsert(self, fact: SemanticMemory) -> None:
        self.upserted.append(fact)


async def test_execute_saves_to_both_the_repository_and_the_index():
    repository = FakeSemanticMemoryRepository()
    index = FakeSemanticMemoryIndex()
    command = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=FakeEmbeddingModel(),
    )

    user_id = uuid.uuid4()
    fact = await command.execute(user_id=user_id, fact_key="favorite_color", fact_value="blue")

    assert repository.saved == [fact]
    assert index.upserted == [fact]


async def test_execute_embeds_the_fact_value_not_the_fact_key():
    repository = FakeSemanticMemoryRepository()
    index = FakeSemanticMemoryIndex()
    embedder = FakeEmbeddingModel()
    command = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=embedder,
    )

    fact = await command.execute(
        user_id=uuid.uuid4(), fact_key="k", fact_value="a much longer fact value string"
    )

    assert fact.embedding == embedder.embed("a much longer fact value string")
    assert fact.embedding != embedder.embed("k")


async def test_execute_generates_a_new_uuid_for_the_fact():
    command = RecordSemanticFact(
        semantic_memory_repository=FakeSemanticMemoryRepository(),
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
    )

    fact_one = await command.execute(user_id=uuid.uuid4(), fact_key="k", fact_value="v")
    fact_two = await command.execute(user_id=uuid.uuid4(), fact_key="k", fact_value="v")

    assert isinstance(fact_one.id, uuid.UUID)
    assert fact_one.id != fact_two.id


async def test_execute_defaults_confidence_to_one():
    command = RecordSemanticFact(
        semantic_memory_repository=FakeSemanticMemoryRepository(),
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
    )

    fact = await command.execute(user_id=uuid.uuid4(), fact_key="k", fact_value="v")

    assert fact.confidence == 1.0


async def test_execute_stores_the_given_user_id_fact_key_and_fact_value():
    command = RecordSemanticFact(
        semantic_memory_repository=FakeSemanticMemoryRepository(),
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
    )

    user_id = uuid.uuid4()
    fact = await command.execute(
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
