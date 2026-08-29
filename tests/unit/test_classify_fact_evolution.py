import json
import uuid

import pytest

from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.application.queries.classify_fact_evolution import ClassifyFactEvolution
from tests.unit.mag_fakes import (
    FakeMemoryGraphRepository,
    FakeSemanticMemoryIndex,
    FakeSemanticMemoryRepository,
)
from tests.unit.rag_fakes import FakeChatModel, FakeEmbeddingModel


async def _seed(
    repository: FakeSemanticMemoryRepository,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    fact_key: str,
    fact_value: str,
) -> None:
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=FakeMemoryGraphRepository(),
    )
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key=fact_key, fact_value=fact_value
    )


async def test_execute_raises_when_no_existing_fact_for_the_key():
    classify = ClassifyFactEvolution(
        semantic_memory_repository=FakeSemanticMemoryRepository(),
        chat_model=FakeChatModel(response="{}"),
    )

    with pytest.raises(ValueError, match="no existing fact"):
        await classify.execute(
            tenant_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            fact_key="never_recorded",
            new_information="anything",
        )


async def test_execute_returns_the_llm_classified_operation_and_reasoning():
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    await _seed(repository, tenant_id, user_id, "location", "lives in New York")
    chat_model = FakeChatModel(
        response=json.dumps({"operation": "update", "reasoning": "direct contradiction"})
    )
    classify = ClassifyFactEvolution(
        semantic_memory_repository=repository, chat_model=chat_model
    )

    result = await classify.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="location",
        new_information="moved to Berlin last week",
    )

    assert result.operation == "update"
    assert result.reasoning == "direct contradiction"


@pytest.mark.parametrize("operation", ["update", "invalidate", "refine", "no_conflict"])
async def test_execute_accepts_every_valid_operation(operation):
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    await _seed(repository, tenant_id, user_id, "k", "v")
    chat_model = FakeChatModel(
        response=json.dumps({"operation": operation, "reasoning": "r"})
    )
    classify = ClassifyFactEvolution(
        semantic_memory_repository=repository, chat_model=chat_model
    )

    result = await classify.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="n"
    )

    assert result.operation == operation


async def test_execute_strips_markdown_fencing_before_parsing():
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    await _seed(repository, tenant_id, user_id, "k", "v")
    fenced = "```json\n" + json.dumps({"operation": "refine", "reasoning": "r"}) + "\n```"
    classify = ClassifyFactEvolution(
        semantic_memory_repository=repository, chat_model=FakeChatModel(response=fenced)
    )

    result = await classify.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="n"
    )

    assert result.operation == "refine"


async def test_execute_defaults_to_no_conflict_after_malformed_json_exhausts_retries():
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    await _seed(repository, tenant_id, user_id, "k", "v")
    classify = ClassifyFactEvolution(
        semantic_memory_repository=repository,
        chat_model=FakeChatModel(response="not json at all"),
    )

    result = await classify.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="n"
    )

    assert result.operation == "no_conflict"


async def test_execute_defaults_to_no_conflict_when_operation_is_not_a_recognized_value():
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    await _seed(repository, tenant_id, user_id, "k", "v")
    chat_model = FakeChatModel(
        response=json.dumps({"operation": "delete_everything", "reasoning": "r"})
    )
    classify = ClassifyFactEvolution(
        semantic_memory_repository=repository, chat_model=chat_model
    )

    result = await classify.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="n"
    )

    assert result.operation == "no_conflict"


async def test_execute_defaults_to_no_conflict_when_operation_is_not_a_string():
    # bool is a subclass of int in Python -- {"operation": True} could
    # slip past a naive isinstance(x, str) check via truthiness elsewhere,
    # so this pins that the type guard actually rejects a non-string.
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    await _seed(repository, tenant_id, user_id, "k", "v")
    chat_model = FakeChatModel(response=json.dumps({"operation": True, "reasoning": "r"}))
    classify = ClassifyFactEvolution(
        semantic_memory_repository=repository, chat_model=chat_model
    )

    result = await classify.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="n"
    )

    assert result.operation == "no_conflict"


async def test_execute_defaults_reasoning_to_empty_string_when_omitted():
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    await _seed(repository, tenant_id, user_id, "k", "v")
    chat_model = FakeChatModel(response=json.dumps({"operation": "no_conflict"}))
    classify = ClassifyFactEvolution(
        semantic_memory_repository=repository, chat_model=chat_model
    )

    result = await classify.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="n"
    )

    assert result.operation == "no_conflict"
    assert result.reasoning == ""
