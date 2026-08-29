import json
import uuid

from src.mag.application.commands.evolve_memory import EvolveMemory
from src.mag.application.commands.invalidate_memory import InvalidateMemory
from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.application.commands.refine_memory import RefineMemory
from src.mag.application.commands.update_memory import UpdateMemory
from src.mag.application.queries.classify_fact_evolution import ClassifyFactEvolution
from tests.unit.mag_fakes import (
    FakeMemoryGraphRepository,
    FakeSemanticMemoryIndex,
    FakeSemanticMemoryRepository,
)
from tests.unit.rag_fakes import FakeChatModel, FakeEmbeddingModel


def _wire(
    repository: FakeSemanticMemoryRepository, classify_chat_model: FakeChatModel
) -> tuple[EvolveMemory, RecordSemanticFact]:
    index = FakeSemanticMemoryIndex()
    graph = FakeMemoryGraphRepository()
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=index,
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=graph,
    )
    evolve = EvolveMemory(
        classify_fact_evolution=ClassifyFactEvolution(
            semantic_memory_repository=repository, chat_model=classify_chat_model
        ),
        update_memory=UpdateMemory(
            semantic_memory_repository=repository, record_semantic_fact=record
        ),
        invalidate_memory=InvalidateMemory(
            semantic_memory_repository=repository,
            semantic_memory_index=index,
            memory_graph_repository=graph,
        ),
        refine_memory=RefineMemory(
            semantic_memory_repository=repository,
            record_semantic_fact=record,
            # RefineMemory's own LLM call shares this fake's fixed response
            # in every test below that doesn't classify as "refine" -- it's
            # never reached in those cases, so its content doesn't matter.
            chat_model=classify_chat_model,
        ),
    )
    return evolve, record


async def test_execute_dispatches_to_update_on_update_classification():
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    chat_model = FakeChatModel(
        response=json.dumps({"operation": "update", "reasoning": "contradiction"})
    )
    evolve, record = _wire(repository, chat_model)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="location", fact_value="New York"
    )

    classification, result = await evolve.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="location",
        new_information="moved to Berlin last week",
    )

    assert classification.operation == "update"
    assert result is not None
    assert result.fact_value == "moved to Berlin last week"


async def test_execute_dispatches_to_invalidate_on_invalidate_classification():
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    chat_model = FakeChatModel(
        response=json.dumps({"operation": "invalidate", "reasoning": "no longer true"})
    )
    evolve, record = _wire(repository, chat_model)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="pet", fact_value="has a pet named Rex"
    )

    classification, result = await evolve.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="pet", new_information="Rex passed away"
    )

    assert classification.operation == "invalidate"
    assert result is not None
    assert result.valid_until is not None
    # Invalidate never replaces the value -- #63's own "without
    # necessarily replacing it with anything."
    assert result.fact_value == "has a pet named Rex"


async def test_execute_dispatches_to_refine_on_refine_classification():
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    # ClassifyFactEvolution and RefineMemory's _merge share this fake's one
    # fixed response -- valid JSON for both call shapes at once, so the
    # dispatch (classify -> refine's own LLM merge call) both parse cleanly.
    chat_model = FakeChatModel(
        response=json.dumps(
            {
                "operation": "refine",
                "reasoning": "adds nuance",
                "merged_fact_value": "prefers Python, especially for data analysis",
            }
        )
    )
    evolve, record = _wire(repository, chat_model)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="language", fact_value="prefers Python"
    )

    classification, result = await evolve.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="language",
        new_information="especially for data analysis",
    )

    assert classification.operation == "refine"
    assert result is not None
    assert result.fact_value == "prefers Python, especially for data analysis"


async def test_execute_does_nothing_on_no_conflict_classification():
    repository = FakeSemanticMemoryRepository()
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    chat_model = FakeChatModel(
        response=json.dumps({"operation": "no_conflict", "reasoning": "unrelated"})
    )
    evolve, record = _wire(repository, chat_model)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="language", fact_value="prefers Python"
    )

    classification, result = await evolve.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="language",
        new_information="the weather is nice today",
    )

    assert classification.operation == "no_conflict"
    assert result is None
    found = await repository.find_by_key(user_id, "language", tenant_id)
    assert found is not None
    assert found.fact_value == "prefers Python"
