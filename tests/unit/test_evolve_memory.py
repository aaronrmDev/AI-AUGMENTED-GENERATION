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
    repository: FakeSemanticMemoryRepository,
    classify_chat_model: FakeChatModel,
    refine_chat_model: FakeChatModel | None = None,
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
            # A SEPARATE chat model from classification's, not the same
            # instance reused -- sharing one fake's fixed response across
            # both LLM call sites made it structurally impossible for a
            # test to tell "the classify call got its own prompt" apart
            # from "the refine call got its own prompt" (a real gap
            # confirmed by review). Defaults to classify_chat_model only
            # for tests where refine is never actually reached (its
            # content then genuinely doesn't matter).
            chat_model=refine_chat_model or classify_chat_model,
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
    # Two DISTINCT chat models, one per LLM call site -- not one fake
    # reused for both. If EvolveMemory/RefineMemory ever routed the wrong
    # prompt to the wrong model (a swapped-argument or wrong-system-prompt
    # bug), classify_chat_model's response (which has no
    # merged_fact_value key) would fail RefineMemory._merge's JSON
    # parsing, or refine_chat_model's response (which has no operation
    # key) would fail ClassifyFactEvolution's parsing -- either mistake
    # is now distinguishable, unlike sharing one blob that satisfies both
    # shapes at once.
    classify_chat_model = FakeChatModel(
        response=json.dumps({"operation": "refine", "reasoning": "adds nuance"})
    )
    refine_chat_model = FakeChatModel(
        response=json.dumps(
            {"merged_fact_value": "prefers Python, especially for data analysis"}
        )
    )
    evolve, record = _wire(repository, classify_chat_model, refine_chat_model)
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
    # Each call site actually received its own distinct prompt, not the
    # other's -- proves the dispatch routed correctly rather than merely
    # happening to parse.
    assert "comparing an existing fact" in classify_chat_model.last_prompt.lower()
    assert "merging an existing fact" in refine_chat_model.last_prompt.lower()


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
