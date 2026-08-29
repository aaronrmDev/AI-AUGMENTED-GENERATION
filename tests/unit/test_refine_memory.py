import json
import uuid
from datetime import UTC, datetime

import pytest

from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.application.commands.refine_memory import RefineMemory
from tests.unit.mag_fakes import (
    FakeMemoryGraphRepository,
    FakeSemanticMemoryIndex,
    FakeSemanticMemoryRepository,
)
from tests.unit.rag_fakes import FakeChatModel, FakeEmbeddingModel


def _wire(
    repository: FakeSemanticMemoryRepository, chat_model: FakeChatModel
) -> tuple[RefineMemory, RecordSemanticFact]:
    record = RecordSemanticFact(
        semantic_memory_repository=repository,
        semantic_memory_index=FakeSemanticMemoryIndex(),
        embedding_model=FakeEmbeddingModel(),
        memory_graph_repository=FakeMemoryGraphRepository(),
    )
    refine = RefineMemory(
        semantic_memory_repository=repository,
        record_semantic_fact=record,
        chat_model=chat_model,
    )
    return refine, record


async def test_execute_raises_when_no_existing_fact_for_the_key():
    refine, _ = _wire(FakeSemanticMemoryRepository(), FakeChatModel(response="{}"))

    with pytest.raises(ValueError, match="no existing fact"):
        await refine.execute(
            tenant_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            fact_key="never_recorded",
            new_information="anything",
        )


async def test_execute_stores_the_llm_merged_value():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    chat_model = FakeChatModel(
        response=json.dumps(
            {"merged_fact_value": "prefers Python, especially for data analysis"}
        )
    )
    refine, record = _wire(repository, chat_model)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="language", fact_value="prefers Python"
    )

    refined = await refine.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="language",
        new_information="especially for data analysis",
    )

    assert refined.fact_value == "prefers Python, especially for data analysis"


async def test_execute_sends_the_refine_prompt_with_existing_value_and_new_information_unswapped():
    # A swapped-argument bug (existing_fact_value/new_information reversed
    # in the prompt builder call) or a wrong-system-prompt bug (using
    # FACT_EVOLUTION_SYSTEM_PROMPT instead of REFINE_SYSTEM_PROMPT) would
    # still parse successfully against FakeChatModel's fixed canned
    # response -- confirmed as a real gap by review. Asserting on the
    # actual prompt content sent is the only way to catch either mistake
    # at this level. Deliberately NOT "prefers Python"/"data analysis" --
    # REFINE_SYSTEM_PROMPT's own worked example contains that exact text,
    # so .find() would locate it in the constant system-prompt portion
    # regardless of argument order, making the ordering check vacuous (a
    # real bug in an earlier version of this test, confirmed by review
    # with an empirical reproduction: swapping the real arguments still
    # passed). These fixture values appear nowhere in either system
    # prompt's own text.
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    chat_model = FakeChatModel(response=json.dumps({"merged_fact_value": "merged"}))
    refine, record = _wire(repository, chat_model)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="hobby", fact_value="enjoys hiking"
    )

    await refine.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="hobby",
        new_information="especially in national parks during autumn",
    )

    prompt = chat_model.last_prompt
    assert "merging an existing fact" in prompt.lower()
    assert "comparing an existing fact" not in prompt.lower()
    existing_index = prompt.find("enjoys hiking")
    new_index = prompt.find("especially in national parks during autumn")
    assert existing_index != -1
    assert new_index != -1
    assert existing_index < new_index


async def test_execute_preserves_a_previously_archived_status():
    # RefineMemory had ZERO coverage of status preservation before this
    # (review flagged: only 1 of the 4 UpdateMemory/RefineMemory x
    # valid_until/archived_at combinations had any test at all).
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    chat_model = FakeChatModel(response=json.dumps({"merged_fact_value": "merged"}))
    refine, record = _wire(repository, chat_model)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="rarely needed but true"
    )
    archived_at = datetime(2026, 1, 1, tzinfo=UTC)
    await repository.archive(user_id, "k", tenant_id, archived_at)

    refined = await refine.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="more nuance"
    )

    assert refined.archived_at == archived_at
    assert refined.fact_value == "merged"


async def test_execute_preserves_a_previously_set_valid_until():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    chat_model = FakeChatModel(response=json.dumps({"merged_fact_value": "merged"}))
    refine, record = _wire(repository, chat_model)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="has a pet named Rex"
    )
    invalidated_at = datetime(2026, 1, 1, tzinfo=UTC)
    await repository.invalidate(user_id, "k", tenant_id, invalidated_at)

    refined = await refine.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="more nuance"
    )

    assert refined.valid_until == invalidated_at
    assert refined.fact_value == "merged"


async def test_execute_carries_over_the_existing_confidence_and_source():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    chat_model = FakeChatModel(response=json.dumps({"merged_fact_value": "merged"}))
    refine, record = _wire(repository, chat_model)
    await record.execute(
        tenant_id=tenant_id,
        user_id=user_id,
        fact_key="language",
        fact_value="prefers Python",
        confidence=0.6,
        source="conversation",
    )

    refined = await refine.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="language", new_information="more nuance"
    )

    # Refine, unlike Update, preserves the fact's existing confidence and
    # source -- it isn't a correction, it's the same fact with more detail.
    assert refined.confidence == 0.6
    assert refined.source == "conversation"


async def test_execute_writes_the_pre_merge_value_to_history():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    chat_model = FakeChatModel(response=json.dumps({"merged_fact_value": "merged value"}))
    refine, record = _wire(repository, chat_model)
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="language", fact_value="prefers Python"
    )

    await refine.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="language", new_information="more nuance"
    )

    history = await repository.find_history(user_id, "language", tenant_id)
    assert len(history) == 1
    assert history[0].fact_value == "prefers Python"
    assert history[0].operation == "refine"


async def test_execute_strips_markdown_fencing_before_parsing():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    fenced = "```json\n" + json.dumps({"merged_fact_value": "merged value"}) + "\n```"
    refine, record = _wire(repository, FakeChatModel(response=fenced))
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="original"
    )

    refined = await refine.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="new"
    )

    assert refined.fact_value == "merged value"


async def test_execute_falls_back_to_concatenation_after_exhausting_retries():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    refine, record = _wire(repository, FakeChatModel(response="not json at all"))
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="original fact"
    )

    refined = await refine.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="new nuance"
    )

    # A malformed response never produces a fact-free outcome -- the new
    # information still lands somewhere, not silently dropped.
    assert refined.fact_value == "original fact; new nuance"


async def test_execute_falls_back_when_merged_fact_value_is_not_a_string():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    refine, record = _wire(
        repository, FakeChatModel(response=json.dumps({"merged_fact_value": 42}))
    )
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="original fact"
    )

    refined = await refine.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="new nuance"
    )

    assert refined.fact_value == "original fact; new nuance"


async def test_execute_falls_back_when_merged_fact_value_is_empty():
    repository = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    refine, record = _wire(
        repository, FakeChatModel(response=json.dumps({"merged_fact_value": "   "}))
    )
    await record.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", fact_value="original fact"
    )

    refined = await refine.execute(
        tenant_id=tenant_id, user_id=user_id, fact_key="k", new_information="new nuance"
    )

    assert refined.fact_value == "original fact; new nuance"
