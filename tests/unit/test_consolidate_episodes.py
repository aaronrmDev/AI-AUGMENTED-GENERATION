import json
import uuid
from datetime import UTC, datetime

from src.mag.application.commands.consolidate_episodes import ConsolidateEpisodes
from src.mag.domain.entities import EpisodicMemory
from tests.unit.mag_fakes import (
    FakeEpisodicMemoryRepository,
    FakeMemoryGraphRepository,
    FakeSemanticMemoryRepository,
)
from tests.unit.mag_fakes import (
    FakeSemanticMemoryIndex as _FakeSemanticMemoryIndex,
)
from tests.unit.rag_fakes import FakeEmbeddingModel


class _ScriptedChatModel:
    """Returns a different completion on each successive complete() call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.call_count = 0
        self.last_prompt: str | None = None

    async def generate(self, question: str, context: str) -> str:
        raise NotImplementedError("ConsolidateEpisodes only ever calls complete()")

    async def complete(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return next(self._responses)


def _episode(session_id: uuid.UUID, content: dict, consolidated: bool = False) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id,
        content=content,
        embedding=[0.0] * 384,
        timestamp=datetime.now(UTC),
        consolidated_at=datetime.now(UTC) if consolidated else None,
    )


_VALID_RESPONSE = json.dumps(
    {
        "facts": [
            {"fact_key": "primary_language", "fact_value": "Python", "confidence": 0.9},
            {"fact_key": "secondary_language", "fact_value": "Go", "confidence": 0.6},
        ]
    }
)


def _use_case(
    episodes_repo: FakeEpisodicMemoryRepository,
    facts_repo: FakeSemanticMemoryRepository,
    index: _FakeSemanticMemoryIndex,
    chat_model: _ScriptedChatModel,
    graph: FakeMemoryGraphRepository | None = None,
) -> ConsolidateEpisodes:
    return ConsolidateEpisodes(
        episodic_memory_repository=episodes_repo,
        semantic_memory_repository=facts_repo,
        semantic_memory_index=index,
        embedding_model=FakeEmbeddingModel(),
        chat_model=chat_model,
        memory_graph_repository=graph or FakeMemoryGraphRepository(),
    )


async def test_execute_returns_empty_list_when_there_are_no_unconsolidated_episodes():
    episodes_repo = FakeEpisodicMemoryRepository()
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(),
        _ScriptedChatModel([_VALID_RESPONSE]),
    )

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), session_id=uuid.uuid4()
    )

    assert result == []


async def test_execute_writes_each_extracted_fact_via_record_semantic_fact():
    episodes_repo = FakeEpisodicMemoryRepository()
    facts_repo = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(
        _episode(session_id, {"input": "what language do you use", "output": "Python"}),
        tenant_id,
    )
    chat_model = _ScriptedChatModel([_VALID_RESPONSE])
    use_case = _use_case(episodes_repo, facts_repo, _FakeSemanticMemoryIndex(), chat_model)

    result = await use_case.execute(tenant_id=tenant_id, user_id=user_id, session_id=session_id)

    assert {f.fact_key for f in result} == {"primary_language", "secondary_language"}
    assert {f.fact_value for f in result} == {"Python", "Go"}
    assert len(facts_repo.saved) == 2
    assert all(fact.source == "consolidation" for fact, _ in facts_repo.saved)


async def test_execute_marks_the_consolidated_episodes_even_when_facts_are_extracted():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    episode = _episode(session_id, {"input": "hi"})
    await episodes_repo.save(episode, tenant_id)
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(),
        _ScriptedChatModel([_VALID_RESPONSE]),
    )

    await use_case.execute(tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id)

    remaining = await episodes_repo.get_unconsolidated_by_session(session_id, tenant_id, limit=10)
    assert remaining == []


async def test_execute_marks_episodes_consolidated_even_when_reflection_finds_nothing():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"input": "hi"}), tenant_id)
    empty_response = json.dumps({"facts": []})
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(),
        _ScriptedChatModel([empty_response]),
    )

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    assert result == []
    remaining = await episodes_repo.get_unconsolidated_by_session(session_id, tenant_id, limit=10)
    assert remaining == []


async def test_execute_only_reflects_on_unconsolidated_episodes():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"n": 1}, consolidated=True), tenant_id)
    empty_response = json.dumps({"facts": []})
    chat_model = _ScriptedChatModel([empty_response])
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(), chat_model
    )

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    assert result == []
    assert chat_model.call_count == 0  # nothing to reflect on -- no LLM call at all


async def test_execute_retries_on_malformed_json_and_succeeds():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"n": 1}), tenant_id)
    chat_model = _ScriptedChatModel(["not valid json{{{", _VALID_RESPONSE])
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(), chat_model
    )

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    assert chat_model.call_count == 2
    assert len(result) == 2


async def test_execute_strips_markdown_fencing_before_parsing():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"n": 1}), tenant_id)
    fenced = f"```json\n{_VALID_RESPONSE}\n```"
    chat_model = _ScriptedChatModel([fenced])
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(), chat_model
    )

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    assert len(result) == 2


async def test_execute_retries_when_a_fact_element_is_not_an_object():
    # Regression test: the outer {"facts": [...]} envelope being valid JSON
    # says nothing about what's inside it -- a bare string element used to
    # reach fact["fact_key"] uncaught, outside this method's retry loop.
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"n": 1}), tenant_id)
    malformed = json.dumps({"facts": ["User's language is Python"]})
    chat_model = _ScriptedChatModel([malformed, _VALID_RESPONSE])
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(), chat_model
    )

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    assert chat_model.call_count == 2
    assert len(result) == 2


async def test_execute_retries_when_a_fact_is_missing_a_required_field():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"n": 1}), tenant_id)
    renamed_fields = json.dumps({"facts": [{"key": "lang", "value": "Python"}]})
    chat_model = _ScriptedChatModel([renamed_fields, _VALID_RESPONSE])
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(), chat_model
    )

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    assert chat_model.call_count == 2
    assert len(result) == 2


async def test_execute_retries_when_confidence_is_not_a_number():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"n": 1}), tenant_id)
    bad_confidence = json.dumps(
        {"facts": [{"fact_key": "lang", "fact_value": "Python", "confidence": "high"}]}
    )
    chat_model = _ScriptedChatModel([bad_confidence, _VALID_RESPONSE])
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(), chat_model
    )

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    assert chat_model.call_count == 2
    assert len(result) == 2


async def test_execute_retries_when_a_fact_key_is_empty():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"n": 1}), tenant_id)
    empty_key = json.dumps({"facts": [{"fact_key": "", "fact_value": "Python"}]})
    chat_model = _ScriptedChatModel([empty_key, _VALID_RESPONSE])
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(), chat_model
    )

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    assert chat_model.call_count == 2
    assert len(result) == 2


async def test_execute_deduplicates_a_repeated_fact_key_keeping_the_last_value():
    episodes_repo = FakeEpisodicMemoryRepository()
    facts_repo = FakeSemanticMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"n": 1}), tenant_id)
    duplicate_key_response = json.dumps(
        {
            "facts": [
                {"fact_key": "primary_language", "fact_value": "Python", "confidence": 0.5},
                {"fact_key": "primary_language", "fact_value": "Go", "confidence": 0.9},
            ]
        }
    )
    chat_model = _ScriptedChatModel([duplicate_key_response])
    use_case = _use_case(episodes_repo, facts_repo, _FakeSemanticMemoryIndex(), chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    # Exactly one fact returned, not two -- and it's the LAST value seen,
    # matching what RecordSemanticFact's deterministic id would actually
    # persist (the second write overwrites the first's row).
    assert len(result) == 1
    assert result[0].fact_value == "Go"
    assert len(facts_repo.saved) == 1


async def test_execute_returns_empty_facts_after_exhausting_retries_but_still_marks_consolidated():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"n": 1}), tenant_id)
    chat_model = _ScriptedChatModel(["garbage"] * 5)
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(), chat_model
    )

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    assert result == []
    assert chat_model.call_count == 3  # bounded retry, not unlimited
    remaining = await episodes_repo.get_unconsolidated_by_session(session_id, tenant_id, limit=10)
    assert remaining == []


async def test_execute_respects_batch_size():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    for i in range(5):
        await episodes_repo.save(_episode(session_id, {"n": i}), tenant_id)
    empty_response = json.dumps({"facts": []})
    chat_model = _ScriptedChatModel([empty_response])
    use_case = _use_case(
        episodes_repo, FakeSemanticMemoryRepository(), _FakeSemanticMemoryIndex(), chat_model
    )

    await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id, batch_size=2
    )

    remaining = await episodes_repo.get_unconsolidated_by_session(session_id, tenant_id, limit=10)
    assert len(remaining) == 3


async def test_execute_links_abstracts_to_from_every_reflected_episode_to_each_written_fact():
    # The reflection is one LLM call over the whole batch, not a 1:1
    # episode:fact mapping -- every episode in the batch links to every
    # fact the LLM extracted from reflecting on all of them together.
    episodes_repo = FakeEpisodicMemoryRepository()
    facts_repo = FakeSemanticMemoryRepository()
    graph = FakeMemoryGraphRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    episode_a = _episode(session_id, {"input": "what language"})
    episode_b = _episode(session_id, {"input": "what framework"})
    await episodes_repo.save(episode_a, tenant_id)
    await episodes_repo.save(episode_b, tenant_id)
    chat_model = _ScriptedChatModel([_VALID_RESPONSE])
    use_case = _use_case(
        episodes_repo, facts_repo, _FakeSemanticMemoryIndex(), chat_model, graph=graph
    )

    result = await use_case.execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id
    )

    assert len(result) == 2  # two facts extracted (see _VALID_RESPONSE)
    # 2 episodes x 2 facts = 4 abstracts_to edges.
    assert len(graph.abstracts_to_links) == 4
    linked_episode_ids = {episode_id for episode_id, _, _ in graph.abstracts_to_links}
    linked_fact_ids = {fact_id for _, fact_id, _ in graph.abstracts_to_links}
    assert linked_episode_ids == {episode_a.id, episode_b.id}
    assert linked_fact_ids == {fact.id for fact in result}


async def test_execute_links_no_abstracts_to_edges_when_reflection_finds_nothing():
    episodes_repo = FakeEpisodicMemoryRepository()
    graph = FakeMemoryGraphRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await episodes_repo.save(_episode(session_id, {"input": "hi"}), tenant_id)
    empty_response = json.dumps({"facts": []})
    use_case = _use_case(
        episodes_repo,
        FakeSemanticMemoryRepository(),
        _FakeSemanticMemoryIndex(),
        _ScriptedChatModel([empty_response]),
        graph=graph,
    )

    await use_case.execute(tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id)

    assert graph.abstracts_to_links == []
