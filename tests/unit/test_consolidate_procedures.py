import json
import uuid
from datetime import UTC, datetime

from src.mag.application.commands.consolidate_procedures import ConsolidateProcedures
from src.mag.domain.entities import EpisodicMemory
from tests.unit.mag_fakes import FakeProceduralMemoryRepository


class _ScriptedChatModel:
    """Returns a different completion on each successive complete() call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.call_count = 0
        self.last_prompt: str | None = None

    async def generate(self, question: str, context: str) -> str:
        raise NotImplementedError("ConsolidateProcedures only ever calls complete()")

    async def complete(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return next(self._responses)


def _episode(content: dict) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content=content,
        embedding=[0.0] * 384,
        timestamp=datetime.now(UTC),
    )


_VALID_RESPONSE = json.dumps(
    {
        "procedures": [
            {
                "task_pattern": "deploy_fastapi_service",
                "workflow": {"steps": ["Docker", "Gunicorn", "Nginx"]},
                "success_rate": 1.0,
            }
        ]
    }
)


def _use_case(
    procedures_repo: FakeProceduralMemoryRepository, chat_model: _ScriptedChatModel
) -> ConsolidateProcedures:
    return ConsolidateProcedures(
        procedural_memory_repository=procedures_repo, chat_model=chat_model
    )


async def test_execute_returns_empty_list_for_empty_episodes():
    use_case = _use_case(FakeProceduralMemoryRepository(), _ScriptedChatModel([_VALID_RESPONSE]))

    result = await use_case.execute(tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=[])

    assert result == []


async def test_execute_writes_each_extracted_procedure_via_record_procedure():
    procedures_repo = FakeProceduralMemoryRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    episodes = [
        _episode({"input": "deploy my FastAPI app", "outcome": "success"}),
        _episode({"input": "deploy the other FastAPI app", "outcome": "success"}),
    ]
    chat_model = _ScriptedChatModel([_VALID_RESPONSE])
    use_case = _use_case(procedures_repo, chat_model)

    result = await use_case.execute(tenant_id=tenant_id, user_id=user_id, episodes=episodes)

    assert {p.task_pattern for p in result} == {"deploy_fastapi_service"}
    assert result[0].workflow == {"steps": ["Docker", "Gunicorn", "Nginx"]}
    assert result[0].success_rate == 1.0
    assert len(procedures_repo.saved) == 1


async def test_execute_returns_empty_list_when_reflection_finds_no_repeated_pattern():
    procedures_repo = FakeProceduralMemoryRepository()
    empty_response = json.dumps({"procedures": []})
    use_case = _use_case(procedures_repo, _ScriptedChatModel([empty_response]))
    episodes = [_episode({"input": "a one-off task", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert result == []
    assert procedures_repo.saved == []


async def test_execute_retries_on_malformed_json_and_succeeds():
    procedures_repo = FakeProceduralMemoryRepository()
    chat_model = _ScriptedChatModel(["not valid json{{{", _VALID_RESPONSE])
    use_case = _use_case(procedures_repo, chat_model)
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert chat_model.call_count == 2
    assert len(result) == 1


async def test_execute_strips_markdown_fencing_before_parsing():
    fenced = f"```json\n{_VALID_RESPONSE}\n```"
    use_case = _use_case(FakeProceduralMemoryRepository(), _ScriptedChatModel([fenced]))
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert len(result) == 1


async def test_execute_retries_when_a_procedure_element_is_not_an_object():
    responses = [
        json.dumps({"procedures": ["not an object"]}),
        _VALID_RESPONSE,
    ]
    chat_model = _ScriptedChatModel(responses)
    use_case = _use_case(FakeProceduralMemoryRepository(), chat_model)
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert chat_model.call_count == 2
    assert len(result) == 1


async def test_execute_retries_when_task_pattern_is_empty():
    responses = [
        json.dumps(
            {"procedures": [{"task_pattern": "  ", "workflow": {}, "success_rate": 1.0}]}
        ),
        _VALID_RESPONSE,
    ]
    chat_model = _ScriptedChatModel(responses)
    use_case = _use_case(FakeProceduralMemoryRepository(), chat_model)
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert chat_model.call_count == 2
    assert len(result) == 1


async def test_execute_retries_when_workflow_is_not_an_object():
    responses = [
        json.dumps(
            {
                "procedures": [
                    {"task_pattern": "deploy", "workflow": "not a dict", "success_rate": 1.0}
                ]
            }
        ),
        _VALID_RESPONSE,
    ]
    chat_model = _ScriptedChatModel(responses)
    use_case = _use_case(FakeProceduralMemoryRepository(), chat_model)
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert chat_model.call_count == 2
    assert len(result) == 1


async def test_execute_retries_when_success_rate_is_not_a_number():
    responses = [
        json.dumps(
            {
                "procedures": [
                    {"task_pattern": "deploy", "workflow": {}, "success_rate": "high"}
                ]
            }
        ),
        _VALID_RESPONSE,
    ]
    chat_model = _ScriptedChatModel(responses)
    use_case = _use_case(FakeProceduralMemoryRepository(), chat_model)
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert chat_model.call_count == 2
    assert len(result) == 1


async def test_execute_retries_when_success_rate_is_a_bool():
    # bool is a subclass of int in Python -- {"success_rate": true} could
    # slip past a naive isinstance(x, (int, float)) check.
    responses = [
        json.dumps(
            {"procedures": [{"task_pattern": "deploy", "workflow": {}, "success_rate": True}]}
        ),
        _VALID_RESPONSE,
    ]
    chat_model = _ScriptedChatModel(responses)
    use_case = _use_case(FakeProceduralMemoryRepository(), chat_model)
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert chat_model.call_count == 2
    assert len(result) == 1


async def test_execute_retries_when_success_rate_is_out_of_range():
    responses = [
        json.dumps(
            {"procedures": [{"task_pattern": "deploy", "workflow": {}, "success_rate": 1.5}]}
        ),
        _VALID_RESPONSE,
    ]
    chat_model = _ScriptedChatModel(responses)
    use_case = _use_case(FakeProceduralMemoryRepository(), chat_model)
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert chat_model.call_count == 2
    assert len(result) == 1


async def test_execute_defaults_success_rate_to_one_when_omitted():
    procedures_repo = FakeProceduralMemoryRepository()
    response = json.dumps(
        {"procedures": [{"task_pattern": "deploy", "workflow": {"steps": ["a"]}}]}
    )
    use_case = _use_case(procedures_repo, _ScriptedChatModel([response]))
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert result[0].success_rate == 1.0


async def test_execute_deduplicates_a_repeated_task_pattern_keeping_the_last_value():
    procedures_repo = FakeProceduralMemoryRepository()
    response = json.dumps(
        {
            "procedures": [
                {"task_pattern": "deploy", "workflow": {"steps": ["old"]}, "success_rate": 0.5},
                {"task_pattern": "deploy", "workflow": {"steps": ["new"]}, "success_rate": 0.9},
            ]
        }
    )
    use_case = _use_case(procedures_repo, _ScriptedChatModel([response]))
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert len(result) == 1
    assert result[0].workflow == {"steps": ["new"]}
    assert result[0].success_rate == 0.9


async def test_execute_returns_empty_list_after_exhausting_retries():
    procedures_repo = FakeProceduralMemoryRepository()
    chat_model = _ScriptedChatModel(["garbage", "still garbage", "more garbage"])
    use_case = _use_case(procedures_repo, chat_model)
    episodes = [_episode({"input": "deploy", "outcome": "success"})]

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), episodes=episodes
    )

    assert result == []
    assert chat_model.call_count == 3
    assert procedures_repo.saved == []
