import json
import uuid
from datetime import UTC, datetime

from src.mag.application.queries.retrieve_by_causal_relevance import CausalRetrieval
from src.mag.domain.entities import EpisodicMemory
from tests.unit.mag_fakes import FakeEpisodicMemoryRepository
from tests.unit.rag_fakes import FakeChatModel


class _ScriptedChatModel:
    """Returns a different completion on each successive complete() call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.call_count = 0
        self.last_prompt: str | None = None

    async def generate(self, question: str, context: str) -> str:
        raise NotImplementedError("CausalRetrieval only ever calls complete()")

    async def complete(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return next(self._responses)


def _episode(session_id: uuid.UUID, content: dict) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id,
        content=content,
        embedding=[0.0] * 384,
        timestamp=datetime.now(UTC),
    )


def _scores_response(scores: list[tuple[int, float]]) -> str:
    return json.dumps(
        {"scores": [{"episode_index": i, "score": s} for i, s in scores]}
    )


async def test_execute_returns_empty_list_when_session_has_no_episodes():
    episodes_repo = FakeEpisodicMemoryRepository()
    chat_model = FakeChatModel()
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=uuid.uuid4(), session_id=uuid.uuid4(), query="why did it fail", top_k=5
    )

    assert result == []
    assert chat_model.last_prompt is None  # no LLM call at all


async def test_execute_ranks_episodes_by_score_descending():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    low = await _save(episodes_repo, session_id, tenant_id, {"n": "low"})
    high = await _save(episodes_repo, session_id, tenant_id, {"n": "high"})
    mid = await _save(episodes_repo, session_id, tenant_id, {"n": "mid"})
    response = _scores_response([(1, 0.1), (2, 0.9), (3, 0.5)])
    chat_model = _ScriptedChatModel([response])
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=3
    )

    assert [r.episode.id for r in result] == [high.id, mid.id, low.id]
    assert [r.score for r in result] == [0.9, 0.5, 0.1]


async def test_execute_truncates_to_top_k():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await _save(episodes_repo, session_id, tenant_id, {"n": 1})
    await _save(episodes_repo, session_id, tenant_id, {"n": 2})
    await _save(episodes_repo, session_id, tenant_id, {"n": 3})
    response = _scores_response([(1, 0.1), (2, 0.9), (3, 0.5)])
    chat_model = _ScriptedChatModel([response])
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=2
    )

    assert len(result) == 2


async def test_execute_strips_markdown_fencing_before_parsing():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    episode = await _save(episodes_repo, session_id, tenant_id, {"n": 1})
    fenced = f"```json\n{_scores_response([(1, 0.7)])}\n```"
    chat_model = _ScriptedChatModel([fenced])
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=5
    )

    assert len(result) == 1
    assert result[0].episode.id == episode.id
    assert result[0].score == 0.7


async def test_execute_falls_back_to_flat_zero_scores_after_exhausting_retries():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    first = await _save(episodes_repo, session_id, tenant_id, {"n": 1})
    second = await _save(episodes_repo, session_id, tenant_id, {"n": 2})
    chat_model = _ScriptedChatModel(["not valid json{{{"] * 5)
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=5
    )

    assert chat_model.call_count == 3  # bounded retry, not unlimited
    assert len(result) == 2  # not a crash, not an empty list
    assert [r.episode.id for r in result] == [first.id, second.id]  # original order kept
    assert all(r.score == 0.0 for r in result)


async def test_execute_respects_top_k_after_exhausting_retries():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await _save(episodes_repo, session_id, tenant_id, {"n": 1})
    await _save(episodes_repo, session_id, tenant_id, {"n": 2})
    await _save(episodes_repo, session_id, tenant_id, {"n": 3})
    chat_model = _ScriptedChatModel(["garbage"] * 5)
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=2
    )

    assert len(result) == 2
    assert all(r.score == 0.0 for r in result)


async def test_execute_retries_when_scores_is_missing():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await _save(episodes_repo, session_id, tenant_id, {"n": 1})
    valid = _scores_response([(1, 0.6)])
    malformed = json.dumps({"not_scores": []})
    chat_model = _ScriptedChatModel([malformed, valid])
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=5
    )

    assert chat_model.call_count == 2
    assert len(result) == 1
    assert result[0].score == 0.6


async def test_execute_retries_when_an_entry_is_missing_episode_index():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await _save(episodes_repo, session_id, tenant_id, {"n": 1})
    valid = _scores_response([(1, 0.6)])
    malformed = json.dumps({"scores": [{"score": 0.5}]})
    chat_model = _ScriptedChatModel([malformed, valid])
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=5
    )

    assert chat_model.call_count == 2
    assert len(result) == 1
    assert result[0].score == 0.6


async def test_execute_retries_when_episode_index_is_out_of_range():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await _save(episodes_repo, session_id, tenant_id, {"n": 1})
    valid = _scores_response([(1, 0.6)])
    out_of_range = _scores_response([(99, 0.5)])
    chat_model = _ScriptedChatModel([out_of_range, valid])
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=5
    )

    assert chat_model.call_count == 2
    assert len(result) == 1
    assert result[0].score == 0.6


async def test_execute_retries_when_score_is_not_numeric():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await _save(episodes_repo, session_id, tenant_id, {"n": 1})
    valid = _scores_response([(1, 0.6)])
    bad_score = json.dumps({"scores": [{"episode_index": 1, "score": "high"}]})
    chat_model = _ScriptedChatModel([bad_score, valid])
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=5
    )

    assert chat_model.call_count == 2
    assert len(result) == 1
    assert result[0].score == 0.6


async def test_execute_retries_when_score_is_out_of_range():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await _save(episodes_repo, session_id, tenant_id, {"n": 1})
    valid = _scores_response([(1, 0.6)])
    out_of_range_score = json.dumps({"scores": [{"episode_index": 1, "score": 1.5}]})
    chat_model = _ScriptedChatModel([out_of_range_score, valid])
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=5
    )

    assert chat_model.call_count == 2
    assert len(result) == 1
    assert result[0].score == 0.6


async def test_execute_defaults_an_unmentioned_episode_to_zero():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    mentioned = await _save(episodes_repo, session_id, tenant_id, {"n": "mentioned"})
    unmentioned = await _save(episodes_repo, session_id, tenant_id, {"n": "unmentioned"})
    # Only episode_index 1 is scored -- episode 2 (unmentioned) must still
    # come back in the result, defaulted to 0.0, not dropped.
    response = _scores_response([(1, 0.8)])
    chat_model = _ScriptedChatModel([response])
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=5
    )

    by_id = {r.episode.id: r.score for r in result}
    assert by_id == {mentioned.id: 0.8, unmentioned.id: 0.0}


async def test_execute_last_entry_wins_on_a_repeated_episode_index():
    episodes_repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    episode = await _save(episodes_repo, session_id, tenant_id, {"n": 1})
    response = _scores_response([(1, 0.2), (1, 0.9)])
    chat_model = _ScriptedChatModel([response])
    use_case = CausalRetrieval(episodes_repo, chat_model)

    result = await use_case.execute(
        tenant_id=tenant_id, session_id=session_id, query="why did it fail", top_k=5
    )

    assert len(result) == 1
    assert result[0].episode.id == episode.id
    assert result[0].score == 0.9


async def _save(
    episodes_repo: FakeEpisodicMemoryRepository,
    session_id: uuid.UUID,
    tenant_id: uuid.UUID,
    content: dict,
) -> EpisodicMemory:
    episode = _episode(session_id, content)
    await episodes_repo.save(episode, tenant_id)
    return episode
