import json

from evaluation.infrastructure.claude_judge import ClaudeJudge


class _FakeMessage:
    def __init__(self, payload: dict) -> None:
        thinking_block = type("ThinkingBlock", (), {"type": "thinking", "thinking": "reasoning"})()
        text_block = type("TextBlock", (), {"type": "text", "text": json.dumps(payload)})()
        self.content = [thinking_block, text_block]


class _FakeMessages:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.last_call_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeMessage(self._payload)


class _FakeAnthropicClient:
    def __init__(self, payload: dict) -> None:
        self.messages = _FakeMessages(payload)


_VALID_PAYLOAD = {
    "response_a": {
        "coherence": 3,
        "relevance": 4,
        "completeness": 2,
        "groundedness": 5,
        "unverifiable_claims": ["claim x"],
    },
    "response_b": {
        "coherence": 5,
        "relevance": 5,
        "completeness": 5,
        "groundedness": 5,
        "unverifiable_claims": [],
    },
}


async def test_score_parses_both_responses_from_the_judge_json():
    fake_client = _FakeAnthropicClient(_VALID_PAYLOAD)
    judge = ClaudeJudge(client=fake_client, model_id="claude-opus-5")

    scores_a, scores_b = await judge.score(
        query="q", response_a="a", response_b="b", context_a="ca", context_b="cb"
    )

    assert scores_a.coherence == 3
    assert scores_a.unverifiable_claims == ["claim x"]
    assert scores_b.coherence == 5
    assert scores_b.unverifiable_claims == []


async def test_score_includes_the_query_and_both_responses_in_the_request():
    fake_client = _FakeAnthropicClient(_VALID_PAYLOAD)
    judge = ClaudeJudge(client=fake_client, model_id="claude-opus-5")

    await judge.score(
        query="What is FastAPI?",
        response_a="Response one.",
        response_b="Response two.",
        context_a="FastAPI is a Python web framework.",
        context_b="Uvicorn is an ASGI server.",
    )

    sent = fake_client.messages.last_call_kwargs
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "Response one." in full_prompt
    assert "Response two." in full_prompt
    assert "FastAPI is a Python web framework." in full_prompt
    assert "Uvicorn is an ASGI server." in full_prompt


async def test_score_judges_each_response_against_its_own_context_only():
    fake_client = _FakeAnthropicClient(_VALID_PAYLOAD)
    judge = ClaudeJudge(client=fake_client, model_id="claude-opus-5")

    await judge.score(
        query="q",
        response_a="a",
        response_b="b",
        context_a="ONLY-IN-A-CONTEXT",
        context_b="ONLY-IN-B-CONTEXT",
    )

    full_prompt = str(fake_client.messages.last_call_kwargs["messages"])
    a_index = full_prompt.index("ONLY-IN-A-CONTEXT")
    a_response_index = full_prompt.index("Response A:")
    b_index = full_prompt.index("ONLY-IN-B-CONTEXT")
    b_response_index = full_prompt.index("Response B:")
    assert a_index < a_response_index < b_index < b_response_index


async def test_score_marks_each_empty_context_explicitly_rather_than_omitting_it():
    fake_client = _FakeAnthropicClient(_VALID_PAYLOAD)
    judge = ClaudeJudge(client=fake_client, model_id="claude-opus-5")

    await judge.score(query="q", response_a="a", response_b="b", context_a="", context_b="")

    full_prompt = str(fake_client.messages.last_call_kwargs["messages"])
    assert full_prompt.count("(none provided)") == 2


async def test_score_includes_the_reference_passage_when_given():
    fake_client = _FakeAnthropicClient(_VALID_PAYLOAD)
    judge = ClaudeJudge(client=fake_client, model_id="claude-opus-5")

    await judge.score(
        query="q",
        response_a="a",
        response_b="b",
        context_a="a's own retrieved chunk",
        context_b="b's own retrieved chunk",
        reference_context="THE-ACTUAL-CORRECT-ANSWER-PASSAGE",
    )

    full_prompt = str(fake_client.messages.last_call_kwargs["messages"])
    assert "THE-ACTUAL-CORRECT-ANSWER-PASSAGE" in full_prompt
    assert "Reference passage" in full_prompt


async def test_score_omits_the_reference_passage_block_when_not_given():
    fake_client = _FakeAnthropicClient(_VALID_PAYLOAD)
    judge = ClaudeJudge(client=fake_client, model_id="claude-opus-5")

    await judge.score(query="q", response_a="a", response_b="b", context_a="ca", context_b="cb")

    full_prompt = str(fake_client.messages.last_call_kwargs["messages"])
    assert "Reference passage" not in full_prompt
