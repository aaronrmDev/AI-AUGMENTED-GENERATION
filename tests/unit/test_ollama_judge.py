import json

from evaluation.infrastructure.ollama_judge import OllamaJudge


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = text


class _FakeChatResponse:
    def __init__(self, text: str) -> None:
        self.message = _FakeMessage(text)


class _FakeOllamaClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.last_call_kwargs: dict | None = None
        self.call_count = 0

    async def chat(self, **kwargs):
        self.call_count += 1
        self.last_call_kwargs = kwargs
        return _FakeChatResponse(json.dumps(self._payload))


class _ScriptedOllamaClient:
    """Returns a different raw response text on each successive call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.call_count = 0
        self.last_call_kwargs: dict | None = None

    async def chat(self, **kwargs):
        self.call_count += 1
        self.last_call_kwargs = kwargs
        return _FakeChatResponse(next(self._responses))


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
    fake_client = _FakeOllamaClient(_VALID_PAYLOAD)
    judge = OllamaJudge(client=fake_client, model_id="qwen3.5")

    scores_a, scores_b = await judge.score(
        query="q", response_a="a", response_b="b", context_a="ca", context_b="cb"
    )

    assert scores_a.coherence == 3
    assert scores_a.unverifiable_claims == ["claim x"]
    assert scores_b.coherence == 5
    assert scores_b.unverifiable_claims == []
    assert scores_a.parse_failed is False
    assert scores_b.parse_failed is False


async def test_score_includes_the_query_and_both_responses_in_the_request():
    fake_client = _FakeOllamaClient(_VALID_PAYLOAD)
    judge = OllamaJudge(client=fake_client, model_id="qwen3.5")

    await judge.score(
        query="What is FastAPI?",
        response_a="Response one.",
        response_b="Response two.",
        context_a="FastAPI is a Python web framework.",
        context_b="Uvicorn is an ASGI server.",
    )

    sent = fake_client.last_call_kwargs
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "Response one." in full_prompt
    assert "Response two." in full_prompt
    assert "FastAPI is a Python web framework." in full_prompt
    assert "Uvicorn is an ASGI server." in full_prompt


async def test_score_judges_each_response_against_its_own_context_only():
    # Regression test for #148: response A's context block must not leak
    # response B's context in the same message, and vice versa.
    fake_client = _FakeOllamaClient(_VALID_PAYLOAD)
    judge = OllamaJudge(client=fake_client, model_id="qwen3.5")

    await judge.score(
        query="q",
        response_a="a",
        response_b="b",
        context_a="ONLY-IN-A-CONTEXT",
        context_b="ONLY-IN-B-CONTEXT",
    )

    full_prompt = str(fake_client.last_call_kwargs["messages"])
    a_index = full_prompt.index("ONLY-IN-A-CONTEXT")
    a_response_index = full_prompt.index("Response A:")
    b_index = full_prompt.index("ONLY-IN-B-CONTEXT")
    b_response_index = full_prompt.index("Response B:")
    # Each context block sits before its own response, not the other one's.
    assert a_index < a_response_index < b_index < b_response_index


async def test_score_requests_json_format():
    fake_client = _FakeOllamaClient(_VALID_PAYLOAD)
    judge = OllamaJudge(client=fake_client, model_id="qwen3.5")

    await judge.score(query="q", response_a="a", response_b="b", context_a="c", context_b="c")

    assert fake_client.last_call_kwargs["format"] == "json"


async def test_score_marks_each_empty_context_explicitly_rather_than_omitting_it():
    fake_client = _FakeOllamaClient(_VALID_PAYLOAD)
    judge = OllamaJudge(client=fake_client, model_id="qwen3.5")

    await judge.score(query="q", response_a="a", response_b="b", context_a="", context_b="")

    # Scoped to the user message only -- the system prompt's own groundedness
    # rubric also mentions the literal string "(none provided)" as guidance,
    # which would inflate a count taken over the whole message list.
    user_message = fake_client.last_call_kwargs["messages"][1]["content"]
    assert user_message.count("(none provided)") == 2


async def test_score_retries_and_succeeds_after_malformed_json():
    # First attempt is truncated/malformed (the exact failure mode observed
    # live in #149's Fort Knox run); second attempt is a valid payload.
    client = _ScriptedOllamaClient(['{"response_a": {"coherence": 3,', json.dumps(_VALID_PAYLOAD)])
    judge = OllamaJudge(client=client, model_id="qwen3.5")

    scores_a, scores_b = await judge.score(
        query="q", response_a="a", response_b="b", context_a="c", context_b="c"
    )

    assert client.call_count == 2
    assert scores_a.parse_failed is False
    assert scores_a.coherence == 3
    assert scores_b.coherence == 5


async def test_score_returns_a_flagged_parse_failure_after_exhausting_retries():
    # Every attempt returns malformed JSON -- the run must not crash.
    client = _ScriptedOllamaClient(["not json at all"] * 5)
    judge = OllamaJudge(client=client, model_id="qwen3.5")

    scores_a, scores_b = await judge.score(
        query="q", response_a="a", response_b="b", context_a="c", context_b="c"
    )

    assert client.call_count == 3  # bounded retry, not unlimited
    assert scores_a.parse_failed is True
    assert scores_b.parse_failed is True
    assert scores_a.unverifiable_claims[0].startswith("JUDGE PARSE FAILURE:")
    assert scores_b.unverifiable_claims[0].startswith("JUDGE PARSE FAILURE:")


async def test_score_retries_on_a_syntactically_valid_but_wrong_shaped_payload():
    # Valid JSON, but missing the expected "response_a"/"response_b" keys --
    # a KeyError, not a JSONDecodeError, and must be retried the same way.
    client = _ScriptedOllamaClient(
        [json.dumps({"unexpected": "shape"}), json.dumps(_VALID_PAYLOAD)]
    )
    judge = OllamaJudge(client=client, model_id="qwen3.5")

    scores_a, _ = await judge.score(
        query="q", response_a="a", response_b="b", context_a="c", context_b="c"
    )

    assert client.call_count == 2
    assert scores_a.parse_failed is False
    assert scores_a.coherence == 3
