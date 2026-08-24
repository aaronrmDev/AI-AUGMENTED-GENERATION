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

    async def chat(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeChatResponse(json.dumps(self._payload))


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

    scores_a, scores_b = await judge.score(query="q", response_a="a", response_b="b", context="c")

    assert scores_a.coherence == 3
    assert scores_a.unverifiable_claims == ["claim x"]
    assert scores_b.coherence == 5
    assert scores_b.unverifiable_claims == []


async def test_score_includes_the_query_and_both_responses_in_the_request():
    fake_client = _FakeOllamaClient(_VALID_PAYLOAD)
    judge = OllamaJudge(client=fake_client, model_id="qwen3.5")

    await judge.score(
        query="What is FastAPI?",
        response_a="Response one.",
        response_b="Response two.",
        context="FastAPI is a Python web framework.",
    )

    sent = fake_client.last_call_kwargs
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "Response one." in full_prompt
    assert "Response two." in full_prompt
    assert "FastAPI is a Python web framework." in full_prompt


async def test_score_requests_json_format():
    fake_client = _FakeOllamaClient(_VALID_PAYLOAD)
    judge = OllamaJudge(client=fake_client, model_id="qwen3.5")

    await judge.score(query="q", response_a="a", response_b="b", context="c")

    assert fake_client.last_call_kwargs["format"] == "json"


async def test_score_marks_empty_context_explicitly_rather_than_omitting_it():
    fake_client = _FakeOllamaClient(_VALID_PAYLOAD)
    judge = OllamaJudge(client=fake_client, model_id="qwen3.5")

    await judge.score(query="q", response_a="a", response_b="b", context="")

    full_prompt = str(fake_client.last_call_kwargs["messages"])
    assert "(none provided)" in full_prompt
