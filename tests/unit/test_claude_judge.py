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

    scores_a, scores_b = await judge.score(query="q", response_a="a", response_b="b")

    assert scores_a.coherence == 3
    assert scores_a.unverifiable_claims == ["claim x"]
    assert scores_b.coherence == 5
    assert scores_b.unverifiable_claims == []


async def test_score_includes_the_query_and_both_responses_in_the_request():
    fake_client = _FakeAnthropicClient(_VALID_PAYLOAD)
    judge = ClaudeJudge(client=fake_client, model_id="claude-opus-5")

    await judge.score(
        query="What is FastAPI?", response_a="Response one.", response_b="Response two."
    )

    sent = fake_client.messages.last_call_kwargs
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "Response one." in full_prompt
    assert "Response two." in full_prompt
