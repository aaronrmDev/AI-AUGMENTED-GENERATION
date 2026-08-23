from src.rag.infrastructure.claude_chat_model import ClaudeChatModel


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [type("Block", (), {"text": text})()]


class _FakeMessages:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_call_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeMessage(self._response_text)


class _FakeAnthropicClient:
    def __init__(self, response_text: str) -> None:
        self.messages = _FakeMessages(response_text)


async def test_generate_returns_the_response_text():
    fake_client = _FakeAnthropicClient("The answer is 42.")
    model = ClaudeChatModel(client=fake_client, model_id="claude-opus-5")

    answer = await model.generate(question="What is the answer?", context="Some context.")

    assert answer == "The answer is 42."


async def test_generate_includes_both_question_and_context_in_the_request():
    fake_client = _FakeAnthropicClient("irrelevant")
    model = ClaudeChatModel(client=fake_client, model_id="claude-opus-5")

    await model.generate(question="What is FastAPI?", context="FastAPI is a Python web framework.")

    sent = fake_client.messages.last_call_kwargs
    assert sent["model"] == "claude-opus-5"
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "FastAPI is a Python web framework." in full_prompt
