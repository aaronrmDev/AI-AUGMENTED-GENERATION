from src.rag.infrastructure.claude_chat_model import ClaudeChatModel


class _FakeMessage:
    def __init__(self, text: str) -> None:
        # A leading thinking block is what a real response from the configured
        # model looks like: claude-opus-5 runs adaptive thinking by default
        # when `thinking` is omitted, and those blocks come first and carry no
        # .text at all. The adapter has to skip past them, so the fake has to
        # produce them -- a single text block would regression-test nothing.
        thinking_block = type(
            "ThinkingBlock", (), {"type": "thinking", "thinking": "some reasoning"}
        )()
        text_block = type("TextBlock", (), {"type": "text", "text": text})()
        self.content = [thinking_block, text_block]


class _FakeTextStream:
    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for delta in self._deltas:
            yield delta


class _FakeMessageStream:
    def __init__(self, deltas: list[str]) -> None:
        self.text_stream = _FakeTextStream(deltas)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeMessages:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_call_kwargs: dict | None = None
        self.last_stream_kwargs: dict | None = None
        self.stream_deltas: list[str] = []

    async def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeMessage(self._response_text)

    def stream(self, **kwargs):
        self.last_stream_kwargs = kwargs
        return _FakeMessageStream(self.stream_deltas)


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


async def test_complete_returns_the_response_text():
    fake_client = _FakeAnthropicClient("a direct completion")
    model = ClaudeChatModel(client=fake_client, model_id="claude-opus-5")

    answer = await model.complete("Write a hypothetical answer to: what is FastAPI?")

    assert answer == "a direct completion"


async def test_complete_sends_no_system_prompt():
    # The bug this batch's final review caught: generate()'s RAG-answering
    # system prompt made the model refuse HyDE/Self-RAG/Multi-Query's non-QA
    # prompts. complete() must never send that system prompt.
    fake_client = _FakeAnthropicClient("irrelevant")
    model = ClaudeChatModel(client=fake_client, model_id="claude-opus-5")

    await model.complete("some prompt")

    sent = fake_client.messages.last_call_kwargs
    assert "system" not in sent
    assert sent["messages"] == [{"role": "user", "content": "some prompt"}]


async def test_stream_yields_each_delta_in_order():
    fake_client = _FakeAnthropicClient("irrelevant")
    fake_client.messages.stream_deltas = ["The ", "answer ", "is 42."]
    model = ClaudeChatModel(client=fake_client, model_id="claude-opus-5")

    chunks = [chunk async for chunk in model.stream(question="q", context="c")]

    assert chunks == ["The ", "answer ", "is 42."]


async def test_stream_sends_the_same_prompt_shape_as_generate():
    fake_client = _FakeAnthropicClient("irrelevant")
    fake_client.messages.stream_deltas = ["x"]
    model = ClaudeChatModel(client=fake_client, model_id="claude-opus-5")

    [_ async for _ in model.stream(question="What is FastAPI?", context="FastAPI is a framework.")]

    sent = fake_client.messages.last_stream_kwargs
    assert sent["model"] == "claude-opus-5"
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "FastAPI is a framework." in full_prompt
