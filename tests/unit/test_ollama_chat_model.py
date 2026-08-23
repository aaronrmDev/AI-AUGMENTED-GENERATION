from src.rag.infrastructure.ollama_chat_model import OllamaChatModel


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = text
        self.thinking = "some reasoning, kept separate from content"


class _FakeChatResponse:
    def __init__(self, text: str) -> None:
        self.message = _FakeMessage(text)


class _FakeOllamaClient:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_call_kwargs: dict | None = None

    async def chat(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeChatResponse(self._response_text)


async def test_generate_returns_the_response_text():
    fake_client = _FakeOllamaClient("The answer is 42.")
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")

    answer = await model.generate(question="What is the answer?", context="Some context.")

    assert answer == "The answer is 42."


async def test_generate_includes_both_question_and_context_in_the_request():
    fake_client = _FakeOllamaClient("irrelevant")
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")

    await model.generate(question="What is FastAPI?", context="FastAPI is a Python web framework.")

    sent = fake_client.last_call_kwargs
    assert sent["model"] == "qwen3.5"
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "FastAPI is a Python web framework." in full_prompt


async def test_generate_returns_empty_string_when_content_is_none():
    fake_client = _FakeOllamaClient("irrelevant")
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")
    response = await fake_client.chat(model="qwen3.5", messages=[])
    response.message.content = None

    async def _chat_returning_none_content(**kwargs):
        return response

    fake_client.chat = _chat_returning_none_content
    answer = await model.generate(question="q", context="c")

    assert answer == ""
