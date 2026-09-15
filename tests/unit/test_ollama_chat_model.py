from src.rag.infrastructure.ollama_chat_model import OllamaChatModel


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = text
        self.thinking = "some reasoning, kept separate from content"


class _FakeChatResponse:
    def __init__(
        self, text: str, prompt_eval_count: int = 0, eval_count: int = 0
    ) -> None:
        self.message = _FakeMessage(text)
        # Real ollama.ChatResponse carries these as top-level fields
        # (verified against the installed client's ChatResponse.model_fields,
        # not assumed) -- defaulted to 0 so every pre-#147 test constructing
        # this fake without them keeps passing unchanged.
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count


class _FakeOllamaClient:
    def __init__(
        self, response_text: str, prompt_eval_count: int = 0, eval_count: int = 0
    ) -> None:
        self._response_text = response_text
        self._prompt_eval_count = prompt_eval_count
        self._eval_count = eval_count
        self.last_call_kwargs: dict | None = None

    async def chat(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeChatResponse(self._response_text, self._prompt_eval_count, self._eval_count)


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


async def test_complete_returns_the_response_text():
    fake_client = _FakeOllamaClient("a direct completion")
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")

    answer = await model.complete("Write a hypothetical answer to: what is FastAPI?")

    assert answer == "a direct completion"


async def test_complete_sends_no_system_prompt():
    # The bug this batch's final review caught: generate()'s RAG-answering
    # system prompt ("if the context doesn't contain the answer, say so")
    # made the model refuse HyDE/Self-RAG/Multi-Query's non-QA prompts.
    # complete() must never send that system prompt.
    fake_client = _FakeOllamaClient("irrelevant")
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")

    await model.complete("some prompt")

    sent = fake_client.last_call_kwargs
    assert sent["messages"] == [{"role": "user", "content": "some prompt"}]
    assert all(m["role"] != "system" for m in sent["messages"])


def test_last_token_counts_start_at_zero_before_any_call():
    model = OllamaChatModel(client=_FakeOllamaClient("irrelevant"), model_id="qwen3.5")

    assert model.last_input_tokens == 0
    assert model.last_output_tokens == 0


async def test_generate_records_real_token_counts_from_the_response():
    # #147: these must be the real counts ollama's ChatResponse carries, not
    # a caller-side estimate or a hardcoded 0.
    fake_client = _FakeOllamaClient("answer", prompt_eval_count=42, eval_count=17)
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")

    await model.generate(question="q", context="c")

    assert model.last_input_tokens == 42
    assert model.last_output_tokens == 17


async def test_complete_records_real_token_counts_from_the_response():
    fake_client = _FakeOllamaClient("answer", prompt_eval_count=9, eval_count=3)
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")

    await model.complete("some prompt")

    assert model.last_input_tokens == 9
    assert model.last_output_tokens == 3


async def test_last_token_counts_reflect_only_the_most_recent_call():
    fake_client = _FakeOllamaClient("answer", prompt_eval_count=100, eval_count=100)
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")
    await model.generate(question="first", context="c")

    fake_client._prompt_eval_count = 5
    fake_client._eval_count = 2
    await model.generate(question="second", context="c")

    assert model.last_input_tokens == 5
    assert model.last_output_tokens == 2
