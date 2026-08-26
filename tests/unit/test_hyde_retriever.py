import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.hyde_retriever import HyDERetriever

_TENANT = uuid.uuid4()


class _FakeChatModel:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []
        self.generate_calls: int = 0

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response

    async def generate(self, question: str, context: str) -> str:
        # Tracked, never expected to be called: generate() injects a
        # RAG-answering system prompt that refuses when asked to invent a
        # hypothetical answer with no context -- HyDE must use complete().
        # See src/rag/infrastructure/hyde_retriever.py's comment.
        self.generate_calls += 1
        return self._response


class _RecordingRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_query: str | None = None

    async def execute(self, tenant_id, query, top_k):
        self.last_query = query
        return self._results[:top_k]


def _result() -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="c", score=0.9)


async def test_searches_with_the_generated_hypothetical_answer_not_the_original_question():
    chat = _FakeChatModel("FastAPI's BackgroundTasks lets you run code after returning a response.")
    inner = _RecordingRetriever([_result()])
    retriever = HyDERetriever(inner=inner, chat_model=chat)

    await retriever.execute(tenant_id=_TENANT, query="FastAPI background tasks", top_k=5)

    expected = "FastAPI's BackgroundTasks lets you run code after returning a response."
    assert inner.last_query == expected
    assert inner.last_query != "FastAPI background tasks"


async def test_passes_the_original_question_to_the_chat_model_for_generation():
    chat = _FakeChatModel("a hypothetical answer")
    inner = _RecordingRetriever([])
    retriever = HyDERetriever(inner=inner, chat_model=chat)

    await retriever.execute(tenant_id=_TENANT, query="the real question", top_k=5)

    assert any("the real question" in p for p in chat.prompts)


async def test_returns_the_inner_retrievers_results():
    result = _result()
    chat = _FakeChatModel("hypothetical")
    inner = _RecordingRetriever([result])
    retriever = HyDERetriever(inner=inner, chat_model=chat)

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert results == [result]


async def test_generates_the_hypothetical_answer_via_complete_not_generate():
    # Regression test for the bug this batch's final review caught: HyDE
    # used to call generate(question=prompt, context=""), whose RAG-answering
    # system prompt made the model refuse to invent an answer instead of
    # actually inventing one, silently turning every HyDE search into a
    # search on a mangled paraphrase of the question. complete() carries no
    # such system prompt.
    chat = _FakeChatModel("a hypothetical answer")
    inner = _RecordingRetriever([])
    retriever = HyDERetriever(inner=inner, chat_model=chat)

    await retriever.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert chat.generate_calls == 0
    assert len(chat.prompts) == 1
