import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.hyde_retriever import HyDERetriever

_TENANT = uuid.uuid4()


class _FakeChatModel:
    def __init__(self, response: str) -> None:
        self._response = response
        self.questions: list[str] = []

    async def generate(self, question: str, context: str) -> str:
        self.questions.append(question)
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

    assert inner.last_query == "FastAPI's BackgroundTasks lets you run code after returning a response."
    assert inner.last_query != "FastAPI background tasks"


async def test_passes_the_original_question_to_the_chat_model_for_generation():
    chat = _FakeChatModel("a hypothetical answer")
    inner = _RecordingRetriever([])
    retriever = HyDERetriever(inner=inner, chat_model=chat)

    await retriever.execute(tenant_id=_TENANT, query="the real question", top_k=5)

    assert any("the real question" in q for q in chat.questions)


async def test_returns_the_inner_retrievers_results():
    result = _result()
    chat = _FakeChatModel("hypothetical")
    inner = _RecordingRetriever([result])
    retriever = HyDERetriever(inner=inner, chat_model=chat)

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert results == [result]
