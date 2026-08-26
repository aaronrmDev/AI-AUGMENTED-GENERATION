import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.multi_query_retriever import MultiQueryRetriever

_TENANT = uuid.uuid4()


def _result(chunk_id: uuid.UUID) -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=chunk_id, content="c", score=0.9)


class _FakeChatModel:
    def __init__(self, response: str) -> None:
        self._response = response
        self.questions: list[str] = []

    async def generate(self, question: str, context: str) -> str:
        self.questions.append(question)
        return self._response


class _RecordingRetriever:
    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self._results_by_query = results_by_query
        self.queries_seen: list[str] = []

    async def execute(self, tenant_id, query, top_k):
        self.queries_seen.append(query)
        return self._results_by_query.get(query, [])[:top_k]


async def test_generates_multiple_variants_and_searches_each():
    chat = _FakeChatModel("How does X work?\nWhat is X used for?\nExplain X in detail.")
    inner = _RecordingRetriever({})
    retriever = MultiQueryRetriever(inner=inner, chat_model=chat, num_queries=3)

    await retriever.execute(tenant_id=_TENANT, query="Tell me about X", top_k=5)

    assert len(inner.queries_seen) == 3
    assert "How does X work?" in inner.queries_seen


async def test_a_chunk_found_by_multiple_variants_ranks_higher():
    shared_id = uuid.uuid4()
    only_in_one_id = uuid.uuid4()
    chat = _FakeChatModel("variant one\nvariant two")
    inner = _RecordingRetriever({
        "variant one": [_result(shared_id), _result(uuid.uuid4())],
        "variant two": [_result(only_in_one_id), _result(shared_id)],
    })
    retriever = MultiQueryRetriever(inner=inner, chat_model=chat, num_queries=2)

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=1)

    assert results[0].chunk_id == shared_id


async def test_blank_lines_in_the_chat_response_are_not_treated_as_query_variants():
    chat = _FakeChatModel("variant one\n\n\nvariant two\n")
    inner = _RecordingRetriever({})
    retriever = MultiQueryRetriever(inner=inner, chat_model=chat, num_queries=4)

    await retriever.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert inner.queries_seen == ["variant one", "variant two"]
