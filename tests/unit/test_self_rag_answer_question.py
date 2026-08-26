import uuid

from src.rag.application.self_rag_answer_question import SelfRAGAnswerQuestion
from src.rag.domain.entities import SearchResult


class _FakeSearch:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.called = False

    async def execute(self, tenant_id, query, top_k):
        self.called = True
        return self._results


class _FakeChatModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    async def generate(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        return next(self._responses)


def _result() -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="grounded content", score=0.9)


async def test_answers_directly_and_skips_retrieval_when_the_gate_says_no():
    search = _FakeSearch([_result()])
    chat = _FakeChatModel(["NO", "7 + 5 is 12."])
    use_case = SelfRAGAnswerQuestion(search_documents=search, chat_model=chat, top_k=5)

    answer = await use_case.execute(tenant_id=uuid.uuid4(), question="What is 7 + 5?")

    assert search.called is False
    assert answer.sources == []
    assert answer.answer == "7 + 5 is 12."


async def test_retrieves_and_grounds_the_answer_when_the_gate_says_yes():
    search = _FakeSearch([_result()])
    chat = _FakeChatModel(["YES", "The answer, grounded in context."])
    use_case = SelfRAGAnswerQuestion(search_documents=search, chat_model=chat, top_k=5)

    answer = await use_case.execute(tenant_id=uuid.uuid4(), question="What does RAG.md say about X?")

    assert search.called is True
    assert len(answer.sources) == 1
    # the second chat_model.generate call should have received the retrieved content as context
    assert "grounded content" in chat.calls[1][1]


async def test_gate_parsing_is_tolerant_of_extra_words_and_case():
    search = _FakeSearch([])
    chat = _FakeChatModel(["no, this is general knowledge.", "an answer"])
    use_case = SelfRAGAnswerQuestion(search_documents=search, chat_model=chat, top_k=5)

    await use_case.execute(tenant_id=uuid.uuid4(), question="q")

    assert search.called is False
