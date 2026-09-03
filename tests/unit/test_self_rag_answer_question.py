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
    def __init__(self, complete_responses: list[str], generate_response: str = "an answer") -> None:
        self._complete_responses = iter(complete_responses)
        self._generate_response = generate_response
        self.complete_calls: list[str] = []
        self.generate_calls: list[tuple[str, str]] = []

    async def complete(self, prompt: str) -> str:
        self.complete_calls.append(prompt)
        return next(self._complete_responses)

    async def generate(self, question: str, context: str) -> str:
        self.generate_calls.append((question, context))
        return self._generate_response


def _result() -> SearchResult:
    return SearchResult(
        document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="grounded content", score=0.9
    )


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
    chat = _FakeChatModel(["YES"], generate_response="The answer, grounded in context.")
    use_case = SelfRAGAnswerQuestion(search_documents=search, chat_model=chat, top_k=5)

    answer = await use_case.execute(
        tenant_id=uuid.uuid4(), question="What does RAG.md say about X?"
    )

    assert search.called is True
    assert len(answer.sources) == 1
    # the grounded-answer call must be generate() (uses real context), and
    # must have received the retrieved content as context
    assert len(chat.generate_calls) == 1
    assert "grounded content" in chat.generate_calls[0][1]


async def test_gate_parsing_is_tolerant_of_extra_words_and_case():
    search = _FakeSearch([])
    chat = _FakeChatModel(["no, this is general knowledge.", "an answer"])
    use_case = SelfRAGAnswerQuestion(search_documents=search, chat_model=chat, top_k=5)

    await use_case.execute(tenant_id=uuid.uuid4(), question="q")

    assert search.called is False


async def test_gate_and_no_context_answer_go_through_complete_not_generate():
    # Regression test for the bug this batch's final review caught: both the
    # gate check and the NO-branch "answer from your own knowledge" call used
    # to go through generate(question=..., context=""), whose RAG-answering
    # system prompt ("if the context doesn't contain the answer, say so")
    # made the model refuse instead of doing either -- every live NO-gate
    # answer degraded into a refusal. complete() carries no such system
    # prompt. Only the YES-branch, real-context answer may call generate().
    search = _FakeSearch([_result()])
    chat = _FakeChatModel(["NO", "7 + 5 is 12."])
    use_case = SelfRAGAnswerQuestion(search_documents=search, chat_model=chat, top_k=5)

    await use_case.execute(tenant_id=uuid.uuid4(), question="What is 7 + 5?")

    assert len(chat.complete_calls) == 2
    assert chat.generate_calls == []


async def test_gate_finds_a_no_appearing_after_the_first_ten_characters():
    # The prior implementation only inspected the first 10 characters of the
    # gate response, so a compliant "NO" arriving later in the sentence (a
    # real, observed qwen3.5 response shape) was misread as YES/retrieve.
    search = _FakeSearch([])
    chat = _FakeChatModel(["Based on my knowledge, NO", "an answer"])
    use_case = SelfRAGAnswerQuestion(search_documents=search, chat_model=chat, top_k=5)

    await use_case.execute(tenant_id=uuid.uuid4(), question="q")

    assert search.called is False


async def test_gate_does_not_misread_no_as_a_substring_of_an_unrelated_word():
    # The prior substring check misread "Unknown" as containing "no" and
    # incorrectly skipped retrieval. The word "no" only counts as a whole
    # word.
    search = _FakeSearch([_result()])
    chat = _FakeChatModel(["Unknown", "an answer"])
    use_case = SelfRAGAnswerQuestion(search_documents=search, chat_model=chat, top_k=5)

    await use_case.execute(tenant_id=uuid.uuid4(), question="q")

    assert search.called is True


async def test_gate_defaults_to_retrieval_when_the_response_contains_neither_word():
    # Safe default: an empty or non-compliant gate response should retrieve
    # rather than silently skip -- over-retrieving costs latency, but
    # skipping a genuinely needed retrieval risks a hallucinated answer.
    search = _FakeSearch([_result()])
    chat = _FakeChatModel(["", "an answer"])
    use_case = SelfRAGAnswerQuestion(search_documents=search, chat_model=chat, top_k=5)

    await use_case.execute(tenant_id=uuid.uuid4(), question="q")

    assert search.called is True
