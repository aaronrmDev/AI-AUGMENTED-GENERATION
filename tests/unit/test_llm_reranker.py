import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.llm_reranker import LLMReranker


class _FakeChatModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.questions: list[str] = []

    async def generate(self, question: str, context: str) -> str:
        self.questions.append(question)
        return next(self._responses)


def _result(content: str) -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content=content, score=0.5)


async def test_sorts_candidates_by_parsed_score_descending():
    chat = _FakeChatModel(["3", "9"])
    reranker = LLMReranker(chat_model=chat)
    results = [_result("low relevance chunk"), _result("high relevance chunk")]

    reranked = await reranker.rerank(query="q", results=results, top_k=2)

    assert reranked[0].content == "high relevance chunk"


async def test_a_malformed_score_is_treated_as_zero_not_a_crash():
    chat = _FakeChatModel(["not a number", "7"])
    reranker = LLMReranker(chat_model=chat)
    results = [_result("garbled response chunk"), _result("clean response chunk")]

    reranked = await reranker.rerank(query="q", results=results, top_k=2)

    assert reranked[0].content == "clean response chunk"
    assert len(reranked) == 2


async def test_respects_top_k():
    chat = _FakeChatModel(["1", "2", "3"])
    reranker = LLMReranker(chat_model=chat)
    results = [_result("a"), _result("b"), _result("c")]

    reranked = await reranker.rerank(query="q", results=results, top_k=1)

    assert len(reranked) == 1
