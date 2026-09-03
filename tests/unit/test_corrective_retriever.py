import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.corrective_retriever import CorrectiveRetriever

_TENANT = uuid.uuid4()


class _FakeChatModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []
        self.generate_calls: int = 0

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self._responses)

    async def generate(self, question: str, context: str) -> str:
        # Tracked, never expected to be called -- see corrective_retriever.py's
        # comment on why relevance/refinement prompts must use complete().
        self.generate_calls += 1
        return "unexpected"


class _FixedRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.queries_seen: list[str] = []

    async def execute(self, tenant_id, query, top_k):
        self.queries_seen.append(query)
        return self._results[:top_k]


def _result(content: str = "c") -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content=content, score=0.9)


def _with_correction(inner: _FixedRetriever, initial: list[SearchResult], corrected: list[SearchResult]):
    # By call order, not by query text: the blank-refined-query fallback
    # test re-searches with the SAME query string as the original call, so
    # distinguishing "first call" from "second call" by query text would be
    # wrong in that case.
    call_count = 0

    async def execute_with_correction(tenant_id, query, top_k):
        nonlocal call_count
        inner.queries_seen.append(query)
        call_count += 1
        return initial[:top_k] if call_count == 1 else corrected

    inner.execute = execute_with_correction


async def test_returns_the_relevant_results_when_most_pass():
    results = [_result("relevant one"), _result("relevant two"), _result("noise")]
    inner = _FixedRetriever(results)
    chat = _FakeChatModel(["YES", "YES", "NO"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)

    filtered = await retriever.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert {r.content for r in filtered} == {"relevant one", "relevant two"}
    assert inner.queries_seen == ["q"]


async def test_returns_a_single_relevant_result_out_of_five_without_triggering_correction():
    # The exact regime this batch's final review caught the original
    # ">half must pass" threshold breaking: the answer lives in exactly one
    # of top_k=5 retrieved chunks (the ordinary case for a precise factual
    # query). A single correct match is never a majority of 5, so the old
    # rule discarded it and replaced it with an unvalidated re-search on
    # 5 of 7 measured questions. The fixed rule -- return whatever passed
    # whenever ANYTHING passed -- must return just the one relevant chunk
    # here, with no correction call at all.
    results = [_result("relevant"), _result("n1"), _result("n2"), _result("n3"), _result("n4")]
    inner = _FixedRetriever(results)
    chat = _FakeChatModel(["YES", "NO", "NO", "NO", "NO"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)

    filtered = await retriever.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert [r.content for r in filtered] == ["relevant"]
    assert inner.queries_seen == ["q"]


async def test_triggers_correction_and_re_searches_when_nothing_passes():
    initial = [_result("noise one"), _result("noise two"), _result("noise three")]
    corrected = [_result("better result")]
    inner = _FixedRetriever(initial)
    chat = _FakeChatModel(["NO", "NO", "NO", "refined query"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)
    _with_correction(inner, initial, corrected)

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert [r.content for r in results] == ["better result"]
    assert inner.queries_seen == ["q", "refined query"]


async def test_empty_inner_results_return_immediately_with_no_evaluation_calls():
    inner = _FixedRetriever([])
    chat = _FakeChatModel([])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert results == []
    assert chat.prompts == []


async def test_an_ambiguous_relevance_response_defaults_to_not_relevant():
    # Opposite default direction from Self-RAG's gate: CRAG exists to keep
    # untrustworthy content out, so an unparseable judgment excludes rather
    # than includes. With the only candidate excluded, zero results pass,
    # which correctly triggers correction.
    results = [_result("only candidate")]
    corrected = [_result("corrected")]
    inner = _FixedRetriever(results)
    chat = _FakeChatModel(["unclear", "refined"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)
    _with_correction(inner, results, corrected)

    filtered = await retriever.execute(tenant_id=_TENANT, query="q", top_k=1)

    assert [r.content for r in filtered] == ["corrected"]


async def test_relevance_and_refinement_prompts_go_through_complete_not_generate():
    results = [_result("noise")]
    corrected = [_result("better")]
    inner = _FixedRetriever(results)
    chat = _FakeChatModel(["NO", "refined"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)
    _with_correction(inner, results, corrected)

    await retriever.execute(tenant_id=_TENANT, query="q", top_k=1)

    assert chat.generate_calls == 0
    assert len(chat.prompts) == 2


async def test_evaluates_each_result_exactly_once():
    results = [_result("a"), _result("b"), _result("c")]
    inner = _FixedRetriever(results)
    chat = _FakeChatModel(["YES", "YES", "YES"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)

    await retriever.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert len(chat.prompts) == 3


async def test_a_blank_refined_query_falls_back_to_the_original_query():
    # Guard against a degenerate completion: the model ignoring "respond
    # with ONLY the alternative query" and returning nothing usable (or
    # whitespace) must not re-search with an empty string.
    results = [_result("noise")]
    corrected = [_result("fallback result")]
    inner = _FixedRetriever(results)
    chat = _FakeChatModel(["NO", "   "])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)
    _with_correction(inner, results, corrected)

    filtered = await retriever.execute(tenant_id=_TENANT, query="q", top_k=1)

    assert [r.content for r in filtered] == ["fallback result"]
    assert inner.queries_seen == ["q", "q"]
