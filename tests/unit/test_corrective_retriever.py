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


async def test_returns_only_the_relevant_results_when_a_majority_pass():
    results = [_result("relevant one"), _result("relevant two"), _result("noise")]
    inner = _FixedRetriever(results)
    chat = _FakeChatModel(["YES", "YES", "NO"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)

    filtered = await retriever.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert [r.content for r in filtered] == ["relevant one", "relevant two"]
    assert inner.queries_seen == ["q"]


async def test_triggers_correction_and_re_searches_when_a_majority_fail():
    initial = [_result("noise one"), _result("noise two"), _result("relevant")]
    corrected = [_result("better result")]
    inner = _FixedRetriever(initial)
    chat = _FakeChatModel(["NO", "NO", "YES", "a refined query"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)

    inner._results = initial

    async def execute_with_correction(tenant_id, query, top_k):
        inner.queries_seen.append(query)
        return corrected if query == "a refined query" else initial[:top_k]

    inner.execute = execute_with_correction

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert [r.content for r in results] == ["better result"]
    assert inner.queries_seen == ["q", "a refined query"]


async def test_a_tie_counts_as_not_a_majority_and_triggers_correction():
    # Strict majority (> half) is required, so an exact 1-of-2 tie must
    # trigger correction, not be treated as "passed".
    initial = [_result("a"), _result("b")]
    corrected = [_result("c")]
    inner = _FixedRetriever(initial)
    chat = _FakeChatModel(["YES", "NO", "refined"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)

    async def execute_with_correction(tenant_id, query, top_k):
        inner.queries_seen.append(query)
        return corrected if query == "refined" else initial[:top_k]

    inner.execute = execute_with_correction

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=2)

    assert [r.content for r in results] == ["c"]


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
    # than includes.
    results = [_result("only candidate")]
    corrected = [_result("corrected")]
    inner = _FixedRetriever(results)
    chat = _FakeChatModel(["unclear", "refined"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)

    async def execute_with_correction(tenant_id, query, top_k):
        inner.queries_seen.append(query)
        return corrected if query == "refined" else results[:top_k]

    inner.execute = execute_with_correction

    filtered = await retriever.execute(tenant_id=_TENANT, query="q", top_k=1)

    assert [r.content for r in filtered] == ["corrected"]


async def test_relevance_and_refinement_prompts_go_through_complete_not_generate():
    results = [_result("noise")]
    corrected = [_result("better")]
    inner = _FixedRetriever(results)
    chat = _FakeChatModel(["NO", "refined"])
    retriever = CorrectiveRetriever(inner=inner, chat_model=chat)

    async def execute_with_correction(tenant_id, query, top_k):
        inner.queries_seen.append(query)
        return corrected if query == "refined" else results[:top_k]

    inner.execute = execute_with_correction

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
