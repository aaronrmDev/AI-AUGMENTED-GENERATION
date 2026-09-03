# RAG Multi-Query + HyDE + Self-RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Multi-Query Retrieval, HyDE, and Self-RAG, then measure all three against a real corpus — Self-RAG with a question mix that actually exercises its retrieval-skip gate.

**Architecture:** `MultiQueryRetriever` and `HyDERetriever` both implement `Retriever` and decorate an inner `Retriever`, substituting what string gets embedded as "the query" — the same trick every decorator in this codebase since `RerankingRetriever` has used. `SelfRAGAnswerQuestion` is a sibling to `AnswerQuestion`, not a `Retriever` decorator, because skipping retrieval entirely isn't expressible through that port.

**Tech Stack:** Reuses `ChatModel`, `Retriever`, existing chunkers/embedders. No new external dependency.

**Spec:** `docs/superpowers/specs/2026-08-25-rag-query-enhancement-design.md`

## Global Constraints

- `Retriever.execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]` — every retriever/decorator implements exactly this signature.
- `mypy --strict` and `ruff check src/` must stay clean.
- Every scenario entrypoint script gets a unique module name across the whole `evaluation/scenarios/` tree (e.g. `run_query_enhancement_comparison.py`) — never `run_comparison.py`. Two prior batches hit real regressions from this; do not reintroduce it.
- Tests: pure logic (RRF math, gate-parsing, variant-parsing) may use fakes; anything touching a real model is an integration test with the real dependency.

---

### Task 1: Shared RRF helper, `HybridSearchDocuments` refactor, `MultiQueryRetriever`

**Files:**
- Create: `src/rag/infrastructure/_result_fusion.py` — `reciprocal_rank_fusion`
- Modify: `src/rag/infrastructure/hybrid_search_documents.py` — use the shared helper, behavior-preserving
- Create: `src/rag/infrastructure/multi_query_retriever.py` — `MultiQueryRetriever`
- Test: `tests/unit/test_result_fusion.py`, `tests/unit/test_multi_query_retriever.py`

**Interfaces:**
- Produces: `reciprocal_rank_fusion(result_lists: list[list[SearchResult]], top_k: int, k_rrf: int = 60) -> list[SearchResult]`
- Produces: `MultiQueryRetriever(Retriever).__init__(self, inner: Retriever, chat_model: ChatModel, num_queries: int = 4) -> None`
- Consumes: `Retriever`, `ChatModel.generate(question: str, context: str) -> str` (existing ports)

- [ ] **Step 1: Write the failing tests for the shared helper, migrated from `test_hybrid_search_documents.py`'s existing coverage**

```python
# tests/unit/test_result_fusion.py
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure._result_fusion import reciprocal_rank_fusion


def _result(chunk_id: uuid.UUID, score: float) -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=chunk_id, content=f"chunk {chunk_id}", score=score)


def test_a_chunk_ranked_second_in_both_lists_beats_a_chunk_ranked_first_in_only_one():
    shared_id = uuid.uuid4()
    vector_only_id = uuid.uuid4()
    keyword_only_id = uuid.uuid4()
    vector = [_result(vector_only_id, 0.9), _result(shared_id, 0.5)]
    keyword = [_result(keyword_only_id, 12.0), _result(shared_id, 3.0)]

    results = reciprocal_rank_fusion([vector, keyword], top_k=3)

    assert results[0].chunk_id == shared_id


def test_a_chunk_found_in_only_one_list_still_appears():
    only_id = uuid.uuid4()
    results = reciprocal_rank_fusion([[_result(only_id, 0.9)], [_result(uuid.uuid4(), 5.0)]], top_k=5)
    assert any(r.chunk_id == only_id for r in results)


def test_respects_top_k():
    lists = [[_result(uuid.uuid4(), 1.0) for _ in range(5)] for _ in range(2)]
    results = reciprocal_rank_fusion(lists, top_k=3)
    assert len(results) == 3


def test_works_with_more_than_two_lists():
    shared_id = uuid.uuid4()
    lists = [[_result(shared_id, 1.0), _result(uuid.uuid4(), 0.5)] for _ in range(4)]
    results = reciprocal_rank_fusion(lists, top_k=1)
    # shared_id is rank 0 in all 4 lists -- must win outright
    assert results[0].chunk_id == shared_id
```

- [ ] **Step 2: Run to verify failure, implement the shared helper**

```python
# src/rag/infrastructure/_result_fusion.py
import uuid

from src.rag.domain.entities import SearchResult

_DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], top_k: int, k_rrf: int = _DEFAULT_RRF_K
) -> list[SearchResult]:
    rrf_scores: dict[uuid.UUID, float] = {}
    by_id: dict[uuid.UUID, SearchResult] = {}
    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (k_rrf + rank + 1)
            by_id[result.chunk_id] = result

    merged_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [
        SearchResult(
            document_id=by_id[cid].document_id, chunk_id=cid,
            content=by_id[cid].content, score=rrf_scores[cid],
        )
        for cid in merged_ids[:top_k]
    ]
```

Run: `uv run pytest tests/unit/test_result_fusion.py -v` — expect PASS, 4/4.

- [ ] **Step 3: Refactor `HybridSearchDocuments` to use the shared helper (behavior-preserving)**

```python
# src/rag/infrastructure/hybrid_search_documents.py
import asyncio
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import Retriever
from src.rag.infrastructure._result_fusion import reciprocal_rank_fusion


class HybridSearchDocuments(Retriever):
    def __init__(
        self, vector_retriever: Retriever, keyword_retriever: Retriever, candidate_k: int = 20
    ) -> None:
        self._vector = vector_retriever
        self._keyword = keyword_retriever
        self._candidate_k = candidate_k

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        vector_results, keyword_results = await asyncio.gather(
            self._vector.execute(tenant_id=tenant_id, query=query, top_k=self._candidate_k),
            self._keyword.execute(tenant_id=tenant_id, query=query, top_k=self._candidate_k),
        )
        return reciprocal_rank_fusion([vector_results, keyword_results], top_k=top_k)
```

Run: `uv run pytest tests/unit/test_hybrid_search_documents.py -v` — expect PASS, all 5 existing tests unchanged. This confirms the refactor is behavior-preserving; if any of these 5 fail, the refactor introduced a regression — fix it before proceeding, do not touch the test file to make it pass.

- [ ] **Step 4: Write the failing tests for `MultiQueryRetriever`**

```python
# tests/unit/test_multi_query_retriever.py
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
```

- [ ] **Step 5: Run to verify failure, implement `MultiQueryRetriever`**

```python
# src/rag/infrastructure/multi_query_retriever.py
import asyncio
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import ChatModel, Retriever
from src.rag.infrastructure._result_fusion import reciprocal_rank_fusion

_PROMPT_TEMPLATE = (
    "Generate {n} genuinely different phrasings of the following question, "
    "each viewing it from a different angle (not just reworded synonyms of "
    "each other). Respond with ONLY the {n} phrasings, one per line, no "
    "numbering, no extra commentary.\n\nQuestion: {query}"
)


class MultiQueryRetriever(Retriever):
    def __init__(self, inner: Retriever, chat_model: ChatModel, num_queries: int = 4) -> None:
        self._inner = inner
        self._chat_model = chat_model
        self._num_queries = num_queries

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        prompt = _PROMPT_TEMPLATE.format(n=self._num_queries, query=query)
        response = await self._chat_model.generate(question=prompt, context="")
        variants = [line.strip() for line in response.splitlines() if line.strip()]
        if not variants:
            variants = [query]

        result_lists = await asyncio.gather(
            *[self._inner.execute(tenant_id=tenant_id, query=v, top_k=top_k) for v in variants]
        )
        return reciprocal_rank_fusion(list(result_lists), top_k=top_k)
```

Run: `uv run pytest tests/unit/test_multi_query_retriever.py -v` — expect PASS, 3/3.

- [ ] **Step 6: Full unit + integration suite, ruff, mypy, commit**

```bash
uv run pytest tests/unit/ tests/integration/ -v
uv run ruff check src/rag/
uv run mypy src/rag/
git add src/rag/infrastructure/_result_fusion.py src/rag/infrastructure/hybrid_search_documents.py src/rag/infrastructure/multi_query_retriever.py tests/unit/test_result_fusion.py tests/unit/test_multi_query_retriever.py
git commit -m "feat: extract shared RRF helper, add MultiQueryRetriever"
```

---

### Task 2: `HyDERetriever`

**Files:**
- Create: `src/rag/infrastructure/hyde_retriever.py`
- Test: `tests/unit/test_hyde_retriever.py`

**Interfaces:**
- Consumes: `Retriever`, `ChatModel` (existing ports)
- Produces: `HyDERetriever(Retriever).__init__(self, inner: Retriever, chat_model: ChatModel) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_hyde_retriever.py
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
```

- [ ] **Step 2: Run to verify failure, implement `HyDERetriever`**

```python
# src/rag/infrastructure/hyde_retriever.py
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import ChatModel, Retriever

_PROMPT_TEMPLATE = (
    "Write a short, confident, hypothetical answer to the following question, "
    "as if you already knew the answer with certainty. It's fine if the "
    "specific details you invent aren't accurate -- the goal is realistic "
    "phrasing and vocabulary, not factual correctness. Respond with ONLY the "
    "hypothetical answer, no preamble.\n\nQuestion: {query}"
)


class HyDERetriever(Retriever):
    def __init__(self, inner: Retriever, chat_model: ChatModel) -> None:
        self._inner = inner
        self._chat_model = chat_model

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        prompt = _PROMPT_TEMPLATE.format(query=query)
        hypothetical_answer = await self._chat_model.generate(question=prompt, context="")
        return await self._inner.execute(
            tenant_id=tenant_id, query=hypothetical_answer, top_k=top_k
        )
```

Run: `uv run pytest tests/unit/test_hyde_retriever.py -v` — expect PASS, 3/3.

- [ ] **Step 3: Full suite, ruff, mypy, commit**

```bash
uv run pytest tests/unit/ tests/integration/ -v
uv run ruff check src/rag/
uv run mypy src/rag/
git add src/rag/infrastructure/hyde_retriever.py tests/unit/test_hyde_retriever.py
git commit -m "feat: add HyDERetriever"
```

---

### Task 3: `SelfRAGAnswerQuestion`

**Files:**
- Create: `src/rag/application/self_rag_answer_question.py`
- Test: `tests/unit/test_self_rag_answer_question.py`

**Interfaces:**
- Consumes: `Retriever`, `ChatModel`, `ChatAnswer` (existing, `src/rag/domain/entities.py`)
- Produces: `SelfRAGAnswerQuestion.__init__(self, search_documents: Retriever, chat_model: ChatModel, top_k: int) -> None`, `.execute(self, tenant_id: uuid.UUID, question: str) -> ChatAnswer`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_self_rag_answer_question.py
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
```

- [ ] **Step 2: Run to verify failure, implement `SelfRAGAnswerQuestion`**

```python
# src/rag/application/self_rag_answer_question.py
import uuid

from src.rag.domain.entities import ChatAnswer
from src.rag.domain.ports import ChatModel, Retriever

_GATE_PROMPT_TEMPLATE = (
    "Does answering the following question require looking up external, "
    "private, or recent information that you would not already know from "
    "general training -- or can it be answered correctly from general "
    "knowledge alone (common facts, basic arithmetic, well-known concepts)? "
    "Respond with ONLY YES (retrieval needed) or NO (no retrieval needed), "
    "nothing else.\n\nQuestion: {question}"
)


class SelfRAGAnswerQuestion:
    def __init__(self, search_documents: Retriever, chat_model: ChatModel, top_k: int) -> None:
        self._search = search_documents
        self._chat_model = chat_model
        self._top_k = top_k

    async def execute(self, tenant_id: uuid.UUID, question: str) -> ChatAnswer:
        gate_prompt = _GATE_PROMPT_TEMPLATE.format(question=question)
        gate_response = await self._chat_model.generate(question=gate_prompt, context="")
        # Tolerant parse, same convention as LLMReranker's score parsing:
        # check the first ~10 characters for "no" before "yes", since a
        # response starting "NO, this is..." should never be read as
        # containing "yes" from somewhere later in the sentence.
        needs_retrieval = "no" not in gate_response.strip().lower()[:10]

        if not needs_retrieval:
            answer = await self._chat_model.generate(question=question, context="")
            return ChatAnswer(answer=answer, sources=[])

        sources = await self._search.execute(tenant_id=tenant_id, query=question, top_k=self._top_k)
        context = "\n\n".join(source.content for source in sources)
        answer = await self._chat_model.generate(question=question, context=context)
        return ChatAnswer(answer=answer, sources=sources)
```

Run: `uv run pytest tests/unit/test_self_rag_answer_question.py -v` — expect PASS, 3/3.

- [ ] **Step 3: Full suite, ruff, mypy, commit**

```bash
uv run pytest tests/unit/ tests/integration/ -v
uv run ruff check src/rag/
uv run mypy src/rag/
git add src/rag/application/self_rag_answer_question.py tests/unit/test_self_rag_answer_question.py
git commit -m "feat: add SelfRAGAnswerQuestion"
```

---

### Task 4: Scenario, live measurement, GitHub results

**Files:**
- Create: `evaluation/scenarios/rag-query-enhancement/corpus/rag.md`
- Create: `evaluation/scenarios/rag-query-enhancement/queries.yaml` — **7 questions**: the same 5 RAG.md-grounded questions this project's scenarios have used (read the live file yourself, don't reuse another batch's quotes verbatim — pick 5 fresh ones covering Multi-Query's own recall figure, HyDE's own mechanic/impact figure, Self-RAG's own mechanic/impact figure, and 2 more of your choice), **plus 2 general-knowledge questions** that need no document lookup at all (e.g. a basic arithmetic question, a well-known general software-terms question) specifically for the Self-RAG run to exercise its NO branch.
- Create: `evaluation/scenarios/rag-query-enhancement/run_query_enhancement_comparison.py` — **name it exactly this, never `run_comparison.py`** (two prior batches broke `mypy evaluation/` on that collision; the fix both times was a unique module name per scenario).

**Interfaces:**
- Consumes: `MultiQueryRetriever`, `HyDERetriever` (Tasks 1-2, both `Retriever`s, composed exactly like every other batch's strategies), `SelfRAGAnswerQuestion` (Task 3, used in place of `AnswerQuestion` only for the `self-rag` strategy). Pattern the whole script on `evaluation/scenarios/rag-parent-doc-compression/run_parent_doc_compression_comparison.py` (the most recent, most reviewed one) — same `PYTHONPATH`/`filename="rag.txt"`/`report_path.parent.mkdir`/`OllamaJudge`/in-memory-repository conventions, same `dataclasses.replace()`-after-`execute()` pattern if you need to append anything computed during `treatment()` calls to the notes field (do NOT build it into the `notes=` keyword argument directly — that argument is evaluated before `execute()` ever calls `treatment()`, a real ordering bug an earlier batch's controller had to catch and fix in exactly this scenario).

- [ ] **Step 1: Corpus**

```bash
cp docs/architecture/RAG.md evaluation/scenarios/rag-query-enhancement/corpus/rag.md
```

- [ ] **Step 2: `queries.yaml` — 5 RAG.md-grounded + 2 general-knowledge, all real**

Read the current `docs/architecture/RAG.md` yourself. Quote the exact source sentence for each of the 5 grounded questions in your report. For the 2 general-knowledge questions, no RAG.md quote is needed (that's the point) — just make sure their `success_criterion` is checkable from the answer text alone regardless of whether any context was used (e.g. the arithmetic answer, or the well-known term's definition).

- [ ] **Step 3: `run_query_enhancement_comparison.py` with 3 named strategies**

```python
def _make_retriever_or_answerer(name, vector_retriever, chat_model, top_k):
    # multi-query and hyde return a Retriever, wired into a plain AnswerQuestion
    # like every other batch. self-rag returns a fully-built SelfRAGAnswerQuestion
    # instead, since it isn't a Retriever at all -- branch the whole
    # answer_question construction on strategy name, don't force self-rag
    # through a shape it doesn't fit.
    ...
```

Every strategy uploads via plain `UploadDocument`/`FixedSizeChunker` (none of these three techniques touch chunking or need the two-tier parent index). Baseline = no-RAG, same as every prior batch. Judge: `OllamaJudge`, same caveat text. **Proactively disclose** (matching the last two batches' pattern) that Multi-Query and HyDE both make an extra LLM call per query before retrieval runs, which the prior `rerank-llm` finding already showed can add substantial latency with a reasoning model — don't wait for a review to point this out.

For the `self-rag` strategy specifically: print each question's gate decision (YES/retrieved or NO/skipped) to stdout as the run executes, so the controller has a real, honest record of the gate's actual behavior on all 7 questions to report — the standard harness has no field for this, and inventing one under this batch's time constraints isn't worth it; printing and reading the real output is.

- [ ] **Step 4: Report to the controller, do NOT run live comparisons**

Same boundary as every prior batch's Task 4 first half. Run `uv run pytest tests/unit/ tests/integration/ -v`, `uv run mypy src/rag/ evaluation/` (the full two-directory form), `uv run ruff check src/rag/ evaluation/`, report the real output. Status DONE or DONE_WITH_CONCERNS with all 7 quoted questions/sources (or explicit "no RAG.md quote needed, general knowledge" for the 2) and any deviations.

(Steps 5+ — bringing up Qdrant/Ollama, running the 3 live comparisons, reading the self-rag gate's real printed decisions, committing, posting to GitHub issues #52, #104, #65, #111, #68, #115 — are the controller's own.)
