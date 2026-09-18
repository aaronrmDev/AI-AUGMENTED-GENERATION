# Streaming Answers over the Unified Session Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /sessions/{session_id}/answers/stream`, a Server-Sent Events endpoint alongside the existing `POST /sessions/{session_id}/answers`, reporting routing/retrieval/budget progress and then the generated answer token-by-token.

**Architecture:** `ChatModel` gains a `stream()` port method; `UnifiedAnswerQuestion` gains a `stream()` method sharing its route/cascade/budget helpers with `execute()`, yielding domain events instead of returning one final value; `AnswerInSession` gains a matching `stream()` with the same ownership check `execute()` uses. Both `stream()` methods are eager coroutines (no `yield` in their own bodies) that do their cheap checks (session ownership; the query-token-budget check) before returning the actual lazy event generator — this is what keeps a cross-tenant denial a plain 404 instead of a stream that opens and then errors. A new `src/api/sse.py` renders those domain events as SSE frames for a new router endpoint.

**Tech Stack:** FastAPI's `StreamingResponse`, Server-Sent Events (`text/event-stream`), `httpx`'s streaming client for integration tests. No new infrastructure — this reuses the Redis-backed rate limiter and Postgres/Qdrant already wired for the JSON endpoint.

**Spec:** `docs/superpowers/specs/2026-09-17-streaming-answers-design.md`

## Global Constraints

- `ChatModel.stream()` signature: `async def stream(self, question: str, context: str) -> AsyncIterator[str]`, yielding text deltas (never the cumulative string).
- Ports stay `ABC` + `@abstractmethod`, never `typing.Protocol` — except `SessionQuestionAnswerer` in `answer_in_session.py`, which is this codebase's one pre-existing `Protocol` exception and is unchanged by this rule; it just gains a matching `stream` signature.
- Test doubles use the `Fake*` naming already established (`FakeChatModel`, `FakeSessionBudgetRecorder`, ...).
- No new rate-limit bucket: the streaming endpoint reuses `chat:{user_id}` and `global:chat`, the exact same keys and limits `POST /sessions/{session_id}/answers` already checks.
- No `response_model` on the streaming route — `StreamingResponse` has no Pydantic body to validate.
- No per-stage latency instrumentation on the streaming path (explicit non-goal; `execute()`'s `StageTimings` is unaffected and stays the only latency-measured path).
- SSE payloads never expose router scores, similarity scores, or stage timings — the same restriction `answer_response()` already applies to the JSON endpoint (spec decision 8).
- `tenant_id`/`user_id` are always derived from `Caller`, never the request body — unchanged from the JSON endpoint, and this plan does not touch that.
- Mypy strict and ruff must stay clean (`uv run mypy src`, `uv run ruff check src tests`) after every task.

---

### Task 1: `ChatModel.stream()` — the port method and every adapter that must implement it

**Files:**
- Modify: `src/rag/domain/ports.py` (the `ChatModel` ABC)
- Modify: `src/rag/infrastructure/ollama_chat_model.py`
- Modify: `src/rag/infrastructure/claude_chat_model.py`
- Modify: `tests/integration/orchestration_env.py` (`ContextEchoChatModel`)
- Modify: `tests/unit/rag_fakes.py` (`FakeChatModel`)
- Modify: `evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py` (`_FreshPassageChatModel`)
- Test: `tests/unit/test_ollama_chat_model.py`, `tests/unit/test_claude_chat_model.py`

**Interfaces:**
- Produces: `ChatModel.stream(question: str, context: str) -> AsyncIterator[str]`, implemented by all five subclasses above. Every later task that needs a chat model with a working `stream()` (Tasks 2 and 4) depends on this.

`ChatModel` is an `ABC`: the moment `stream` becomes an `@abstractmethod` on it, every existing subclass stops being instantiable until it implements the new method too — that break is expected and is this task's own "red" state, not a bug to route around.

- [ ] **Step 1: Add the abstract method to the port**

```python
# src/rag/domain/ports.py -- inside class ChatModel(ABC), directly below `complete`
from collections.abc import AsyncIterator  # add to the top-of-file imports


class ChatModel(ABC):
    @abstractmethod
    async def generate(self, question: str, context: str) -> str: ...

    @abstractmethod
    async def complete(self, prompt: str) -> str: ...

    @abstractmethod
    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        """Same contract as generate(), but yields text deltas instead of
        returning the whole answer at once."""
        ...
```

This matches `generate()`/`complete()`'s own style exactly: a body of just `...` is mypy's documented stub exemption from "missing return statement," regardless of the declared return type, so this needs nothing more even though its return type is `AsyncIterator[str]` rather than `str`. The declared return type alone is what makes `chat_model.stream(...)` usable as `async for delta in ...` at every call site and in every subclass override — nothing in the abstract body itself needs to look like a generator.

- [ ] **Step 2: Run the existing chat-model test files to see the expected break**

Run: `uv run pytest tests/unit/test_ollama_chat_model.py tests/unit/test_claude_chat_model.py -v`
Expected: every test FAILs with `TypeError: Can't instantiate abstract class OllamaChatModel without an implementation for abstract method 'stream'` (and the same for `ClaudeChatModel`) — confirming the port change is what's driving the break, not a typo.

- [ ] **Step 3: Write the new `stream()`-specific tests**

Append to `tests/unit/test_ollama_chat_model.py`:

```python
async def test_stream_yields_each_chunks_content_in_order():
    fake_client = _FakeOllamaClient("irrelevant", stream_chunks=["Hel", "lo, ", "world."])
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")

    chunks = [chunk async for chunk in model.stream(question="q", context="c")]

    assert chunks == ["Hel", "lo, ", "world."]


async def test_stream_skips_a_chunk_with_no_content():
    # ollama's real streaming carries a final chunk whose content is empty --
    # skipping falsy content keeps that chunk from becoming a spurious "" delta.
    fake_client = _FakeOllamaClient("irrelevant", stream_chunks=["first", None, "last"])
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")

    chunks = [chunk async for chunk in model.stream(question="q", context="c")]

    assert chunks == ["first", "last"]


async def test_stream_sends_stream_true_and_the_same_prompt_shape_as_generate():
    fake_client = _FakeOllamaClient("irrelevant", stream_chunks=["x"])
    model = OllamaChatModel(client=fake_client, model_id="qwen3.5")

    [_ async for _ in model.stream(question="What is FastAPI?", context="FastAPI is a framework.")]

    sent = fake_client.last_call_kwargs
    assert sent["model"] == "qwen3.5"
    assert sent["stream"] is True
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "FastAPI is a framework." in full_prompt
```

Modify `_FakeOllamaClient` in the same file so `stream_chunks` is a constructor option and `chat()` honors `stream=True`:

```python
class _FakeOllamaClient:
    def __init__(
        self,
        response_text: str,
        prompt_eval_count: int = 0,
        eval_count: int = 0,
        stream_chunks: list[str | None] | None = None,
    ) -> None:
        self._response_text = response_text
        self._prompt_eval_count = prompt_eval_count
        self._eval_count = eval_count
        self._stream_chunks = stream_chunks or []
        self.last_call_kwargs: dict | None = None

    async def chat(self, **kwargs):
        self.last_call_kwargs = kwargs
        if kwargs.get("stream"):
            return self._stream_response()
        return _FakeChatResponse(self._response_text, self._prompt_eval_count, self._eval_count)

    async def _stream_response(self):
        for text in self._stream_chunks:
            yield _FakeChatResponse(text if text is not None else "")
```

`_FakeChatResponse`'s existing constructor already accepts a plain string for `text` and wraps it in `_FakeMessage(text)`, whose `.content` is that string — this reuses it unchanged for stream chunks too.

Append to `tests/unit/test_claude_chat_model.py`:

```python
async def test_stream_yields_each_delta_in_order():
    fake_client = _FakeAnthropicClient("irrelevant")
    fake_client.messages.stream_deltas = ["The ", "answer ", "is 42."]
    model = ClaudeChatModel(client=fake_client, model_id="claude-opus-5")

    chunks = [chunk async for chunk in model.stream(question="q", context="c")]

    assert chunks == ["The ", "answer ", "is 42."]


async def test_stream_sends_the_same_prompt_shape_as_generate():
    fake_client = _FakeAnthropicClient("irrelevant")
    fake_client.messages.stream_deltas = ["x"]
    model = ClaudeChatModel(client=fake_client, model_id="claude-opus-5")

    [_ async for _ in model.stream(question="What is FastAPI?", context="FastAPI is a framework.")]

    sent = fake_client.messages.last_stream_kwargs
    assert sent["model"] == "claude-opus-5"
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "FastAPI is a framework." in full_prompt
```

Modify `_FakeMessages`/`_FakeAnthropicClient` in the same file to add a `stream()` method matching the real `anthropic` SDK's shape (`messages.stream(...)` is a plain, non-async method that returns an async context manager — not itself awaited):

```python
class _FakeTextStream:
    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for delta in self._deltas:
            yield delta


class _FakeMessageStream:
    def __init__(self, deltas: list[str]) -> None:
        self.text_stream = _FakeTextStream(deltas)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeMessages:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_call_kwargs: dict | None = None
        self.last_stream_kwargs: dict | None = None
        self.stream_deltas: list[str] = []

    async def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeMessage(self._response_text)

    def stream(self, **kwargs):
        self.last_stream_kwargs = kwargs
        return _FakeMessageStream(self.stream_deltas)
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/test_ollama_chat_model.py tests/unit/test_claude_chat_model.py -v -k stream`
Expected: FAIL with `TypeError: Can't instantiate abstract class` (same as Step 2 — `stream()` still isn't implemented on either adapter).

- [ ] **Step 5: Implement `stream()` on every subclass**

```python
# src/rag/infrastructure/ollama_chat_model.py -- add import at top
from collections.abc import AsyncIterator

# add as a new method on OllamaChatModel, after complete()
    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        stream = await self._client.chat(
            model=self._model_id,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            stream=True,
        )
        async for chunk in stream:
            content = chunk.message.content
            if content:
                yield content
```

```python
# src/rag/infrastructure/claude_chat_model.py -- add import at top
from collections.abc import AsyncIterator

# add as a new method on ClaudeChatModel, after complete()
    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self._model_id,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        ) as stream:
            async for text in stream.text_stream:
                yield text
```

```python
# tests/integration/orchestration_env.py -- add import at top
from collections.abc import AsyncIterator

# replace the existing ContextEchoChatModel class with:
class ContextEchoChatModel(ChatModel):
    async def generate(self, question: str, context: str) -> str:
        return context

    async def complete(self, prompt: str) -> str:
        return prompt

    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        # Two chunks (never one), unless there's nothing to say -- a real
        # integration test reassembling multiple chunks exercises real
        # reassembly, not a degenerate single-chunk case.
        if not context:
            return
        midpoint = max(1, len(context) // 2)
        yield context[:midpoint]
        yield context[midpoint:]
```

```python
# tests/unit/rag_fakes.py -- add import at top if not already present
from collections.abc import AsyncIterator

# replace the existing FakeChatModel class with:
class FakeChatModel(ChatModel):
    def __init__(
        self,
        response: str = "a fake answer",
        *,
        stream_chunk_size: int = 4,
        stream_error: Exception | None = None,
    ) -> None:
        self._response = response
        self._stream_chunk_size = stream_chunk_size
        self._stream_error = stream_error
        self.last_question: str | None = None
        self.last_context: str | None = None
        self.last_prompt: str | None = None

    async def generate(self, question: str, context: str) -> str:
        self.last_question = question
        self.last_context = context
        return self._response

    async def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response

    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        self.last_question = question
        self.last_context = context
        for start in range(0, len(self._response), self._stream_chunk_size):
            yield self._response[start : start + self._stream_chunk_size]
        if self._stream_error is not None:
            raise self._stream_error
```

`stream_error` is raised only after every chunk of `response` has already been yielded, so a test using it exercises "some chunks streamed, then a failure" rather than "failed before anything streamed" — the realistic shape of a generation failure partway through.

```python
# evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py
# add import near the top if AsyncIterator isn't already imported there
from collections.abc import AsyncIterator

# add as a new method on _FreshPassageChatModel, after generate()
    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        # This harness measures classification cost, not generation streaming --
        # a single-chunk stand-in keeps it instantiable without adding a
        # streaming-specific measurement nothing here asks for.
        yield await self.generate(question, context)
```

- [ ] **Step 6: Run the full affected test files to verify everything passes**

Run: `uv run pytest tests/unit/test_ollama_chat_model.py tests/unit/test_claude_chat_model.py -v`
Expected: PASS, every test including the pre-existing ones from before this task.

Run: `uv run mypy src` and `uv run ruff check src tests`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/rag/domain/ports.py src/rag/infrastructure/ollama_chat_model.py \
  src/rag/infrastructure/claude_chat_model.py tests/integration/orchestration_env.py \
  tests/unit/rag_fakes.py "evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py" \
  tests/unit/test_ollama_chat_model.py tests/unit/test_claude_chat_model.py
git commit -m "feat(#195): add ChatModel.stream() and implement it on every adapter"
```

---

### Task 2: `UnifiedAnswerQuestion.stream()` and `AnswerInSession.stream()`

**Files:**
- Modify: `src/orchestration/application/unified_answer_question.py`
- Modify: `src/orchestration/application/answer_in_session.py`
- Test: `tests/unit/test_unified_answer_question.py`
- Test: `tests/unit/test_answer_in_session.py`

**Interfaces:**
- Consumes: `ChatModel.stream()` from Task 1.
- Produces: `UnifiedAnswerQuestion.stream(tenant_id, user_id, session_id, question) -> AsyncIterator[UnifiedAnswerEvent]` and the `UnifiedAnswerEvent` union (`RoutingStageEvent`, `RetrievalStageEvent`, `BudgetStageEvent`, `AnswerChunkEvent`, `AnswerCompleteEvent`, `AnswerErrorEvent`); `AnswerInSession.stream(tenant_id, user_id, session_id, question) -> AsyncIterator[UnifiedAnswerEvent]`. Task 3's router depends on both.

This task refactors `execute()` to share three private helpers with the new `stream()` — `execute()`'s observable behavior (return value, `StageTimings`, exceptions) does not change, and its full existing test suite must stay green unmodified as proof.

- [ ] **Step 1: Write the failing tests**

Add these imports to `tests/unit/test_unified_answer_question.py` (alongside the existing ones):

```python
from src.orchestration.application.unified_answer_question import (
    AnswerChunkEvent,
    AnswerCompleteEvent,
    AnswerErrorEvent,
    BudgetStageEvent,
    RetrievalStageEvent,
    RoutingStageEvent,
    UnifiedAnswerQuestion,
)
```

(This replaces the existing single-name import of `UnifiedAnswerQuestion` from that module — keep everything else in the file unchanged.)

Append these tests to the same file:

```python
async def test_stream_yields_the_stage_events_then_the_chunks_then_done():
    classifier = FakeQueryClassifier(_RAG_ONLY)
    rag = FakeCascadeTier(RAG, _hit(RAG, "fresh doc"))
    chat_model = FakeChatModel("the answer", stream_chunk_size=4)
    recorder = FakeSessionBudgetRecorder()
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(),
        classifier,
        LatencyCascade([rag], _GENEROUS),
        chat_model,
        budget_recorder=recorder,
    )
    tenant_id, user_id, session_id = _ids()

    events = [
        event async for event in await use_case.stream(tenant_id, user_id, session_id, _QUESTION)
    ]

    routing, retrieval, budget, *rest = events
    assert isinstance(routing, RoutingStageEvent)
    assert routing.decision is not None
    assert routing.decision.paradigms == frozenset({RAG})
    assert isinstance(retrieval, RetrievalStageEvent)
    assert [attempt.paradigm for attempt in retrieval.attempts] == [RAG]
    assert retrieval.degraded is False
    assert isinstance(budget, BudgetStageEvent)
    assert [source.content for source in budget.sources] == ["fresh doc"]
    assert budget.dropped == {CAG: 0, MAG: 0, RAG: 0}
    *chunks, done = rest
    assert chunks and all(isinstance(c, AnswerChunkEvent) for c in chunks)
    assert "".join(c.text for c in chunks) == "the answer"
    assert isinstance(done, AnswerCompleteEvent)
    assert len(recorder.records) == 1
    recorded_tenant, recorded_user, recorded_session, _, recorded_contributing = recorder.records[0]
    assert (recorded_tenant, recorded_user, recorded_session) == (tenant_id, user_id, session_id)
    assert recorded_contributing == frozenset({RAG})


async def test_stream_rejects_an_over_budget_question_before_any_event():
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(),
        FakeQueryClassifier(_RAG_ONLY),
        LatencyCascade([FakeCascadeTier(RAG)], _GENEROUS),
        FakeChatModel(),
        total_context_tokens=100,
    )

    with pytest.raises(QueryExceedsBudget) as raised:
        await use_case.stream(*_ids(), "word " * 50)

    assert raised.value.query_slice == 10


async def test_a_chat_model_failure_ends_the_stream_with_one_error_event():
    chat_model = FakeChatModel(
        "partial", stream_chunk_size=4, stream_error=RuntimeError("model exploded")
    )
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(),
        FakeQueryClassifier(_RAG_ONLY),
        LatencyCascade([FakeCascadeTier(RAG, _hit(RAG, "doc"))], _GENEROUS),
        chat_model,
    )

    events = [event async for event in await use_case.stream(*_ids(), _QUESTION)]

    assert isinstance(events[0], RoutingStageEvent)
    assert isinstance(events[-1], AnswerErrorEvent)
    assert events[-1].message == "answer generation failed"
    assert not any(isinstance(event, AnswerCompleteEvent) for event in events)


async def test_without_a_classifier_streaming_still_answers_unrouted():
    cag = FakeCascadeTier(CAG, _hit(CAG, "cached"))
    rag = FakeCascadeTier(RAG, _hit(RAG, "fresh"))
    chat_model = FakeChatModel("ok", stream_chunk_size=2)
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(), None, LatencyCascade([cag, rag], _GENEROUS), chat_model
    )

    events = [event async for event in await use_case.stream(*_ids(), _QUESTION)]

    routing = events[0]
    assert isinstance(routing, RoutingStageEvent)
    assert routing.decision is None
```

Add these imports/tests to `tests/unit/test_answer_in_session.py` (`AsyncIterator` isn't needed directly, only used inside the fake):

```python
async def test_streaming_a_question_in_the_callers_own_session_delegates_to_the_answerer():
    repository, answerer = FakeChatSessionRepository(), _RecordingAnswerer()
    answerer.events = ["event-1", "event-2"]
    session_id = await _owned_session(repository)

    stream = await AnswerInSession(repository, answerer).stream(TENANT, OWNER, session_id, "q")
    events = [event async for event in stream]

    assert events == ["event-1", "event-2"]
    assert answerer.stream_calls == [(TENANT, OWNER, session_id, "q")]


@pytest.mark.parametrize("caller", ["another user", "another tenant", "unknown session"])
async def test_streaming_a_session_that_isnt_the_callers_is_refused_before_any_event(caller):
    repository, answerer = FakeChatSessionRepository(), _RecordingAnswerer()
    session_id = await _owned_session(repository)
    tenant_id, user_id = TENANT, OWNER
    if caller == "another user":
        user_id = uuid.uuid4()
    elif caller == "another tenant":
        tenant_id = uuid.uuid4()
    else:
        session_id = uuid.uuid4()

    with pytest.raises(SessionNotFound):
        await AnswerInSession(repository, answerer).stream(tenant_id, user_id, session_id, "q")

    assert answerer.stream_calls == []
```

Replace the existing `_RecordingAnswerer` class in the same file with this extended version (it must keep satisfying both `execute()` and the new `stream()`):

```python
class _RecordingAnswerer:
    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID, str]] = []
        self.stream_calls: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID, str]] = []
        self.answer = object()
        self.events: list[object] = []

    async def execute(self, tenant_id, user_id, session_id, question):
        self.calls.append((tenant_id, user_id, session_id, question))
        return self.answer

    async def stream(self, tenant_id, user_id, session_id, question):
        self.stream_calls.append((tenant_id, user_id, session_id, question))
        return self._events()

    async def _events(self):
        for event in self.events:
            yield event
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_unified_answer_question.py tests/unit/test_answer_in_session.py -v`
Expected: the new tests FAIL with `AttributeError: 'UnifiedAnswerQuestion' object has no attribute 'stream'` and `AttributeError: 'AnswerInSession' object has no attribute 'stream'`; the pre-existing tests in both files still PASS (this task hasn't touched `execute()` yet).

- [ ] **Step 3: Refactor and implement**

Replace the full contents of `src/orchestration/application/unified_answer_question.py` with:

```python
import asyncio
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from src.orchestration.application.assemble_context import AssembledContext, assemble_context
from src.orchestration.application.latency_cascade import LatencyCascade
from src.orchestration.domain.budget_allocator import DEFAULT_SHARES, allocate
from src.orchestration.domain.entities import (
    BudgetAllocation,
    BudgetShares,
    CascadeResult,
    ContextItem,
    Paradigm,
    RoutingDecision,
    TierAttempt,
    TierRequest,
)
from src.orchestration.domain.errors import QueryExceedsBudget
from src.orchestration.domain.paradigm_router import (
    DEFAULT_SELECT_THRESHOLD,
    DEFAULT_UNCERTAINTY_MARGIN,
    decide,
    fallback_decision,
)
from src.orchestration.domain.ports import QueryClassifier, SessionBudgetRecorder
from src.rag.domain.ports import ChatModel, EmbeddingModel
from src.shared.tokenization import count_tokens

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKENS = 128_000
# Seconds. The router comparison (evaluation/reports/orchestration-meta-layer-router.md)
# measured the local qwen3.5 classifier at p50 18.9s and p95 60.0s, against 0.1ms for
# the lexical classifier and 2.4ms for the prototype one. At this default the LLM
# classifier therefore falls back to routing every tier on at least half of all
# queries, going by its median alone. That is deliberate:
# waiting tens of seconds to pick between tiers budgeted at 10ms-2s defeats the
# cascade, so a request-path classifier has to be a millisecond-scale one.
DEFAULT_CLASSIFIER_TIMEOUT = 2.0

RoutingFallback = Literal["classifier_timeout", "classifier_error"]


@dataclass(frozen=True)
class StageTimings:
    embed_ms: float
    route_ms: float
    cascade_ms: float
    assemble_ms: float
    record_ms: float  # 0.0 when no SessionBudgetRecorder is configured
    generate_ms: float


@dataclass(frozen=True)
class UnifiedAnswer:
    answer: str
    sources: list[ContextItem]
    decision: RoutingDecision | None
    # None when the classifier decided the route (or none was configured).
    routing_fallback: RoutingFallback | None
    attempts: list[TierAttempt]
    allocation: BudgetAllocation
    dropped: dict[Paradigm, int]
    degraded: bool
    timings: StageTimings


@dataclass(frozen=True)
class RoutingStageEvent:
    """The router's decision, once made -- the streaming counterpart of
    UnifiedAnswer.decision/routing_fallback."""

    decision: RoutingDecision | None
    routing_fallback: RoutingFallback | None


@dataclass(frozen=True)
class RetrievalStageEvent:
    """The cascade's result, once every attempted tier has answered or given up --
    the streaming counterpart of UnifiedAnswer.attempts/degraded."""

    attempts: list[TierAttempt]
    degraded: bool


@dataclass(frozen=True)
class BudgetStageEvent:
    """Context assembly's result -- the streaming counterpart of
    UnifiedAnswer.sources/dropped. Never carries the raw BudgetAllocation numbers,
    for the same reason answer_response() withholds them from the JSON endpoint
    (spec decision 8)."""

    sources: list[ContextItem]
    dropped: dict[Paradigm, int]


@dataclass(frozen=True)
class AnswerChunkEvent:
    """One text delta from the chat model's own stream()."""

    text: str


@dataclass(frozen=True)
class AnswerCompleteEvent:
    """The stream's normal terminal event; carries no data."""


@dataclass(frozen=True)
class AnswerErrorEvent:
    """The stream's terminal event on failure. `message` is always a generic,
    safe-to-show string -- the real exception is logged, never surfaced here."""

    message: str


UnifiedAnswerEvent = (
    RoutingStageEvent
    | RetrievalStageEvent
    | BudgetStageEvent
    | AnswerChunkEvent
    | AnswerCompleteEvent
    | AnswerErrorEvent
)


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000


class UnifiedAnswerQuestion:
    """Section 3.4 Pattern 1, "The Smart Router": route, cascade, budget, answer.

    With classifier=None the cascade runs unrouted, exactly as Concept 5
    draws it -- the ablation baseline the router is measured against. A
    classifier that times out or raises doesn't fail the request: the query
    is routed with fallback_decision(), every tier in parallel.

    The budget is recorded after the cascade and before generation: a
    session that doesn't exist, or isn't this user's, fails the request
    before an answer is paid for, though the retrieval work is already spent.

    stream() answers the same question as execute(), sharing the same
    _decide_route/_do_budget/_record_budget helpers, but reports progress as
    domain events instead of returning one final UnifiedAnswer -- see
    docs/superpowers/specs/2026-09-17-streaming-answers-design.md for why its
    query-budget check has to stay eager (in stream() itself, not in the lazy
    generator stream() hands back) and why a failure after streaming has
    already started becomes an AnswerErrorEvent rather than a raised exception.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        classifier: QueryClassifier | None,
        cascade: LatencyCascade,
        chat_model: ChatModel,
        *,
        total_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        shares: BudgetShares = DEFAULT_SHARES,
        select_threshold: float = DEFAULT_SELECT_THRESHOLD,
        uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
        classifier_timeout: float = DEFAULT_CLASSIFIER_TIMEOUT,
        budget_recorder: SessionBudgetRecorder | None = None,
    ) -> None:
        if total_context_tokens <= 0:
            raise ValueError("total_context_tokens must be positive")
        if classifier_timeout <= 0.0:
            raise ValueError("classifier_timeout must be positive")
        self._embedder = embedding_model
        self._classifier = classifier
        self._cascade = cascade
        self._chat_model = chat_model
        self._total = total_context_tokens
        self._shares = shares
        self._select_threshold = select_threshold
        self._uncertainty_margin = uncertainty_margin
        self._classifier_timeout = classifier_timeout
        self._budget_recorder = budget_recorder

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> UnifiedAnswer:
        # Same floor allocate() applies, so this check and the allocation
        # below can never disagree about the Query slice's size.
        query_slice = math.floor(self._total * self._shares.query)
        query_tokens = count_tokens(question)
        if query_tokens > query_slice:
            raise QueryExceedsBudget(query_tokens, query_slice)

        started = time.perf_counter()
        # Off the event loop: milliseconds of CPU that would otherwise stall
        # every tier and every other request sharing the loop.
        embedding = await asyncio.to_thread(self._embedder.embed, question)
        embedded = time.perf_counter()

        decision, routing_fallback = await self._decide_route(question, embedding)
        routed = time.perf_counter()

        request = TierRequest(tenant_id, user_id, session_id, question, embedding)
        cascade_result = await self._cascade.run(request, decision)
        cascaded = time.perf_counter()

        allocation, assembled = self._do_budget(cascade_result)
        assembled_at = time.perf_counter()

        await self._record_budget(
            tenant_id, user_id, session_id, allocation, cascade_result.contributing
        )
        recorded = time.perf_counter()

        answer = await self._chat_model.generate(question=question, context=assembled.text)
        generated = time.perf_counter()

        return UnifiedAnswer(
            answer=answer,
            sources=assembled.included,
            decision=decision,
            routing_fallback=routing_fallback,
            attempts=cascade_result.attempts,
            allocation=allocation,
            dropped=assembled.dropped,
            degraded=cascade_result.degraded,
            timings=StageTimings(
                embed_ms=_ms(started, embedded),
                route_ms=_ms(embedded, routed),
                cascade_ms=_ms(routed, cascaded),
                assemble_ms=_ms(cascaded, assembled_at),
                record_ms=_ms(assembled_at, recorded) if self._budget_recorder else 0.0,
                generate_ms=_ms(recorded, generated),
            ),
        )

    async def stream(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> AsyncIterator[UnifiedAnswerEvent]:
        query_slice = math.floor(self._total * self._shares.query)
        query_tokens = count_tokens(question)
        if query_tokens > query_slice:
            raise QueryExceedsBudget(query_tokens, query_slice)
        return self._stream_events(tenant_id, user_id, session_id, question)

    async def _stream_events(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> AsyncIterator[UnifiedAnswerEvent]:
        try:
            embedding = await asyncio.to_thread(self._embedder.embed, question)
            decision, routing_fallback = await self._decide_route(question, embedding)
            yield RoutingStageEvent(decision, routing_fallback)

            request = TierRequest(tenant_id, user_id, session_id, question, embedding)
            cascade_result = await self._cascade.run(request, decision)
            yield RetrievalStageEvent(cascade_result.attempts, cascade_result.degraded)

            allocation, assembled = self._do_budget(cascade_result)
            yield BudgetStageEvent(assembled.included, assembled.dropped)

            await self._record_budget(
                tenant_id, user_id, session_id, allocation, cascade_result.contributing
            )

            async for delta in self._chat_model.stream(question=question, context=assembled.text):
                yield AnswerChunkEvent(delta)
            yield AnswerCompleteEvent()
        except Exception:
            logger.exception("answer streaming failed")
            yield AnswerErrorEvent("answer generation failed")

    async def _decide_route(
        self, question: str, embedding: list[float]
    ) -> tuple[RoutingDecision | None, RoutingFallback | None]:
        if self._classifier is None:
            return None, None
        return await self._route(self._classifier, question, embedding)

    def _do_budget(
        self, cascade_result: CascadeResult
    ) -> tuple[BudgetAllocation, AssembledContext]:
        allocation = allocate(self._total, cascade_result.contributing, self._shares)
        return allocation, assemble_context(cascade_result.items, allocation)

    async def _record_budget(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        allocation: BudgetAllocation,
        contributing: frozenset[Paradigm],
    ) -> None:
        if self._budget_recorder is not None:
            await self._budget_recorder.record(
                tenant_id, user_id, session_id, allocation, contributing
            )

    async def _route(
        self, classifier: QueryClassifier, question: str, embedding: list[float]
    ) -> tuple[RoutingDecision, RoutingFallback | None]:
        try:
            scores = await asyncio.wait_for(
                classifier.score(question, embedding), self._classifier_timeout
            )
            return decide(scores, self._select_threshold, self._uncertainty_margin), None
        except TimeoutError:
            # The classifier_timeout limit, or a timeout the classifier raised itself.
            logger.warning(
                "query classifier timed out (limit %.2fs); routing every tier in parallel",
                self._classifier_timeout,
            )
            return fallback_decision(), "classifier_timeout"
        except Exception:
            logger.warning("query classifier failed; routing every tier in parallel", exc_info=True)
            return fallback_decision(), "classifier_error"
```

Replace the full contents of `src/orchestration/application/answer_in_session.py` with:

```python
import uuid
from collections.abc import AsyncIterator
from typing import Protocol

from src.identity.domain.ports import ChatSessionRepository
from src.orchestration.application.unified_answer_question import UnifiedAnswer, UnifiedAnswerEvent
from src.orchestration.domain.errors import SessionNotFound


class SessionQuestionAnswerer(Protocol):
    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> UnifiedAnswer: ...

    async def stream(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> AsyncIterator[UnifiedAnswerEvent]: ...


class AnswerInSession:
    """Answers a question in one of the caller's own chat sessions, either as one
    final UnifiedAnswer (execute()) or as a stream of UnifiedAnswerEvents (stream()).

    Both methods look the session up as the caller's before anything else happens.
    A session that doesn't exist, belongs to another user, or lives in another
    tenant raises the same SessionNotFound, so nothing is embedded, retrieved, or
    recorded for it, and the caller can't tell which of the three it was.
    PostgresSessionBudgetRecorder's own ownership check stays as a second line.

    stream() is a plain coroutine (no `yield` in its own body), so this ownership
    check runs the moment it's awaited -- not on the caller's first iteration of
    whatever it returns. That is what keeps a cross-tenant or cross-user stream
    request a plain SessionNotFound (a 404 with zero bytes streamed, at the router)
    instead of a response that already opened as a stream before being denied.
    """

    def __init__(
        self, sessions: ChatSessionRepository, answerer: SessionQuestionAnswerer
    ) -> None:
        self._sessions = sessions
        self._answerer = answerer

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> UnifiedAnswer:
        if await self._sessions.find_owned(tenant_id, user_id, session_id) is None:
            raise SessionNotFound(session_id)
        return await self._answerer.execute(tenant_id, user_id, session_id, question)

    async def stream(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> AsyncIterator[UnifiedAnswerEvent]:
        if await self._sessions.find_owned(tenant_id, user_id, session_id) is None:
            raise SessionNotFound(session_id)
        return await self._answerer.stream(tenant_id, user_id, session_id, question)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_unified_answer_question.py tests/unit/test_answer_in_session.py -v`
Expected: PASS, all of them — the new streaming tests and every pre-existing `execute()` test in both files, unmodified and still green.

Run: `uv run mypy src` and `uv run ruff check src tests`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/application/unified_answer_question.py \
  src/orchestration/application/answer_in_session.py \
  tests/unit/test_unified_answer_question.py tests/unit/test_answer_in_session.py
git commit -m "feat(#195): add UnifiedAnswerQuestion.stream() and AnswerInSession.stream()"
```

---

### Task 3: SSE encoding and the router endpoint

**Files:**
- Create: `src/api/sse.py`
- Modify: `src/api/routers/sessions.py`
- Test: `tests/unit/test_sse.py`

**Interfaces:**
- Consumes: `UnifiedAnswerEvent` and its six variants from Task 2; `AnswerInSession.stream()` from Task 2; `AnswerRequest`, `enforce_rate_limit`, `chat_rate_limit`, `GLOBAL_CHAT_KEY`, `global_chat_limit`, `GLOBAL_CHAT_WINDOW_SECONDS`, `get_caller`, `get_answer_in_session`, `get_rate_limiter` — all already imported at the top of `src/api/routers/sessions.py`.
- Produces: `encode_sse(events: AsyncIterator[UnifiedAnswerEvent]) -> AsyncIterator[bytes]`; the router gains `POST /sessions/{session_id}/answers/stream`. Task 4's integration tests call this endpoint directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sse.py
import json

from src.api.sse import encode_sse
from src.orchestration.application.unified_answer_question import (
    AnswerChunkEvent,
    AnswerCompleteEvent,
    AnswerErrorEvent,
    BudgetStageEvent,
    RetrievalStageEvent,
    RoutingStageEvent,
)
from src.orchestration.domain.entities import (
    ContextItem,
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierAttempt,
    TierOutcome,
)

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


async def _aiter(items):
    for item in items:
        yield item


async def _frames(events):
    return [frame async for frame in encode_sse(_aiter(events))]


async def test_a_routing_event_with_a_decision_encodes_paradigms_mode_and_fallback():
    decision = RoutingDecision(frozenset({RAG}), RoutingMode.CASCADE, {RAG: 0.9})

    [frame] = await _frames([RoutingStageEvent(decision, None)])

    assert frame == (
        b"event: routing\n"
        b'data: {"paradigms": ["rag"], "mode": "cascade", "fallback": null}\n\n'
    )


async def test_a_routing_event_with_no_decision_encodes_empty_paradigms_and_null_mode():
    [frame] = await _frames([RoutingStageEvent(None, "classifier_timeout")])

    assert frame == (
        b"event: routing\n"
        b'data: {"paradigms": [], "mode": null, "fallback": "classifier_timeout"}\n\n'
    )


async def test_a_retrieval_event_encodes_attempts_and_degraded():
    attempt = TierAttempt(RAG, TierOutcome.HIT, 12.5)

    [frame] = await _frames([RetrievalStageEvent([attempt], True)])

    assert frame == (
        b"event: retrieval\n"
        b'data: {"attempts": [{"paradigm": "rag", "outcome": "hit", "elapsed_ms": 12.5}],'
        b' "degraded": true}\n\n'
    )


async def test_a_budget_event_encodes_sources_and_dropped_counts():
    item = ContextItem(RAG, "some text", 0.8, None)

    [frame] = await _frames([BudgetStageEvent([item], {CAG: 0, MAG: 0, RAG: 2})])

    payload = json.loads(frame.decode().split("data: ", 1)[1])
    assert payload["sources"] == [{"paradigm": "rag", "content": "some text", "source_id": None}]
    assert payload["dropped"] == {"cag": 0, "mag": 0, "rag": 2}


async def test_a_chunk_event_encodes_its_text():
    [frame] = await _frames([AnswerChunkEvent("hello")])

    assert frame == b'event: chunk\ndata: {"text": "hello"}\n\n'


async def test_a_chunk_event_with_an_embedded_newline_round_trips_through_json():
    [frame] = await _frames([AnswerChunkEvent("line one\nline two")])

    text_line, data_line = frame.split(b"\n", 1)
    assert text_line == b"event: chunk"
    # The frame carries exactly two real newlines -- the one ending the data
    # line and the blank line terminating the SSE event -- because json.dumps
    # escaped the chunk's own newline as the two characters backslash-n, never
    # a real line break, so it can't introduce a third.
    assert frame.count(b"\n") == 2
    payload = json.loads(data_line.decode().removeprefix("data: ").rstrip("\n"))
    assert payload["text"] == "line one\nline two"


async def test_a_complete_event_encodes_as_done_with_no_payload():
    [frame] = await _frames([AnswerCompleteEvent()])

    assert frame == b"event: done\ndata: {}\n\n"


async def test_an_error_event_encodes_its_message_as_detail():
    [frame] = await _frames([AnswerErrorEvent("answer generation failed")])

    assert frame == b'event: error\ndata: {"detail": "answer generation failed"}\n\n'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.sse'`

- [ ] **Step 3: Write the implementation**

```python
# src/api/sse.py
import json
from collections.abc import AsyncIterator

from src.orchestration.application.unified_answer_question import (
    AnswerChunkEvent,
    AnswerCompleteEvent,
    AnswerErrorEvent,
    BudgetStageEvent,
    RetrievalStageEvent,
    RoutingStageEvent,
    UnifiedAnswerEvent,
)
from src.orchestration.domain.entities import PARADIGM_ORDER


def _wire(event: UnifiedAnswerEvent) -> tuple[str, dict]:
    """Maps one domain event to its SSE event name and JSON-serializable payload.
    Never includes router scores, similarity scores, or stage timings -- the same
    restriction answer_response() already applies to the JSON endpoint."""
    if isinstance(event, RoutingStageEvent):
        decision = event.decision
        return "routing", {
            "paradigms": (
                []
                if decision is None
                else [p.value for p in PARADIGM_ORDER if p in decision.paradigms]
            ),
            "mode": None if decision is None else decision.mode.value,
            "fallback": event.routing_fallback,
        }
    if isinstance(event, RetrievalStageEvent):
        return "retrieval", {
            "attempts": [
                {
                    "paradigm": attempt.paradigm.value,
                    "outcome": attempt.outcome.value,
                    "elapsed_ms": attempt.elapsed_ms,
                }
                for attempt in event.attempts
            ],
            "degraded": event.degraded,
        }
    if isinstance(event, BudgetStageEvent):
        return "budget", {
            "sources": [
                {
                    "paradigm": source.paradigm.value,
                    "content": source.content,
                    "source_id": str(source.source_id) if source.source_id else None,
                }
                for source in event.sources
            ],
            "dropped": {paradigm.value: count for paradigm, count in event.dropped.items()},
        }
    if isinstance(event, AnswerChunkEvent):
        return "chunk", {"text": event.text}
    if isinstance(event, AnswerCompleteEvent):
        return "done", {}
    assert isinstance(event, AnswerErrorEvent)
    return "error", {"detail": event.message}


async def encode_sse(events: AsyncIterator[UnifiedAnswerEvent]) -> AsyncIterator[bytes]:
    async for event in events:
        name, payload = _wire(event)
        yield f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
```

Add the streaming route to `src/api/routers/sessions.py`. First, extend its existing imports:

```python
# src/api/routers/sessions.py -- add to the existing `from fastapi import ...` line
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse

# add a new import line
from src.api.sse import encode_sse
```

Then add the new route function at the end of the file, after `answer()`:

```python
@router.post("/{session_id}/answers/stream")
async def answer_stream(
    session_id: uuid.UUID,
    payload: AnswerRequest,
    request: Request,
    response: Response,
    caller: Caller = Depends(get_caller),
    answer_in_session: AnswerInSession = Depends(get_answer_in_session),
) -> StreamingResponse:
    # Same two checks, same order, same keys and limits as answer() above --
    # this is the same resource's budget, not a second one.
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=f"chat:{caller.user_id}",
        limit=chat_rate_limit(),
    )
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=GLOBAL_CHAT_KEY,
        limit=global_chat_limit(),
        window_seconds=GLOBAL_CHAT_WINDOW_SECONDS,
    )
    # Raises SessionNotFound/QueryExceedsBudget HERE, before StreamingResponse
    # is ever constructed -- see AnswerInSession.stream()'s docstring for why
    # that's what keeps a denial a plain 404/422 instead of a stream that
    # opens and then errors.
    events = await answer_in_session.stream(
        caller.tenant_id, caller.user_id, session_id, payload.question
    )
    return StreamingResponse(encode_sse(events), media_type="text/event-stream")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sse.py -v`
Expected: PASS

Run: `uv run mypy src` and `uv run ruff check src tests`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/api/sse.py src/api/routers/sessions.py tests/unit/test_sse.py
git commit -m "feat(#195): add SSE encoding and POST /sessions/{session_id}/answers/stream"
```

---

### Task 4: Integration tests against the real endpoint

**Files:**
- Create: `tests/integration/test_sessions_stream_endpoint.py`

**Interfaces:**
- Consumes: `build_unified_pipeline` (`src/api/unified_pipeline.py`), `ContextEchoChatModel`/`VALID_HASH` (`tests/integration/orchestration_env.py`, both updated in Task 1), the app's real dependency wiring — all identical to how `tests/integration/test_sessions_endpoints.py` already exercises the JSON endpoint.

This task adds no new production code; it proves Tasks 1-3 hold end to end against real Postgres, Qdrant, and Redis, mirroring `tests/integration/test_sessions_endpoints.py`'s fixtures exactly so the two files stay easy to compare side by side. There is deliberately no integration-level test for `QueryExceedsBudget`: `AnswerRequest.question` is capped at 4,000 characters by its own Pydantic schema, which in practice never reaches the default 12,800-token Query slice, so neither this endpoint nor the JSON one has (or can usefully have) an HTTP-level test for it — `test_stream_rejects_an_over_budget_question_before_any_event` in Task 2 already proves the eager check itself, with a small `total_context_tokens` override.

- [ ] **Step 1: Write the test file**

```python
# tests/integration/test_sessions_stream_endpoint.py
import json
import os
import uuid
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from src.identity.infrastructure.jwt_token_issuer import JWTTokenIssuer
from src.rag.domain.entities import Chunk
from tests.integration.orchestration_env import VALID_HASH, ContextEchoChatModel

pytestmark = pytest.mark.asyncio(loop_scope="module")

_SECRET = "test-secret-key"
_POLICY = "Our return policy allows returns of unopened items within forty-five days."
_NOT_FOUND = {"detail": "Session not found"}


@pytest.fixture(autouse=True)
def _default_chat_rate_limit():
    previous = os.environ.pop("CHAT_RATE_LIMIT_PER_MINUTE", None)
    yield
    os.environ.pop("CHAT_RATE_LIMIT_PER_MINUTE", None)
    if previous is not None:
        os.environ["CHAT_RATE_LIMIT_PER_MINUTE"] = previous


@pytest.fixture(autouse=True)
def _default_global_chat_rate_limit():
    previous = os.environ.pop("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", None)
    yield
    os.environ.pop("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", None)
    if previous is not None:
        os.environ["CHAT_RATE_LIMIT_GLOBAL_PER_HOUR"] = previous


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    import sys

    main = sys.modules.get("src.api.main")
    if main is not None:
        main.app.dependency_overrides.clear()


async def _client(app_database_url, redis_url, qdrant_url, embedding_model):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["JWT_SECRET_KEY"] = _SECRET
    from src.api import dependencies
    from src.api.main import app
    from src.api.unified_pipeline import build_unified_pipeline

    await dependencies.get_vector_store().ensure_collection()
    pipeline = build_unified_pipeline(
        sessionmaker=dependencies._sessionmaker,
        sessions=dependencies.get_chat_session_repository(),
        embedding_model=embedding_model,
        vector_store=dependencies.get_vector_store(),
        chat_model=ContextEchoChatModel(),
    )
    app.dependency_overrides[dependencies.get_answer_in_session] = (
        lambda: pipeline.answer_in_session
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    now, user_id = datetime.now(UTC), uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id, "created_at": now, "updated_at": now,
        },
    )
    await db_session.commit()
    return user_id


def _auth(user_id: uuid.UUID, tenant_id: uuid.UUID) -> dict[str, str]:
    token = JWTTokenIssuer(secret_key=_SECRET).issue_pair(user_id, tenant_id).access_token.value
    return {"Authorization": f"Bearer {token}"}


async def _seed_policy(tenant_id: uuid.UUID, embedding_model) -> None:
    from src.api.dependencies import get_vector_store

    chunk = Chunk(uuid.uuid4(), uuid.uuid4(), _POLICY, embedding_model.embed(_POLICY))
    await get_vector_store().upsert(chunk, tenant_id)


def _parse_sse(raw: str) -> list[tuple[str | None, dict]]:
    events: list[tuple[str | None, dict]] = []
    for block in raw.strip("\n").split("\n\n"):
        if not block:
            continue
        name: str | None = None
        data: dict = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        events.append((name, data))
    return events


async def _post_stream(client, url, *, json_body, headers):
    async with client.stream("POST", url, json=json_body, headers=headers) as response:
        await response.aread()
    return response


async def test_a_streamed_answer_reassembles_the_same_stages_and_text_the_json_endpoint_would_give(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    user_id = await _user(db_session, tenant_id)
    await _seed_policy(tenant_id, embedding_model)
    headers = _auth(user_id, tenant_id)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        response = await _post_stream(
            client,
            f"/sessions/{session_id}/answers/stream",
            json_body={"question": "What is the return policy for unopened items?"},
            headers=headers,
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "routing"
    assert names[1] == "retrieval"
    assert names[2] == "budget"
    assert names[-1] == "done"
    assert names[3:-1] and all(name == "chunk" for name in names[3:-1])
    full_answer = "".join(data["text"] for name, data in events if name == "chunk")
    assert "forty-five days" in full_answer
    budget_sources = next(data for name, data in events if name == "budget")["sources"]
    assert any(s["paradigm"] == "rag" and "forty-five" in s["content"] for s in budget_sources)


async def test_streaming_another_users_session_in_the_same_tenant_returns_a_plain_404_with_no_stream(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    owner, intruder = await _user(db_session, tenant_id), await _user(db_session, tenant_id)
    await _seed_policy(tenant_id, embedding_model)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (
            await client.post("/sessions", json={}, headers=_auth(owner, tenant_id))
        ).json()["id"]
        response = await _post_stream(
            client,
            f"/sessions/{session_id}/answers/stream",
            json_body={"question": "What is the return policy?"},
            headers=_auth(intruder, tenant_id),
        )

    assert (response.status_code, json.loads(response.text)) == (404, _NOT_FOUND)
    assert "text/event-stream" not in response.headers.get("content-type", "")


async def test_streaming_another_tenants_session_returns_a_plain_404_with_no_stream(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    owner_tenant, intruder_tenant = uuid.uuid4(), uuid.uuid4()
    owner = await _user(db_session, owner_tenant)
    intruder = await _user(db_session, intruder_tenant)
    await _seed_policy(owner_tenant, embedding_model)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (
            await client.post("/sessions", json={}, headers=_auth(owner, owner_tenant))
        ).json()["id"]
        response = await _post_stream(
            client,
            f"/sessions/{session_id}/answers/stream",
            json_body={"question": "What is the return policy?"},
            headers=_auth(intruder, intruder_tenant),
        )

    assert (response.status_code, json.loads(response.text)) == (404, _NOT_FOUND)
    assert "text/event-stream" not in response.headers.get("content-type", "")


async def test_a_rate_limited_caller_gets_429_not_a_stream(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    os.environ["CHAT_RATE_LIMIT_PER_MINUTE"] = "2"
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    await _seed_policy(tenant_id, embedding_model)
    question = {"question": "What is the return policy?"}
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        responses = [
            await _post_stream(
                client,
                f"/sessions/{session_id}/answers/stream",
                json_body=question,
                headers=headers,
            )
            for _ in range(3)
        ]

    assert [r.status_code for r in responses] == [200, 200, 429]
    assert "text/event-stream" not in responses[2].headers.get("content-type", "")


async def test_every_stream_route_refuses_a_request_without_a_token(
    app_database_url, redis_url, qdrant_url, embedding_model
):
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        response = await _post_stream(
            client,
            f"/sessions/{uuid.uuid4()}/answers/stream",
            json_body={"question": "q"},
            headers={},
        )

    assert response.status_code == 401
    assert "text/event-stream" not in response.headers.get("content-type", "")
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/integration/test_sessions_stream_endpoint.py -v`
Expected: PASS, all five tests, against real TestContainers-provisioned Postgres, Qdrant, and Redis.

- [ ] **Step 3: Run the full test suite to check for regressions**

Run: `uv run pytest tests/unit -q`
Expected: PASS, same count as before this plan plus every test this plan added.

Run: `uv run pytest tests/integration -q`
Expected: PASS (plus the project's already-established GPU-only and no-local-Ollama skips, unaffected by this work).

Run: `uv run mypy src` and `uv run ruff check src tests`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_sessions_stream_endpoint.py
git commit -m "test(#195): add end-to-end integration tests for the streaming answers endpoint"
```

---

### Task 5: Documentation pass

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/architecture/OVERVIEW.md`
- Modify: `docs/architecture/CONTEXT_GRAPH.md`
- Modify: `docs/security/SECURITY.md`

**Interfaces:** None — this task changes no code.

Two of these edits need the real test counts observed after Task 4, not the numbers below (which are this plan's placeholders for "wherever the suite actually lands" — replace them with the actual `uv run pytest tests/unit -q` and `uv run pytest tests/integration -q` totals when you get here, the same way every prior batch in this project recorded real numbers rather than projected ones).

- [ ] **Step 1: Update `CLAUDE.md`**

In the opening paragraph (currently line 3), find:

```
Freshness-routed ingestion is now served over HTTP too: `POST /data-sources` and `GET /data-sources/jobs/{task_id}` run on this project's first background-worker subsystem, Celery against a Redis broker and result backend, under `src/workers/`. What's still ahead is real too, and named plainly rather than left implicit: the frontend and Kubernetes deployment layers.
```

Replace with:

```
Freshness-routed ingestion is now served over HTTP too: `POST /data-sources` and `GET /data-sources/jobs/{task_id}` run on this project's first background-worker subsystem, Celery against a Redis broker and result backend, under `src/workers/`. The per-query path can now stream, too: `POST /sessions/{session_id}/answers/stream` reports routing, retrieval, and budget progress as Server-Sent Events before streaming the generated answer token-by-token, sharing the same object-level authorization and the same abuse controls as the JSON endpoint it sits beside. What's still ahead is real too, and named plainly rather than left implicit: the frontend and Kubernetes deployment layers.
```

Also in that same opening paragraph, find the test-count clause (`1,204 unit tests and 301 integration tests`) and replace both numbers with the real totals from Task 4's Step 3 run.

Further down, in the context-graph paragraph (currently around line 45), find:

```
The freshness-routed ingestion endpoint and its `src/workers/` background-worker subsystem moved from dashed to solid by hand in the same pass that built them, without opening a seventh generation; only the frontend and the Kubernetes/production-deployment layer remain the dashed, genuinely-unbuilt nodes the diagram's own "What isn't here yet" section names directly.
```

Replace with:

```
The freshness-routed ingestion endpoint and its `src/workers/` background-worker subsystem, and later the streaming variant of the per-query answer endpoint, each moved from dashed to solid by hand in the pass that built them, without opening a seventh generation; only the frontend and the Kubernetes/production-deployment layer remain the dashed, genuinely-unbuilt nodes the diagram's own "What isn't here yet" section names directly.
```

- [ ] **Step 2: Update `docs/architecture/OVERVIEW.md`**

In the per-query HTTP-exposure section, find:

```
Clients reach the whole per-query path over HTTP through chat sessions (`src/api/routers/sessions.py`):

- `POST /sessions` creates a session for the caller.
- `GET /sessions` lists the caller's own sessions.
- `POST /sessions/{session_id}/answers` answers through `AnswerInSession`, which checks that the session belongs to the token's user before anything is embedded or retrieved. A session that's missing, another user's, or another tenant's gets the same 404.

The API composes the cascade with MAG and RAG tiers only (`src/api/unified_pipeline.py`). A CAG tier needs a warmed frozen cache in the serving process and a worker to warm it, and neither exists yet, so a query the router sends to CAG alone is answered from RAG. Answering shares a per-user rate limit with the RAG-only `POST /chat`. The batch's live run against uvicorn and Ollama, and its security review, are recorded in `docs/superpowers/plans/2026-09-13-unified-api-sessions.md`.
```

Replace with:

```
Clients reach the whole per-query path over HTTP through chat sessions (`src/api/routers/sessions.py`):

- `POST /sessions` creates a session for the caller.
- `GET /sessions` lists the caller's own sessions.
- `POST /sessions/{session_id}/answers` answers through `AnswerInSession`, which checks that the session belongs to the token's user before anything is embedded or retrieved. A session that's missing, another user's, or another tenant's gets the same 404.
- `POST /sessions/{session_id}/answers/stream` answers the same question through the same `AnswerInSession`, but as Server-Sent Events: a `routing` event once the router decides, a `retrieval` event once the cascade returns, a `budget` event once context is assembled, then the generated answer as `chunk` events, ending in `done` (or `error` if generation itself fails after the stream has already opened). `UnifiedAnswerQuestion.stream()` and `AnswerInSession.stream()` share the exact ownership check `execute()` uses, run eagerly before `StreamingResponse` is ever constructed, so a session that isn't the caller's gets the identical 404 with zero bytes streamed rather than a stream that opens and then fails.

The API composes the cascade with MAG and RAG tiers only (`src/api/unified_pipeline.py`). A CAG tier needs a warmed frozen cache in the serving process and a worker to warm it, and neither exists yet, so a query the router sends to CAG alone is answered from RAG. Answering shares a per-user rate limit with the RAG-only `POST /chat`, and the streaming endpoint shares that same budget too rather than getting a second one. The batch's live run against uvicorn and Ollama, and its security review, are recorded in `docs/superpowers/plans/2026-09-13-unified-api-sessions.md`; the streaming variant's own design and review are in `docs/superpowers/specs/2026-09-17-streaming-answers-design.md`.
```

Further down, in the Phase 1 roadmap section, find:

```
The unified per-query path is now served over HTTP through the `sessions` router. What genuinely remains unbuilt, stated plainly rather than left to this paragraph's own out-of-date framing: a scheduler to drive the router's refresh and review, and everything past Phase 1 — the frontend, and the Kubernetes/production-Docker layer.
```

Replace with:

```
The unified per-query path is now served over HTTP through the `sessions` router, with a streaming variant (`POST /sessions/{session_id}/answers/stream`, Server-Sent Events) alongside the original JSON one. What genuinely remains unbuilt, stated plainly rather than left to this paragraph's own out-of-date framing: a scheduler to drive the router's refresh and review, and everything past Phase 1 — the frontend, and the Kubernetes/production-Docker layer.
```

- [ ] **Step 3: Update `docs/architecture/CONTEXT_GRAPH.md`**

In the "What isn't here yet" section, find the sentence ending `...moved into the new \`REAL_INGESTION\` subgraph above.` (immediately followed by `That leaves two things genuinely unbuilt, down from three: ...`), and insert this new sentence between them:

```
The per-query path gained a second HTTP shape next, without a new subgraph: `POST /sessions/{session_id}/answers/stream` streams `UnifiedAnswerQuestion`'s own progress and generated answer as Server-Sent Events, reusing the same `AnswerInSession` ownership check and the same `MOD_ORCH_APP`/`MOD_API` nodes `REAL_ORCH` already draws for the JSON endpoint.
```

In the "Keeping this current" section, at the end of the long history paragraph (the one ending `...the ingestion endpoint and worker layers this sentence used to name are \`REAL_INGESTION\` now.`), append this sentence to the same paragraph:

```
The streaming variant of the per-query endpoint updated this diagram by hand next, the same way: no new node or subgraph was needed, since `POST /sessions/{session_id}/answers/stream` reuses `REAL_ORCH`'s existing `MOD_ORCH_APP`, `MOD_ORCH_DOMAIN`, and `MOD_API` nodes rather than adding a class this diagram's per-layer counts would need to recount.
```

- [ ] **Step 4: Update `docs/security/SECURITY.md`**

In the object-level-authorization section, find (the sentence ending the session-routes paragraph, immediately before the ingestion paragraph):

```
`tests/integration/test_sessions_endpoints.py` verifies this end to end against real Postgres, Redis, and Qdrant. It attempts another user's session and another tenant's, and asserts the identical 404 and an untouched budget. `test_postgres_chat_session_repository.py` verifies it at the repository level.

`GET /data-sources/jobs/{task_id}` guards the same object-level-authorization risk with a different mechanism, ...
```

Insert a new paragraph between them:

```
`POST /sessions/{session_id}/answers/stream` carries the identical guarantee by construction rather than by a second implementation: it calls the same `AnswerInSession` ownership check the JSON endpoint uses, and that check runs before the response ever becomes a stream — `StreamingResponse` is only constructed after `await answer_in_session.stream(...)` returns successfully, so `SessionNotFound` still produces the plain `404 {"detail": "Session not found"}` JSON response, never a stream that opens and then closes. A failure that happens after the stream has already started (a chat-model error mid-generation, whose HTTP status is by then locked in at `200`) is reported as one SSE `error` event carrying a generic message, never the underlying exception text, with the real exception logged in full server-side instead. `tests/integration/test_sessions_stream_endpoint.py` verifies the cross-tenant and cross-user cases end to end, asserting zero SSE bytes and a `content-type` that never becomes `text/event-stream` for either.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/architecture/OVERVIEW.md docs/architecture/CONTEXT_GRAPH.md docs/security/SECURITY.md
git commit -m "docs(#195): describe the streaming answers endpoint as built"
```
