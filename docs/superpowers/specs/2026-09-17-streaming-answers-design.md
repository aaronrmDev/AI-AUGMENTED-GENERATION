# Streaming answers over the unified session endpoint — Design

## Context

Epic #182 ("Unified API") names three deliverables: a session-scoped unified
answer endpoint, freshness-routed ingestion, and streaming. The first two are
built and merged (`POST /sessions`, `GET /sessions`, `POST
/sessions/{session_id}/answers` from Story #183; `POST /data-sources`, `GET
/data-sources/jobs/{task_id}` from Story #194, released as v1.3.0). Streaming
is the last item, and unlike the other two it has never been scoped — no
Story issue exists for it yet.

`POST /sessions/{session_id}/answers`
([src/api/routers/sessions.py:67-101](../../../src/api/routers/sessions.py))
answers a question by calling `AnswerInSession.execute()`
([src/orchestration/application/answer_in_session.py](../../../src/orchestration/application/answer_in_session.py)),
which checks session ownership and then delegates to
`UnifiedAnswerQuestion.execute()`
([src/orchestration/application/unified_answer_question.py](../../../src/orchestration/application/unified_answer_question.py)).
That method runs five stages in sequence — embed, route, cascade, allocate
budget and assemble context, generate — and returns one `UnifiedAnswer` only
once every stage, including the chat-model call, has finished. A caller
therefore waits for the full round trip (embedding, tier fan-out, and however
long the configured chat model takes to produce its whole answer) before
seeing a single byte back.

Epic #182 explicitly excludes the frontend from its own scope, and no
frontend exists anywhere in this repository yet (confirmed and scrubbed from
this project's own docs in the session that immediately preceded this one).
Streaming therefore has no real client to design a consumption contract
against. This spec treats that as a backend-transport deliverable, provable
the same way the ingestion endpoint was proven without a frontend: real
integration tests driving the real HTTP wire format with an `httpx`
streaming client, against real infrastructure.

## Goals

- A new `POST /sessions/{session_id}/answers/stream` endpoint, alongside the
  existing JSON one (which is untouched), returning a `text/event-stream`
  Server-Sent Events response.
- The response reports progress as each pipeline stage completes — routing,
  retrieval, budget/context assembly — and then streams the generated answer
  token-by-token, ending with an explicit terminal event.
- Object-level authorization parity with the JSON endpoint and with
  Epic #182's own success criterion: no caller can open a stream for another
  user's or another tenant's session, and a denial must look exactly like
  the JSON endpoint's 404 — same status, same body, zero bytes streamed —
  not a stream that opens and then errors.
- The same abuse controls that already gate the JSON endpoint (the per-user
  chat quota, the global chat quota, the 16 KiB default body limit) gate this
  one too, because it answers the same resource.
- A security review against `docs/security/SECURITY.md` before merge, per
  the Epic's Definition of Done.

## Non-goals

- **No frontend.** Nothing in this repository consumes the stream; proof is
  by integration test against the raw wire format.
- **No new rate-limit bucket.** This is the same `chat:{user_id}` and
  `global:chat` budget the JSON endpoint already spends against, not a
  second quota — a caller who could otherwise call the JSON endpoint 100
  times a minute must not get 100 more by calling the streaming one instead.
- **No reconnection or resume support.** SSE's `Last-Event-ID`/retry
  machinery exists for a browser `EventSource` reconnecting after a dropped
  connection; with no real consumer yet, building resume semantics now would
  be speculative. A dropped connection simply ends the (already-unbilled,
  in-flight) request, the same as a dropped connection to the JSON endpoint
  today.
- **No per-stage latency instrumentation on the streaming path.**
  `UnifiedAnswerQuestion.execute()`'s `StageTimings` stays the only
  latency-measured path for this iteration; the streaming path can gain the
  same instrumentation later without any wire-protocol change, once there is
  a concrete operational reason to measure it separately from `execute()`'s
  own numbers.
- **No change to `ChatModel.generate()`'s callers.** Every existing RAG
  technique (HyDE, CRAG, Self-RAG, reranking, the evaluation harness) keeps
  calling `generate()`/`complete()` exactly as today. Streaming adds a third
  method; it does not touch the other two.
- **No new numbers exposed to the client that the JSON endpoint doesn't
  already expose.** `answer_response()`
  ([src/api/schemas/sessions.py:87-119](../../../src/api/schemas/sessions.py))
  deliberately withholds router scores, similarity scores, and stage
  timings from the client ("spec decision 8" in that file's own docstring).
  The streaming events carry the same restriction: a `budget` event reports
  which sources made it into context and how many were dropped, never the
  raw token-allocation numbers behind that decision.

## Architecture

```text
POST /sessions/{id}/answers/stream
        |
        v
enforce_rate_limit(chat:{user}) --> enforce_rate_limit(global:chat)   [unchanged, same keys/limits as the JSON route]
        |
        v
await AnswerInSession.stream(tenant, user, session, question)   <-- eager: ownership check + query-budget
        |                                                            check both run HERE, before any byte
        |  raises SessionNotFound / QueryExceedsBudget --> normal    is sent; a rejection is a normal JSON
        |  JSON 404 / 422, exactly like the JSON endpoint today       404/422 response, not a broken stream
        v
StreamingResponse(encode_sse(events), media_type="text/event-stream")
        |
        v
   UnifiedAnswerQuestion._stream_events()  (an async generator, lazy from here on)
        |
        |-- embed (off the event loop, unchanged)
        |-- route            --> yield RoutingStageEvent
        |-- cascade           --> yield RetrievalStageEvent
        |-- allocate + assemble --> yield BudgetStageEvent
        |-- record session budget (unchanged, silent)
        |-- chat_model.stream()  --> yield AnswerChunkEvent, once per delta
        |-- (success)         --> yield AnswerCompleteEvent
        `-- (any exception)   --> log full detail, yield AnswerErrorEvent (generic message)
```

### Why the ownership and budget checks must stay eager

An `async def` function containing `yield` is a generator: calling it runs
none of its body until the caller starts iterating it. Starlette's
`StreamingResponse` sends the ASGI `http.response.start` message — the
status code and headers — *before* it asks the body iterator for its first
item. If `AnswerInSession`'s ownership check lived inside the same generator
that yields SSE events, a cross-tenant request would receive a `200` and
`text/event-stream` headers before `SessionNotFound` ever had a chance to
fire, and only then see the connection close — a strictly worse information
leak than today's clean `404`, because it confirms a stream *would have*
opened for that session id.

The fix is the same shape used elsewhere in this codebase for "check first,
then hand back the real work": `AnswerInSession.stream()` and
`UnifiedAnswerQuestion.stream()` are both plain `async def` coroutines with
no `yield` in their own bodies. Each does its cheap, synchronous-shaped
check up front — session ownership in the first, the query-token-vs-budget
check in the second — and then *returns* the inner async generator that
does the actual streaming. The router's handler does:

```python
events = await answer_in_session.stream(caller.tenant_id, caller.user_id, session_id, payload.question)
return StreamingResponse(encode_sse(events), media_type="text/event-stream")
```

`SessionNotFound` and `QueryExceedsBudget` both raise during that `await`,
before `StreamingResponse` is constructed at all, so both become the exact
same JSON error responses the JSON endpoint already returns (via the
existing `session_not_found_handler`/`query_exceeds_budget_handler` in
[src/api/exception_handlers.py](../../../src/api/exception_handlers.py) —
unmodified, reused as-is). The two rate-limit checks stay where they are
today, before this `await`, for the identical reason.

### Why mid-stream failures become an SSE event, not an HTTP error

Once `encode_sse` has yielded its first frame, the status code is
irreversibly `200`. Any exception raised by a later stage — most plausibly
the chat model's `stream()` call — is therefore caught *inside* the inner
generator (`_stream_events`), logged with full detail via the module
logger, and reported to the client as one `error` SSE event carrying a
generic message. This mirrors the ingestion worker's own established
discipline
([src/workers/ingestion_worker.py](../../../src/workers/ingestion_worker.py):
catches everything unexpected and re-raises as a generic `RuntimeError`
rather than letting an internal exception string reach the caller). A
client that sees an `error` event knows the stream ended abnormally without
learning anything about why.

## Components

### 1. `ChatModel` port gains a streaming method

[src/rag/domain/ports.py](../../../src/rag/domain/ports.py) — one new
abstract method on the existing `ChatModel(ABC)`:

```python
async def stream(self, question: str, context: str) -> AsyncIterator[str]: ...
```

Mirrors `generate()`'s contract exactly (same two parameters, same RAG
system prompt at the adapter level) but yields text deltas instead of
returning one final string. `generate()` and `complete()` are untouched.

Because `ChatModel` is an ABC, every existing subclass must implement the
new method before it can be instantiated. That is four production/test
classes, found by grepping `class \w+\(ChatModel\)` against the whole repo:

- `OllamaChatModel`
  ([src/rag/infrastructure/ollama_chat_model.py](../../../src/rag/infrastructure/ollama_chat_model.py))
  — `stream()` calls `self._client.chat(..., stream=True)`, which returns an
  async iterator of `ChatResponse` chunks; yields
  `chunk.message.content` for each chunk that carries one (ollama's
  streaming chunks can carry an empty final chunk with `done=True` and no
  content).
- `ClaudeChatModel`
  ([src/rag/infrastructure/claude_chat_model.py](../../../src/rag/infrastructure/claude_chat_model.py))
  — `stream()` uses `async with self._client.messages.stream(...) as stream:
  async for text in stream.text_stream: yield text`. `text_stream` already
  filters down to text deltas, sidestepping the same
  thinking-block-ordering hazard `generate()`'s docstring warns about (a
  `ThinkingBlock` never appears on `text_stream`).
- `ContextEchoChatModel`
  ([tests/integration/orchestration_env.py:85-90](../../../tests/integration/orchestration_env.py))
  — the integration-test fake. `stream()` yields `context` in two or three
  arbitrary-width chunks (not one chunk) specifically so a test asserting
  "the client reassembles multiple chunks correctly" is exercising real
  reassembly, not a degenerate single-chunk case.
- `FakeChatModel`
  ([tests/unit/rag_fakes.py:63-77](../../../tests/unit/rag_fakes.py)) — the
  unit-test fake. `stream()` records the call the same way `generate()`
  does (`last_question`/`last_context`) and yields its configured response
  split into fixed-size chunks, configurable via a constructor parameter so
  individual tests can control chunk boundaries.
- `_FreshPassageChatModel`
  (`evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py:209`)
  — an evaluation-harness fake with no bearing on production or CI; gets the
  minimal `stream()` needed only so the class remains instantiable.

### 2. New domain event types

Added to
[src/orchestration/application/unified_answer_question.py](../../../src/orchestration/application/unified_answer_question.py),
beside the existing `StageTimings`/`UnifiedAnswer` dataclasses that already
live in this file rather than in `domain/entities.py` — both are
`UnifiedAnswerQuestion`-specific result shapes, and these new ones are too:

```python
@dataclass(frozen=True)
class RoutingStageEvent:
    decision: RoutingDecision | None
    routing_fallback: RoutingFallback | None

@dataclass(frozen=True)
class RetrievalStageEvent:
    attempts: list[TierAttempt]
    degraded: bool

@dataclass(frozen=True)
class BudgetStageEvent:
    sources: list[ContextItem]
    dropped: dict[Paradigm, int]

@dataclass(frozen=True)
class AnswerChunkEvent:
    text: str

@dataclass(frozen=True)
class AnswerCompleteEvent:
    pass

@dataclass(frozen=True)
class AnswerErrorEvent:
    message: str

UnifiedAnswerEvent = (
    RoutingStageEvent
    | RetrievalStageEvent
    | BudgetStageEvent
    | AnswerChunkEvent
    | AnswerCompleteEvent
    | AnswerErrorEvent
)
```

### 3. `UnifiedAnswerQuestion` gains `stream()`, sharing real logic with `execute()`

Three stages currently inlined in `execute()`'s body are worth extracting
into private helpers precisely because both `execute()` and the new
streaming path need to do the identical computation and only differ in
whether they return a value or `yield` an event built from it:

```python
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
        await self._budget_recorder.record(tenant_id, user_id, session_id, allocation, contributing)
```

`execute()` is refactored to call these three helpers instead of inlining
their bodies; its behavior, return value, and `StageTimings` are unchanged
(the task implementing this must re-run the full existing unit suite for
`UnifiedAnswerQuestion` unmodified and green, proving the refactor is
behavior-preserving before any streaming code is added). The cascade call
itself (`await self._cascade.run(request, decision)`, needing only a
`TierRequest` built from already-local variables) is a single line already
shared verbatim by both call sites without needing its own wrapper — adding
one would be a wrapper around a wrapper for no reader benefit.

`stream()` and its inner generator:

```python
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

        await self._record_budget(tenant_id, user_id, session_id, allocation, cascade_result.contributing)

        async for delta in self._chat_model.stream(question=question, context=assembled.text):
            yield AnswerChunkEvent(delta)
        yield AnswerCompleteEvent()
    except Exception:
        logger.exception("answer streaming failed")
        yield AnswerErrorEvent("answer generation failed")
```

Note `stream()` itself has no `yield` — it is a coroutine that returns an
async generator — which is exactly what makes the eager `QueryExceedsBudget`
check possible (see Architecture above). `_stream_events` is the lazy part.

### 4. `AnswerInSession` gains `stream()`

[src/orchestration/application/answer_in_session.py](../../../src/orchestration/application/answer_in_session.py):

```python
class SessionQuestionAnswerer(Protocol):
    async def execute(self, tenant_id, user_id, session_id, question) -> UnifiedAnswer: ...
    async def stream(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> AsyncIterator[UnifiedAnswerEvent]: ...


class AnswerInSession:
    ...
    async def stream(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> AsyncIterator[UnifiedAnswerEvent]:
        if await self._sessions.find_owned(tenant_id, user_id, session_id) is None:
            raise SessionNotFound(session_id)
        return await self._answerer.stream(tenant_id, user_id, session_id, question)
```

Same ownership-first shape as `execute()`, and for the same documented
reason (the class's docstring already explains why the session is looked up
"before anything else happens" — that reasoning is unchanged, just now
shared by two methods instead of one). `SessionQuestionAnswerer` stays a
`Protocol` (this file's existing, pre-dated choice, not something this spec
changes) and gains the matching `stream` signature so the real
`UnifiedAnswerQuestion` and any test double both satisfy it structurally.

### 5. SSE encoding

New file `src/api/sse.py`:

```python
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
    if isinstance(event, RoutingStageEvent):
        decision = event.decision
        return "routing", {
            "paradigms": [] if decision is None else [p.value for p in PARADIGM_ORDER if p in decision.paradigms],
            "mode": None if decision is None else decision.mode.value,
            "fallback": event.routing_fallback,
        }
    if isinstance(event, RetrievalStageEvent):
        return "retrieval", {
            "attempts": [
                {"paradigm": a.paradigm.value, "outcome": a.outcome.value, "elapsed_ms": a.elapsed_ms}
                for a in event.attempts
            ],
            "degraded": event.degraded,
        }
    if isinstance(event, BudgetStageEvent):
        return "budget", {
            "sources": [
                {"paradigm": s.paradigm.value, "content": s.content, "source_id": str(s.source_id) if s.source_id else None}
                for s in event.sources
            ],
            "dropped": {p.value: c for p, c in event.dropped.items()},
        }
    if isinstance(event, AnswerChunkEvent):
        return "chunk", {"text": event.text}
    if isinstance(event, AnswerCompleteEvent):
        return "done", {}
    return "error", {"detail": event.message}  # AnswerErrorEvent


async def encode_sse(events: AsyncIterator[UnifiedAnswerEvent]) -> AsyncIterator[bytes]:
    async for event in events:
        name, payload = _wire(event)
        yield f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
```

`json.dumps` escapes any newline inside `content`/`text` as `\n` before it
ever reaches the frame, so an SSE data line — which the spec forbids from
containing a literal newline — is safe by construction; this needs a test
asserting a chunk containing an embedded `\n` round-trips correctly, because
it is the one easy way to build this wrong (interpolating raw text into the
frame instead of JSON-encoding it first).

### 6. The new router endpoint

[src/api/routers/sessions.py](../../../src/api/routers/sessions.py), added
beside `answer()`, reusing the same `AnswerRequest` request schema (no new
request body shape) and the same two rate-limit calls in the same order:

```python
from fastapi.responses import StreamingResponse
from src.api.sse import encode_sse

@router.post("/{session_id}/answers/stream")
async def answer_stream(
    session_id: uuid.UUID,
    payload: AnswerRequest,
    request: Request,
    response: Response,
    caller: Caller = Depends(get_caller),
    answer_in_session: AnswerInSession = Depends(get_answer_in_session),
) -> StreamingResponse:
    await enforce_rate_limit(
        request, response, limiter=get_rate_limiter(), key=f"chat:{caller.user_id}", limit=chat_rate_limit()
    )
    await enforce_rate_limit(
        request, response, limiter=get_rate_limiter(), key=GLOBAL_CHAT_KEY,
        limit=global_chat_limit(), window_seconds=GLOBAL_CHAT_WINDOW_SECONDS,
    )
    events = await answer_in_session.stream(caller.tenant_id, caller.user_id, session_id, payload.question)
    return StreamingResponse(encode_sse(events), media_type="text/event-stream")
```

No `response_model` (a `StreamingResponse` has no Pydantic body to
validate), no change to `src/api/main.py`'s `MaxBodySizeMiddleware`
overrides (this route takes the same `AnswerRequest` body as the JSON
endpoint, well inside the existing 16 KiB default), and no new router
registration (this lives in the already-registered `sessions_router`).

## Authorization

Identical guarantee to the JSON endpoint, by construction rather than by a
second implementation: `AnswerInSession.stream()` calls the exact same
`ChatSessionRepository.find_owned()` the JSON path already calls, raising
the exact same `SessionNotFound`, handled by the exact same
`session_not_found_handler` (which already logs the `authz_denied` security
event with tenant/user/session ids — unmodified, reused as-is). The
Architecture section above is the reasoning for why this check is safe to
share: because it happens before `StreamingResponse` is ever constructed,
"denied" and "not yet started" are the same state, with no window where a
stream could open and then get denied.

## Testing plan

**Unit** (`tests/unit/test_unified_answer_question.py`, extended):

- `stream()` yields the exact sequence `RoutingStageEvent`,
  `RetrievalStageEvent`, `BudgetStageEvent`, one or more `AnswerChunkEvent`
  (using `FakeChatModel`'s multi-chunk mode), then `AnswerCompleteEvent`, on
  a happy path built the same way the existing `execute()` tests build their
  fixtures.
- `stream()` raises `QueryExceedsBudget` before returning anything
  iterable, for an over-budget question (proves the eager check).
- When `FakeChatModel.stream()` is configured to raise mid-iteration,
  `_stream_events` yields `AnswerErrorEvent` as its last event and nothing
  after it.
- `execute()`'s full existing test suite stays green unmodified, proving
  the `_decide_route`/`_do_budget`/`_record_budget` extraction changed
  nothing observable about it.
- New unit tests for `OllamaChatModel.stream()` and `ClaudeChatModel.stream()`
  against each provider's fake client, asserting the deltas are yielded in
  order and concatenate to the same text `generate()` would have returned
  for an equivalent fake response.
- `src/api/sse.py`'s `encode_sse()`/`_wire()` tested directly: one test per
  event type asserting exact frame bytes, plus the embedded-newline
  round-trip test called out above.

**Integration** (new file, e.g.
`tests/integration/test_sessions_stream_endpoint.py`, reusing
`tests/integration/orchestration_env.py`'s `build_env`/`ContextEchoChatModel`
exactly as `test_sessions_endpoints.py` already does):

- A real request against the streaming endpoint, driven with `httpx`'s
  `client.stream("POST", ...)`, reassembles into a well-formed sequence of
  named SSE events ending in `done`, and the concatenated `chunk` text
  matches what the JSON endpoint would have returned for the same question
  against the same fixtures.
- **The security-critical test**: a second tenant's (and, separately, a
  second user's, same tenant) caller requesting another session's stream
  gets a plain `404` with the same `{"detail": "Session not found"}` body
  the JSON endpoint returns, with the response's `content-type` never
  becoming `text/event-stream` and zero SSE bytes in the body — proving the
  eager-check design actually holds against the real ASGI stack, not just
  in a unit test's mocked-out world.
- A request that exceeds the chat rate limit gets the existing `429` JSON
  response, not a stream — same reasoning, proven end-to-end.
- A request whose question exceeds the Query budget slice gets the existing
  `422`, not a stream.

## SOLID / GRASP / DRY notes

- **Single Responsibility**: `src/api/sse.py` knows only how to render a
  domain event as bytes on the wire; it has zero orchestration knowledge.
  `UnifiedAnswerQuestion` knows only how to run the pipeline and report on
  it (by return value or by event); it has zero knowledge that HTTP or SSE
  exist. `AnswerInSession` still knows only "is this session the caller's."
- **Open/Closed**: adding `stream()` to `ChatModel` is the one interface
  change existing adapters must react to (unavoidable — a new capability
  needs a new abstract method), but no existing method on any existing
  class changes its signature or behavior.
- **DRY**: `_decide_route`, `_do_budget`, and `_record_budget` are each
  written once and called from both `execute()` and `_stream_events()`; the
  session-ownership check is written once in `AnswerInSession` and called
  from both of its own methods. The one duplication this design accepts
  knowingly is the *sequencing* of five stage calls appearing in both
  `execute()` and `_stream_events()` — extracting that into a single shared
  routine would require it to either return a value or yield one, which is
  precisely the difference the two methods exist to express; forcing that
  through one code path would need its own event-vs-return abstraction that
  serves no third caller today (YAGNI).
- **GRASP Information Expert**: `UnifiedAnswerQuestion` is where the
  pipeline's data already lives, so it is where both "compute the full
  answer" and "report the answer as it's computed" belong — not a new class
  that would need the same constructor dependencies duplicated.
- **Protected Variations**: the router depends only on
  `AnswerInSession`/`ChatModel` abstractions, never on which concrete chat
  provider is configured or on Celery/Redis (irrelevant here) — consistent
  with how the ingestion endpoint kept its router ignorant of Celery.

## Disclosed risks

- **`ollama`'s and `anthropic`'s real streaming response shapes are trusted
  from their client library documentation, not re-verified against a
  captured live response as part of this spec.** The task implementing
  `OllamaChatModel.stream()`/`ClaudeChatModel.stream()` must confirm the
  exact chunk attribute names (`chunk.message.content` for ollama,
  `stream.text_stream` for Anthropic) against the versions actually pinned
  in `pyproject.toml`, the same way `OllamaChatModel.generate()`'s existing
  comments already record having verified `ChatResponse`'s fields against
  the installed client rather than assuming them.
- **A hung chat-model stream, or a connected client that never reads, was a
  real gap this design shipped with -- and follow-up `#196` has since closed
  it with both pieces this section originally said were still needed: a
  per-request generation deadline and a per-user concurrent-stream cap.**
  `UnifiedAnswerQuestion._stream_events()`
  (`src/orchestration/application/unified_answer_question.py`) now wraps the
  chat model's `stream()` call in an overall wall-clock deadline via a
  module-level `_with_deadline()` helper -- `DEFAULT_STREAM_DEADLINE = 90.0`
  seconds, configurable through `STREAM_DEADLINE_SECONDS` -- a reasoned
  engineering default disclosed explicitly as such rather than one derived
  from a measured p95 generation latency this project doesn't have yet,
  chosen to sit far inside the `anthropic` SDK's own ten-minute read-timeout
  ceiling while still bounding the worst case this section originally
  flagged. A stream that exceeds it ends in one generic `AnswerErrorEvent`,
  the same shape a mid-generation chat-model failure already produces.
  Separately, `STREAM_CONCURRENCY_LIMIT_PER_USER` (default 5) now bounds how
  many streams one user can hold open at once, shared across every session,
  via a new `StreamConcurrencyLimiter` port
  (`src/identity/domain/ports.py`) and a Redis sorted-set adapter
  (`src/identity/infrastructure/redis_stream_concurrency_limiter.py`) whose
  slots self-expire even if the process holding one dies without releasing
  it; the router acquires a slot only after `AnswerInSession.stream()` has
  already succeeded, so a request this endpoint's existing rate limits or
  object-level-authorization check would already reject never touches a
  slot. `docs/security/SECURITY.md`'s rate-limiting section has the full
  numbers, file paths, and citations for both controls' tests.
- **A genuinely dropped TCP connection mid-stream was verified, not merely
  assumed.** A live probe against a real uvicorn server confirmed Starlette
  cancels the body generator promptly (within milliseconds) on a hard socket
  close, raising `CancelledError` into `_stream_events` -- which propagates
  rather than being caught by the generic `except Exception` (`CancelledError`
  inherits from `BaseException`, not `Exception`), so cleanup runs correctly.
  This downgrades what was originally an untested assumption to a confirmed,
  correct behavior.
