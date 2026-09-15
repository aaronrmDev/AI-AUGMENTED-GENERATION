# Unified API Batch A: Session-Scoped Unified Answer Endpoint — Design Spec

**Scope:** Story #183 under Epic #182. The plan that implements this spec is `docs/superpowers/plans/2026-09-13-unified-api-sessions.md`.

## What this batch builds, and why now

Epic #150 built and measured all five orchestration coordinators. A client still can't reach them: `POST /chat` answers from RAG alone, and `UnifiedAnswerQuestion` is called only by tests and evaluation runners. This batch exposes the unified path over HTTP. It adds three endpoints:

- `POST /sessions` creates a chat session for the caller.
- `GET /sessions` lists the caller's own sessions.
- `POST /sessions/{session_id}/answers` answers a question within one of them, through the router, the budget allocator, and the latency cascade.

The unified path needs a session: `UnifiedAnswerQuestion` records each turn's context budget against one. No endpoint creates a session today; only tests and runners insert rows. And a client-supplied session id is exactly the object-level authorization surface (OWASP API1) that Epic #150 said needed its own review. So sessions and answers ship together, with that review.

The frontend, the durable workers, and the ingestion endpoint all come after this, and the frontend needs this endpoint first.

## Decisions made for this batch

The user delegated design authority for this session ("take full control over the project and ensure to finish each minimal detail with full testing"). Every decision a brainstorming session would normally put to them is recorded here with its reason, so any of them can be revisited.

1. **Answers live under a session the caller created.**
   - The session id is a path parameter, so the object under authorization is explicit in every route that touches it.
   - Why: a session id in a request body is easy to forget to check, and three resource-shaped routes are the smallest set that lets a client start, find, and use a session.
2. **`POST /chat` stays the RAG-only baseline.**
   - Its contract doesn't change. It gains only the rate limit in decision 6.
   - Why: existing callers keep working, and RAG-only against unified stays comparable from a client, the way Batch A measured it.
3. **Identity comes only from the verified access token.**
   - The tenant is the token's `tenant_id`, and the user is its `sub`. No body field or query parameter names either.
   - Why: anything a client can type, it can type wrong on purpose.
4. **A session that isn't the caller's looks exactly like a session that doesn't exist.**
   - `AnswerInSession` looks the session up with `ChatSessionRepository.find_owned(tenant_id, user_id, session_id)` before it embeds, routes, or retrieves anything. RLS enforces the tenant, and the query also matches `user_id`.
   - A missing session, another user's, and another tenant's all raise `SessionNotFound`, which becomes the same `404 {"detail": "Session not found"}`.
   - `PostgresSessionBudgetRecorder` keeps its own `UPDATE ... WHERE id = :id AND user_id = :user_id` check as a second line.
   - Why: until now a foreign session id was caught only by that recorder, after the cascade had already spent retrieval work (see `UnifiedAnswerQuestion`'s docstring). Checking first spends nothing. One status and body for all three cases means a caller can't learn which session ids exist.
5. **The API wires MAG and RAG tiers, and no CAG tier yet.**
   - `LatencyCascade` attempts only the tiers it was given and always keeps RAG as the last resort, so a query the router sends to CAG alone is answered from RAG.
   - Why: a CAG tier needs a warmed frozen cache inside the serving process, and something to warm it. The only `FrozenCache` is `HFFrozenCache`, a CPU distilgpt2 proxy built for measurement, and nothing warms any cache outside tests and runners, because `src/workers/` doesn't exist. Wiring it would load a model into every API process to serve nothing but misses.
   - Consequence: answers come from MAG and RAG until the workers batch brings warming and a serving cache. The response's `attempts` show which tiers ran.
6. **Answering is rate limited per user.**
   - `POST /sessions/{session_id}/answers` and `POST /chat` share one budget: 100 requests per 60 seconds per user, under the key `chat:{user_id}`, the limit `docs/security/SECURITY.md` names for chat.
   - `POST /sessions` gets 20 per 60 seconds per user, under `sessions:{user_id}`. A row is cheap, but unbounded creation fills a table.
   - `GET /sessions` isn't limited: it reads at most 100 indexed rows.
   - The limiter logic moves out of `src/api/routers/auth.py` into `src/api/rate_limit.py`, parameterized by key, limit, and window. The auth routes keep their IP-keyed 5 per minute. Every limited route sends `X-RateLimit-*` headers.
   - The chat limit is read per call from `CHAT_RATE_LIMIT_PER_MINUTE`, default 100, the same per-call pattern `COOKIE_SECURE` uses. That gives operators a knob, and gives the integration test a way to reach the limit without 101 requests.
7. **One pipeline per process, built on first use.**
   - `src/api/unified_pipeline.py` builds, once:
     - a `CachingEmbeddingModel` around the process's existing `SentenceTransformersEmbedder`;
     - `SearchDocuments` over the existing `QdrantVectorStore`;
     - `PrototypeQueryClassifier`, the millisecond-scale classifier Batch A measured (the LLM classifier is too slow for a request path, as `DEFAULT_CLASSIFIER_TIMEOUT` records);
     - a `LatencyCascade` of `MagTier` and `RagTier` with Concept 5's default timeouts;
     - `PostgresSessionBudgetRecorder` over the app's sessionmaker;
     - the chat model from `get_chat_model()`.
   - A FastAPI dependency, `get_answer_in_session`, hands it to the router, so tests can override it through `app.dependency_overrides`.
   - The app's shutdown hook calls `LatencyCascade.drain()`, so a cancelled tier's cleanup finishes before the process exits.
   - Why built on first use: importing `src.api.main` shouldn't embed every routing exemplar, which each test process would otherwise pay.
8. **The response shows how an answer was made, not the numbers that tune it.**
   - It carries `answer`, `sources` (each with `paradigm`, `content`, and `source_id`), `routing` (`paradigms`, `mode`, and `fallback`), `attempts` (each with `paradigm`, `outcome`, and `elapsed_ms`), `degraded`, and `dropped` per paradigm.
   - It leaves out router scores, per-source similarity scores, and stage timings. The turn's budget allocation is persisted on the session, and `GET /sessions` returns it.
   - Why: provenance, routing, and degradation are what a client shows a user. Scores and timings are a tuning surface with no client use yet, and fine-grained timings are a side channel.
9. **A session holds a title and its latest budget, and nothing more yet.**
   - This batch stores no turn history, working memory, or episodes.
   - Why: which turns MAG should keep is a gating decision `docs/architecture/MAG.md` covers on its own terms, and `UnifiedAnswerQuestion` reads MAG facts but doesn't write them. Storing turns deserves its own story.
10. **Errors map to precise statuses, and nothing new leaks.**
    - `SessionNotFound` returns 404.
    - `QueryExceedsBudget` returns 422. The 4,000-character question cap can't reach the 12,800-token Query slice, so it's mapped defensively.
    - A blank or oversized question, a title over 200 characters, and a `limit` outside 1 to 100 return 422 from request validation.
    - A chat model failure stays an unhandled 500 with no details in the body, as it is for `POST /chat` today.
11. **One migration: an index for listing a user's sessions.**
    - `sessions` already has every column this needs (`id`, `user_id`, `tenant_id`, `title`, `context_budget`, `created_at`), with RLS and `app_user` grants since migration 0001.
    - Migration 0007 adds `ix_sessions_user_id_created_at` on `(user_id, created_at DESC)`, which is the list query's shape. It lands before the repository that uses it.
    - Deleting a user who has sessions still fails on the foreign key. No user deletion flow exists, so that belongs to whichever batch builds one.

## Domain

**`src/identity/domain/entities.py`:**

- `ChatSession(id, tenant_id, user_id, title: str | None, context_budget: dict[str, Any] | None, created_at: datetime)`. A title longer than 200 characters raises `ValueError`.

**`src/identity/domain/ports.py`:**

- `ChatSessionRepository`, every method taking the tenant and the owning user:
  - `create(tenant_id, user_id, title) -> ChatSession`;
  - `find_owned(tenant_id, user_id, session_id) -> ChatSession | None`;
  - `list_owned(tenant_id, user_id, limit) -> list[ChatSession]`, newest first.

## Application

- **`src/identity/application/start_chat_session.py`: `StartChatSession(repository)`.** `execute(tenant_id, user_id, title)` strips the title, stores a blank one as `None`, and returns the created session.
- **`src/identity/application/list_chat_sessions.py`: `ListChatSessions(repository)`.** `execute(tenant_id, user_id, limit)` refuses a limit outside 1 to 100 with `ValueError`, and returns the caller's sessions newest first.
- **`src/orchestration/application/answer_in_session.py`: `AnswerInSession(sessions, answerer)`.** `execute(tenant_id, user_id, session_id, question)`:
  1. Looks the session up with `find_owned`. If it's missing, raises `SessionNotFound(session_id)` without calling the answerer.
  2. Otherwise returns `answerer.execute(tenant_id, user_id, session_id, question)`.

  `answerer` is typed by a small `SessionQuestionAnswerer` protocol that `UnifiedAnswerQuestion` already satisfies, so unit tests pass a recording fake.

## Infrastructure

- **`alembic/versions/0007_sessions_user_index.py`.** Adds `ix_sessions_user_id_created_at` on `sessions (user_id, created_at DESC)`, and drops it on downgrade.
- **`src/identity/infrastructure/postgres_chat_session_repository.py`: `PostgresChatSessionRepository(sessionmaker)`.**
  - Each call opens its own short transaction and sets the tenant context, following `PostgresSessionBudgetRecorder`. The ownership check then never holds a connection open across the cascade and generation.
  - `create` inserts and returns the row with its server-generated `id` and `created_at`.
  - `find_owned` selects by `id` and `user_id`; RLS supplies the tenant.
  - `list_owned` orders by `created_at DESC, id DESC` and applies the limit.

## API

- **`src/api/rate_limit.py`.** `enforce_rate_limit(request, response, key, limit, window_seconds)` and `RateLimitExceeded`, moved from `src/api/routers/auth.py` without changing behavior, together with `RateLimitHeadersMiddleware`. `auth.py` calls it with its IP key and 5 per minute.
- **`src/api/schemas/sessions.py`.**
  - `CreateSessionRequest(title: str | None, max 200)`.
  - `SessionResponse(id, title, created_at, context_budget)` and `SessionListResponse(sessions)`.
  - `AnswerRequest(question: str, 1 to 4,000 characters, not blank)`.
  - `AnswerResponse(answer, sources, routing, attempts, degraded, dropped)`, with `SourceSchema`, `RoutingSchema`, and `AttemptSchema`.
  - A pure `answer_response(unified_answer) -> AnswerResponse` mapping, unit tested on its own.
- **`src/api/unified_pipeline.py`.** The process-wide composition in decision 7, behind a cached builder.
- **`src/api/caller.py`.** `Caller(tenant_id, user_id)` and `caller_from_claims(claims)`, which reads the token's `tenant_id` and `sub` and treats claims this issuer would never mint as an invalid token. It lives apart from `dependencies.py`, which reads the environment and loads a model at import, so a unit test can check it.
- **`src/api/dependencies.py`.** Adds `get_caller()`, `get_chat_session_repository()`, a cached `get_unified_pipeline()`, and `get_answer_in_session()`.
- **`src/api/routers/sessions.py`.** The three routes. Each takes `Caller` from the token, applies its rate limit, and calls one use case.
- **`src/api/routers/chat.py`.** Adds the shared per-user chat limit.
- **`src/api/exception_handlers.py`.** `SessionNotFound` returns 404 with `{"detail": "Session not found"}`, and `QueryExceedsBudget` returns 422.
- **`src/api/main.py`.** Includes the sessions router, and drains the cascade on shutdown.

## Tenant and user isolation

- Every repository call sets the tenant context in its own transaction, so RLS enforces the tenant, and every query also filters by the token's user.
- The unified path's MAG tier already searches by the token's user (`SessionScopedSemanticFactSearch`), and its RAG tier by the token's tenant. The session check adds what neither covered: that the session in the path belongs to that user.
- The integration tests check the following:
  - another user in the same tenant gets 404 for a session, and the victim's `context_budget` is unchanged;
  - another tenant gets the same 404;
  - an unknown id gets a byte-identical 404;
  - `GET /sessions` never lists another user's or another tenant's sessions.

## Testing plan

Unit tests use fakes and never assert on wall-clock time.

- **`ChatSession`** is tested for title length at 200 and 201 characters.
- **`StartChatSession`** is tested for a stripped title, a blank title stored as `None`, and an owner taken from its arguments.
- **`ListChatSessions`** is tested for limits 0, 1, 100, and 101, and for passing the caller through.
- **`AnswerInSession`** is tested for delegating when the session is owned, and for raising `SessionNotFound` without calling the answerer when it isn't.
- **The schemas** are tested with blank, whitespace-only, 4,000-character, and 4,001-character questions, and 200- and 201-character titles.
- **`answer_response`** is tested for mapping every field, including a fallback route and a degraded answer.
- **`enforce_rate_limit`** is tested with a fake limiter for the allowed path's headers and the refused path's `RateLimitExceeded`.

Integration tests use real PostgreSQL, Redis, and Qdrant through TestContainers, and the app through `httpx.ASGITransport`, with the chat model overridden by a context-echo model:

- migration 0007's index;
- `PostgresChatSessionRepository` round trips, ordering, limit, and cross-user and cross-tenant refusal;
- `401` without a token on all three routes;
- create, then list, then answer, with the answer's routing, attempts, and sources present and the session's `context_budget` written;
- the isolation cases above;
- the 429 on the chat limit, lowered through `CHAT_RATE_LIMIT_PER_MINUTE`, shared between `POST /chat` and answers;
- validation 422s through the real app.

A live check runs once before the merge, against a real `uvicorn` process with Ollama's `qwen3.5` as the chat model: register, log in, create a session, answer in it, and try another user's session. The plan's execution notes record the transcript's statuses.

## Security review

Before the merge, the `fullstack-e2e-security-engineer` skill reviews the branch against `docs/security/SECURITY.md` and OWASP's API Security Top 10. Every finding is fixed test-first, or recorded with its reason, in the plan's execution notes. `docs/security/SECURITY.md` is updated to describe the object-level authorization check and the chat rate limit as enforced controls.

## Documentation

- `docs/architecture/OVERVIEW.md` describes the endpoints and removes them from what remains unbuilt.
- `docs/database/DATABASE.md` covers the sessions repository and index 0007.
- `docs/testing/TESTING.md`, `CLAUDE.md`, and `README.md` get the new test counts and built scope.
- `docs/architecture/CONTEXT_GRAPH.md` gets the new API and identity classes.

## What this batch does not do

- **It wires no CAG tier** (decision 5).
- **It doesn't stream answers.** Streaming belongs with the frontend that consumes it.
- **It adds no ingestion endpoint.** Who may write tenant-wide sources needs a role model this project doesn't have.
- **It stores no turn history or memory** (decision 9).
- **It doesn't rename or delete sessions**, or page through them with a cursor.
- **It builds no workers, frontend, or Kubernetes deployment.**
