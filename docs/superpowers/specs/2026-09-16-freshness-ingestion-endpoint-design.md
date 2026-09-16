# Freshness-Routed Ingestion Endpoint — Design Spec

## Context

The "Unified API" milestone (Epic #182) names three deliverables: session-scoped unified answers (built, Story #183), hardening the session endpoints against abuse (built, Task #193), and an HTTP endpoint for freshness-routed ingestion — `IngestDataSource` (`src/orchestration/application/ingest_data_source.py`), built and live-measured against real infrastructure, but reachable today only from tests, the evaluation harness, and nowhere a client can call it. This spec covers that endpoint. Streaming (delivering an answer incrementally rather than as one JSON response) is the milestone's remaining item; it's deliberately out of scope here, because no frontend or other consumer exists yet to design its framing against, and it doesn't share a design question with ingestion.

Story #183's security review recorded the ingestion endpoint as blocked on "a tenant role model this project doesn't have yet," reasoning that the endpoint accepts client-supplied scope (`tenant` vs. `user`) and someone has to decide who's allowed to write a tenant-wide source. Investigation during this spec's brainstorming found that blocker doesn't hold in its original form: `RegisterUser.execute` (`src/identity/application/register_user.py`) mints a brand-new `tenant_id` on every registration, with no code path anywhere that lets two users end up sharing one — a tenant is, today, always exactly one user. So there is no multi-user privilege question to adjudicate yet. The actual gap is the same one `sessions.py` already closed for session ownership: the endpoint must derive `tenant_id` and, when the request declares `scope=user`, `user_id` from the verified `Caller` (`src/api/caller.py`), never from the request body. A real multi-member tenant role model — the thing that would make "tenant-wide vs. personal" a genuine authorization question — is out of scope here and named explicitly as future work, to build only once a tenant can actually have more than one member.

Per direction from this spec's brainstorming session, the endpoint is built as a fully asynchronous, worker-backed operation rather than a synchronous request/response call or an in-process `BackgroundTasks` callback. This is this project's first background-worker subsystem — `src/workers/` has been named in the Phase 1 blueprint since before any code existed and has stayed empty until now. Celery is the task-queue library (already named in `docs/architecture/OVERVIEW.md`'s stack table); Redis is the broker and result backend, not RabbitMQ — Redis is already a running, depended-on service in this stack (rate limiting, refresh tokens), and standing up RabbitMQ for a single job type would be new operational surface with no second consumer yet to justify it. `docs/architecture/OVERVIEW.md`'s own "when you might deviate from this stack" section already names Redis-backed job handling as the sanctioned simpler substitute for Celery+RabbitMQ, so this isn't a departure from anything this project has committed to — RabbitMQ stays named as the option a second, higher-volume job type might justify revisiting.

## Goals

- A caller can submit a data source for ingestion over HTTP and later check whether it succeeded.
- The endpoint is exactly as safe as the rest of this API: scope derived from the verified token, a rate limit, a body-size ceiling, and object-level authorization on every read — not weaker guarantees because the work happens to run asynchronously.
- `IngestDataSource` itself does not change. It is already built, tested, and used by the evaluation harness; this spec adds a new way to invoke it, not a new way it behaves.

## Non-goals

- A multi-member tenant role model (future work, named above).
- Streaming answers (the milestone's other remaining item; unrelated design question).
- A generic, multi-job-type worker framework. This spec builds exactly what ingestion needs. A second background job type, when one is actually needed, reuses the same shape rather than something built speculatively now.
- Job cancellation, retry-policy tuning, or a job-history/list endpoint. None of these are asked for; building them now would be scope this feature doesn't need yet.

## Architecture

```text
Client
  │
  │ POST /data-sources  { source_key, scope, expected_change_interval_seconds, content }
  ▼
src/api/routers/data_sources.py
  │  - validates the request (Pydantic)
  │  - derives tenant_id from Caller; derives user_id from Caller only if scope == "user"
  │  - enforce_rate_limit (10/min per user)
  ▼
IngestionJobDispatcher.dispatch(...)             <-- port, src/api/ingestion_jobs.py
  ▼
CeleryIngestionJobDispatcher                      <-- adapter, src/workers/celery_ingestion_dispatcher.py
  │  - records ownership: Redis "ingestion_job:{task_id}" -> "{tenant_id}:{user_id or ''}", TTL 24h
  │  - ingest_data_source_task.delay(...)
  ▼
Redis (broker)
  ▼
Celery worker process (docker-compose "worker" service)
  ▼
src/workers/ingestion_worker.py: ingest_data_source_task
  │  - builds a fully-wired IngestDataSource (real Postgres/Qdrant/Redis adapters)
  │  - bridges the sync Celery task to IngestDataSource.execute (async) via asyncio.run
  ▼
IngestDataSource.execute(...)   <-- unchanged, src/orchestration/application/ingest_data_source.py
  │
  ▼
Celery result backend (Redis) stores state + IngestionResult (or the failure)


Client
  │
  │ GET /data-sources/jobs/{task_id}
  ▼
src/api/routers/data_sources.py
  ▼
IngestionJobDispatcher.status(task_id, caller)
  ▼
CeleryIngestionJobDispatcher
  │  - reads the ownership record; caller mismatch or missing record -> 404, same shape as
  │    AnswerInSession's "missing or not yours" 404
  │  - AsyncResult(task_id, app=celery_app).state / .result / .traceback
  ▼
JobStatusResponse { state: "pending" | "success" | "failure", result, error }
```

## Components

### `src/api/ingestion_jobs.py` (new)

The port this feature is built around. Defines:

- `class JobState(str, Enum)`: `PENDING`, `SUCCESS`, `FAILURE`.
- `@dataclass(frozen=True) class JobStatus`: `state: JobState`, `result: IngestionResult | None`, `error: str | None`.
- `class IngestionJobDispatcher(Protocol)`:
  - `def dispatch(self, *, tenant_id: uuid.UUID, profile: DataSourceProfile, content: str, user_id: uuid.UUID | None) -> str` — returns a task id.
  - `def status(self, task_id: str, *, tenant_id: uuid.UUID, user_id: uuid.UUID) -> JobStatus | None` — `None` means "not found or not yours," so the router maps it to the same 404 shape used everywhere else in this API.

This lives in `src/api/`, not in `src/orchestration/domain/ports.py` alongside `IngestDataSource`'s own ports (`DataSourceRepository`, `RagIndex`, `ExpiringCache`, `SessionFactWriter`). Those exist because `IngestDataSource`'s constructor needs them; `IngestionJobDispatcher` is not one of `IngestDataSource`'s dependencies — it's how the HTTP layer chooses to invoke `IngestDataSource` at all. Mixing the two would blur a real distinction: `IngestDataSource` has no idea it's being run synchronously in a test, from the evaluation harness, or inside a Celery task, and this port is what keeps it that way.

### `src/api/routers/data_sources.py` (new)

`POST /data-sources` and `GET /data-sources/jobs/{task_id}`, matching the existing router shape (`Caller = Depends(get_caller)`, `IngestionJobDispatcher = Depends(get_ingestion_job_dispatcher)`). Pure HTTP-shape responsibility: validate, derive scope, call the port, map the result to a response — no ingestion logic of its own, the same division `sessions.py` already keeps between the router and `AnswerInSession`.

### `src/api/schemas/data_sources.py` (new)

- `IngestDataSourceRequest`: `source_key: str`, `scope: Literal["tenant", "user"]`, `expected_change_interval_seconds: int` (must be `> 0`), `content: str` (must be non-empty).
- `IngestDataSourceAcceptedResponse`: `task_id: str`.
- `JobStatusResponse`: `state: Literal["pending", "success", "failure"]`, `result: IngestionResultSchema | None`, `error: str | None`.

### `src/workers/` (new package — this project's first)

- `celery_app.py`: the Celery application instance. `broker` and `backend` both point at `REDIS_URL` (the same environment variable `src/api/dependencies.py` already reads), `task_serializer="json"`, `result_serializer="json"`, `accept_content=["json"]` — explicit rather than Celery's pickle-capable defaults, since this queue only ever needs to carry the plain JSON-shaped ingestion arguments and results.
- `ingestion_worker.py`: `@celery_app.task(name="ingest_data_source") def ingest_data_source_task(...)`. This is the worker process's composition root — the one place that builds a fully-wired `IngestDataSource` with real `PostgresDataSourceRepository`, `ChunkedRagIndex`, `ExpiringFrozenCache`, `CacheWarmedRetrieve`, and `RecordSemanticFactWriter` adapters, the same infrastructure classes `tests/integration/freshness_env.py` already wires for the same purpose. Celery tasks are synchronous by default; the task body bridges to `IngestDataSource.execute` (an `async def`) with `asyncio.run(...)`, since a Celery worker process has no already-running event loop the way `uvicorn` does — a real, disclosed wrinkle, not an oversight.
- `celery_ingestion_dispatcher.py`: `CeleryIngestionJobDispatcher`, the only piece of `src/workers/` the API process ever imports. `dispatch()` writes the ownership record to Redis and calls `ingest_data_source_task.delay(...)`; `status()` checks ownership first, then wraps `celery.result.AsyncResult`.

### `src/api/dependencies.py` (modified)

Adds `get_ingestion_job_dispatcher() -> IngestionJobDispatcher`, `functools.cache`d like `get_unified_pipeline`, returning a `CeleryIngestionJobDispatcher` wired with the existing Celery app and a Redis client from this module's existing Redis-client factory.

### `src/api/main.py` (modified)

Registers `data_sources_router`; adds `"/data-sources": 11 * 1024 * 1024` to `MaxBodySizeMiddleware`'s `path_overrides` (matching `/documents`, since `content` can be a full document's worth of text).

### `docker/docker-compose.yml` (modified)

A new `worker` service, reusing `Dockerfile.api`'s existing image (no new Dockerfile — the worker's dependencies are identical to the API's today; a dedicated `Dockerfile.worker` stays named-but-deferred in the Phase 1 blueprint until the two processes' dependencies actually diverge), `command: uv run celery -A src.workers.celery_app:celery_app worker --loglevel=info` (the explicit `:celery_app` attribute reference, since Celery's `-A module` shorthand only auto-discovers an attribute literally named `app` or `celery` — this module's instance is named `celery_app` for clarity, so the CLI invocation has to say so), `depends_on` Postgres, Redis, and Qdrant healthy, same environment variables the `api` service already sets.

## Authorization and abuse controls

- `tenant_id` always comes from `Caller.tenant_id`; `user_id` comes from `Caller.user_id` when `scope == "user"` and is otherwise `None` — never read from the request body. This is what closes Story #183's original finding without a role model: `ScopeMismatch` (`IngestDataSource`'s own guard against a scope/user_id mismatch) becomes unreachable through this endpoint by construction, the same way a client can no longer request another user's session.
- `GET /data-sources/jobs/{task_id}` enforces the ownership record before ever touching Celery's result backend. A task id belonging to another tenant or user returns the same `404` shape `AnswerInSession` already uses for a session that's missing, another user's, or another tenant's — a caller can't learn whether a task id exists at all, matching the existing "can't learn which ids exist" discipline.
- `POST /data-sources` is rate-limited at 10 requests/minute per user via the existing `enforce_rate_limit` (`src/api/rate_limit.py`), the same figure `docs/security/SECURITY.md` already names as planned-but-unbuilt for document uploads — ingestion is a comparably heavy write (RAG re-index, cache eviction, semantic-fact write).
- `/data-sources` gets the same exact-path 11 MiB body-size override `/documents` has, via `MaxBodySizeMiddleware`.

## Failure semantics — a deliberate, disclosed trade-off

`202 Accepted` means "accepted for processing," not "succeeded." Any rejection `IngestDataSource.execute` can still raise (a route/scope conflict inside `_new_source`, for instance) now surfaces only through `GET /data-sources/jobs/{task_id}` reporting `state="failure"` with the exception's message as `error` — never as an immediate 4xx at POST time, because the router can't know the outcome synchronously anymore. This is the real cost of going fully asynchronous rather than a synchronous call, and it's called out here explicitly rather than left for a caller to discover by surprise; the response schema's own description says so too.

## Testing plan

- **Unit** (`tests/unit/`): `IngestDataSourceRequest`/`JobStatusResponse` schema validation; an `InMemoryIngestionJobDispatcher` fake (runs the given task synchronously in-process, no Celery or Redis at all) added to `tests/unit/fakes.py`, used to test the router's request-handling and scope-derivation logic in isolation; `CeleryIngestionJobDispatcher`'s ownership-record encode/decode logic against a fake Redis client.
- **Integration** (`tests/integration/`): a real Celery worker (via Celery's own `celery_worker`/`celery_app` pytest fixtures, consuming from the real TestContainers Redis instance the rest of this project's Redis-backed integration tests already use) running the real task against real Postgres and Qdrant — this proves the actual queue-to-worker-to-persistence path, not just that the classes compile, matching this project's stated reason for testing MAG and CAG against real infrastructure rather than mocks. A cross-tenant and a cross-user status-check denial test, the same shape as `#183`'s and `#193`'s object-level-authorization tests. A body-size-rejection test and a rate-limit test, matching `#193`'s pattern for `/documents` and the session endpoints.
- `tests/integration/test_docker_compose_smoke.py`, if it drives the full stack today, gets the new `worker` service added to what it expects to come up healthy.

## SOLID / GRASP / DRY, made explicit

- **Single Responsibility**: the router only handles HTTP shape; the dispatcher port only defines the async-invocation contract; the Celery adapter only knows how to talk to Celery and Redis; the worker task only knows how to build and call `IngestDataSource` for real; `IngestDataSource` itself is untouched and still knows nothing about any of this.
- **Dependency Inversion**: the router depends on `IngestionJobDispatcher`, an abstraction it owns, not on Celery directly. Swapping Redis for RabbitMQ later, or adding retry policy, touches `CeleryIngestionJobDispatcher` alone — the router and its tests don't change.
- **DRY**: no ingestion logic is duplicated. The worker task is a thin adapter around the exact `IngestDataSource` use case already built, tested, and used by `evaluation/scenarios/freshness-router/run_freshness_measurements.py` and `tests/integration/freshness_env.py`.
- **GRASP Information Expert**: `IngestDataSource` remains the only thing that knows how to route and persist a source. The dispatcher knows only how to run something asynchronously and report on it — it carries no ingestion-specific knowledge, so it wouldn't need to change even if `IngestDataSource`'s own logic did.
- **GRASP Protected Variations**: the port boundary means a second background job type, whenever one is actually needed, reuses this same dispatcher shape without the router's ingestion-specific code changing at all.
- **YAGNI guardrail, stated as a design decision rather than an omission**: no job cancellation, no configurable retry policy beyond Celery's own defaults, no job-history listing. None of these are asked for; `IngestDataSource`'s deterministic source ids already make a failed ingestion safely re-`POST`-able, which covers the retry case that matters today without building a retry system.

## Honestly-disclosed risks

- Celery's default prefork worker pool has known problems running natively on Windows. This design sidesteps the question entirely by only ever running the worker inside the Linux container `docker-compose.yml` defines — never on the Windows host directly — so it's disclosed here rather than left for someone to rediscover as a surprise.
- Redis as the broker means delivery durability rests on Redis's own persistence configuration (AOF/RDB) rather than a broker built around delivery guarantees the way RabbitMQ is. Acceptable for one job type today; named here as the concrete reason a future, higher-stakes job type might justify the RabbitMQ move `docs/architecture/OVERVIEW.md`'s stack table already names.
- No retry-on-failure is configured. A failed ingestion reports `failure` and stops; the caller re-`POST`s to retry, which is safe because ingestion is idempotent by source id.

## Documentation updated in the same pass

`docs/architecture/OVERVIEW.md` (Phase 1 blueprint's `src/workers/` entry moves from "unbuilt" to built; stack table's Task Queue row gets a real citation), `docs/architecture/CONTEXT_GRAPH.md` (the `MOD_ORCH_META` FUTURE node loses the ingestion-endpoint half of its description; a new `REAL` module for `src/workers/` and the extended `src/api/` surface), `docs/security/SECURITY.md` (the new rate limit and body-size entries), `docs/database/DATABASE.md` (the new `ingestion_job:{task_id}` Redis key namespace, alongside the existing documented Redis key shapes), and `CLAUDE.md` (its own "what's still ahead" line drops the ingestion endpoint, keeping streaming, the frontend, and the rest).
