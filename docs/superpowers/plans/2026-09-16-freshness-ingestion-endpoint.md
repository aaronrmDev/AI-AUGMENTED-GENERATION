# Freshness-Routed Ingestion Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `IngestDataSource` over HTTP as `POST /data-sources`, backed by this project's first background-worker subsystem (Celery + Redis), with a `GET /data-sources/jobs/{task_id}` status endpoint.

**Architecture:** A thin FastAPI router derives `tenant_id`/`user_id` from the verified `Caller` and hands off to an `IngestionJobDispatcher` port. The real implementation, `CeleryIngestionJobDispatcher`, enqueues a Celery task and records who owns it in Redis; a Celery worker process (new `docker-compose.yml` service) picks the task up, builds a fully-wired `IngestDataSource` with real infrastructure adapters, and runs it. `IngestDataSource` itself is untouched.

**Tech Stack:** FastAPI, Celery 5.4+ with Redis as broker and result backend, Pydantic v2, pytest + TestContainers (Postgres, Qdrant, Redis), Celery's own `celery.contrib.testing.worker.start_worker` for a real in-process worker in integration tests.

**Spec:** `docs/superpowers/specs/2026-09-16-freshness-ingestion-endpoint-design.md`

## Global Constraints

- `tenant_id`/`user_id` are always derived from `Caller` (`src/api/caller.py`), never read from the request body.
- Rate limit: 10 requests/minute per user on `POST /data-sources` (`INGESTION_RATE_LIMIT = 10`, `WINDOW_SECONDS` default from `src/api/rate_limit.py`).
- Body size: `/data-sources` gets the same `11 * 1024 * 1024` (11 MiB) exact-path override `/documents` already has in `MaxBodySizeMiddleware`.
- Job/ownership-record lifetime: 24 hours (`JOB_TTL_SECONDS = 24 * 60 * 60`), used for both the Celery result backend's `result_expires` and the Redis ownership record's TTL, so neither outlives the other.
- Ports are `ABC` with `@abstractmethod`, matching every existing port in this codebase — never `typing.Protocol`.
- Test doubles use the `Fake*` naming already established in `tests/unit/fakes.py` (`FakeUserRepository`, `FakeTokenIssuer`, ...).
- `pyproject.toml` gets `"celery[redis]>=5.4",` added to `dependencies`.
- Any synchronous Celery client call (`.delay()`, `AsyncResult.state`) made from inside an `async def` route or port method runs via `asyncio.to_thread(...)`, matching this project's existing discipline of moving blocking work off the event loop (`UnifiedAnswerQuestion` already embeds its query on a worker thread for the same reason).
- Mypy strict and ruff must stay clean (`uv run mypy src`, `uv run ruff check src tests`) — this codebase's standing bar, not new for this plan.

---

### Task 1: `NullFrozenCache` — a disclosed stand-in for CAG serving that doesn't exist yet

**Files:**
- Create: `src/orchestration/infrastructure/null_frozen_cache.py`
- Test: `tests/unit/test_null_frozen_cache.py`

**Interfaces:**
- Consumes: `src.orchestration.domain.ports.ExpiringCache` (existing ABC — `preload`, `lookup`, `evict`, `contains` from `FrozenCache`; `preload_until`, `renew` from `ExpiringCache`).
- Produces: `class NullFrozenCache(ExpiringCache)` — every method a safe no-op. Task 4's worker composition root constructs one of these directly (`NullFrozenCache()`, no arguments).

**Why this exists:** `IngestDataSource`'s constructor requires a real `ExpiringCache`, and `_apply_route` (`src/orchestration/application/ingest_data_source.py:142-164`) calls `self._cache.evict(...)` and `self._cache.renew(...)` for any source routed `CAG_WITH_RAG_BACKUP`. `docs/architecture/OVERVIEW.md` already discloses that no warmed, GPU-resident frozen cache runs in production yet — the API's own `unified_pipeline.py` composes MAG and RAG tiers only, "since neither exists yet." This class satisfies `IngestDataSource`'s constructor without silently pretending caching works, and without pulling a real transformer model into a background worker for a milestone that never asked for one.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_null_frozen_cache.py
import uuid

from src.orchestration.infrastructure.null_frozen_cache import NullFrozenCache

TENANT = uuid.uuid4()
DOC = uuid.uuid4()


def test_lookup_always_reports_absent():
    cache = NullFrozenCache()
    assert cache.lookup(TENANT, DOC) is None


def test_contains_is_always_false():
    cache = NullFrozenCache()
    assert cache.contains(TENANT, DOC) is False


def test_preload_and_evict_do_not_raise():
    cache = NullFrozenCache()
    cache.preload(TENANT, DOC, "content")
    cache.evict(TENANT, DOC)


def test_preload_until_does_not_raise():
    cache = NullFrozenCache()
    cache.preload_until(TENANT, DOC, "content", expires_at=None)


def test_renew_reports_no_live_entry():
    cache = NullFrozenCache()
    assert cache.renew(TENANT, DOC, expires_at=None) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_null_frozen_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.orchestration.infrastructure.null_frozen_cache'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/orchestration/infrastructure/null_frozen_cache.py
import uuid
from datetime import datetime

from src.orchestration.domain.entities import CacheHit
from src.orchestration.domain.ports import ExpiringCache


class NullFrozenCache(ExpiringCache):
    """A stand-in for a real, GPU-resident CAG cache, which no production process
    in this project serves yet (docs/architecture/OVERVIEW.md discloses this for
    the API's own unified pipeline). Every method is a safe no-op: nothing is ever
    cached, so IngestDataSource's cache-touching code paths (eviction on change,
    renewal on confirmation) do real, harmless nothing until a real cache exists.
    """

    def preload(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        return None

    def lookup(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> CacheHit | None:
        return None

    def evict(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        return None

    def contains(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        return False

    def preload_until(
        self,
        tenant_id: uuid.UUID,
        document_id: uuid.UUID,
        content: str,
        expires_at: datetime | None,
    ) -> None:
        return None

    def renew(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, expires_at: datetime | None
    ) -> bool:
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_null_frozen_cache.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/infrastructure/null_frozen_cache.py tests/unit/test_null_frozen_cache.py
git commit -m "feat(#194): add NullFrozenCache, a disclosed stand-in for unserved CAG"
```

---

### Task 2: The `IngestionJobDispatcher` port, its value types, and its fake

**Files:**
- Modify: `src/orchestration/domain/entities.py` (append `JobState`, `JobStatus`)
- Modify: `src/orchestration/domain/ports.py` (append `IngestionJobDispatcher`)
- Modify: `tests/unit/fakes.py` (append `FakeIngestionJobDispatcher`)
- Test: `tests/unit/test_fake_ingestion_job_dispatcher.py`

**Interfaces:**
- Consumes: `src.orchestration.domain.entities.IngestionResult` (existing: `source_id: uuid.UUID`, `route: IngestionRoute`, `changed: bool`), `DataSourceProfile`.
- Produces: `class JobState(str, Enum)` with members `PENDING`, `SUCCESS`, `FAILURE`; `@dataclass(frozen=True) class JobStatus` with fields `state: JobState`, `result: IngestionResult | None`, `error: str | None`; `class IngestionJobDispatcher(ABC)` with `async def dispatch(self, *, tenant_id: uuid.UUID, profile: DataSourceProfile, content: str, user_id: uuid.UUID | None) -> str` and `async def status(self, task_id: str, *, tenant_id: uuid.UUID, user_id: uuid.UUID) -> JobStatus | None`. `FakeIngestionJobDispatcher` in `tests/unit/fakes.py` implements it. Task 3's `CeleryIngestionJobDispatcher` and Task 6's router both depend on `IngestionJobDispatcher` by this exact signature.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_fake_ingestion_job_dispatcher.py
import uuid
from datetime import timedelta

import pytest

from src.orchestration.domain.entities import DataSourceProfile, JobState, SourceScope
from tests.unit.fakes import FakeIngestionJobDispatcher

TENANT = uuid.uuid4()
OTHER_TENANT = uuid.uuid4()
USER = uuid.uuid4()
OTHER_USER = uuid.uuid4()
PROFILE = DataSourceProfile("return-policy", SourceScope.TENANT, timedelta(days=90))


@pytest.mark.asyncio
async def test_dispatch_then_status_reports_success_with_the_real_result():
    dispatcher = FakeIngestionJobDispatcher()
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=PROFILE, content="policy text", user_id=None
    )
    status = await dispatcher.status(task_id, tenant_id=TENANT, user_id=USER)
    assert status is not None
    assert status.state is JobState.SUCCESS
    assert status.result is not None
    assert status.result.changed is True


@pytest.mark.asyncio
async def test_status_denies_another_tenant():
    dispatcher = FakeIngestionJobDispatcher()
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=PROFILE, content="policy text", user_id=None
    )
    assert await dispatcher.status(task_id, tenant_id=OTHER_TENANT, user_id=USER) is None


@pytest.mark.asyncio
async def test_status_denies_another_user_for_a_user_scoped_job():
    dispatcher = FakeIngestionJobDispatcher()
    user_profile = DataSourceProfile("shoe-size", SourceScope.USER, timedelta(days=30))
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=user_profile, content="size 10", user_id=USER
    )
    assert await dispatcher.status(task_id, tenant_id=TENANT, user_id=OTHER_USER) is None


@pytest.mark.asyncio
async def test_status_is_none_for_an_unknown_task_id():
    dispatcher = FakeIngestionJobDispatcher()
    assert await dispatcher.status("no-such-task", tenant_id=TENANT, user_id=USER) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_fake_ingestion_job_dispatcher.py -v`
Expected: FAIL with `ImportError: cannot import name 'JobState'` (or `FakeIngestionJobDispatcher` missing)

- [ ] **Step 3: Write minimal implementation**

Append to `src/orchestration/domain/entities.py`:

```python
class JobState(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class JobStatus:
    state: JobState
    result: IngestionResult | None
    error: str | None
```

(`Enum` and `dataclass`/`field` are already imported at the top of this file for the existing `SourceScope`/`IngestionRoute`/`DataSource` definitions — no new imports needed.)

Append to `src/orchestration/domain/ports.py`:

```python
class IngestionJobDispatcher(ABC):
    """Runs an ingestion asynchronously and reports on it. IngestDataSource never
    depends on this -- it has no idea whether it's being run synchronously in a
    test, from the evaluation harness, or inside a Celery task, and this port is
    what keeps it that way."""

    @abstractmethod
    async def dispatch(
        self,
        *,
        tenant_id: uuid.UUID,
        profile: DataSourceProfile,
        content: str,
        user_id: uuid.UUID | None,
    ) -> str:
        """Enqueues the ingestion and returns a task id the caller can poll."""

    @abstractmethod
    async def status(
        self, task_id: str, *, tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> JobStatus | None:
        """None means the task id doesn't exist or doesn't belong to this caller --
        the router maps that to the same 404 shape used everywhere else in this API."""
```

Add `JobStatus` to the existing `from src.orchestration.domain.entities import (...)` block at the top of `ports.py` (alongside `DataSource`, `DataSourceProfile`, etc. already imported there).

Append to `tests/unit/fakes.py`:

```python
import asyncio

from src.orchestration.domain.entities import IngestionResult, JobState, JobStatus
from src.orchestration.domain.ports import IngestionJobDispatcher


class FakeIngestionJobDispatcher(IngestionJobDispatcher):
    """Runs the 'ingestion' synchronously and in-memory -- no Celery, no Redis.
    Stores just enough to answer status() and to enforce the same ownership rule
    CeleryIngestionJobDispatcher enforces for real."""

    def __init__(self) -> None:
        self._jobs: dict[str, tuple[uuid.UUID, uuid.UUID | None, JobStatus]] = {}

    async def dispatch(
        self,
        *,
        tenant_id: uuid.UUID,
        profile: DataSourceProfile,
        content: str,
        user_id: uuid.UUID | None,
    ) -> str:
        task_id = str(uuid.uuid4())
        # A real ingestion isn't run here -- this fake exists to exercise the
        # port's contract (dispatch/status/ownership), not IngestDataSource's own
        # behavior, which already has its own tests. changed=True is a fixed,
        # arbitrary stand-in result.
        result = IngestionResult(source_id=uuid.uuid4(), route=None, changed=True)  # type: ignore[arg-type]
        self._jobs[task_id] = (tenant_id, user_id, JobStatus(JobState.SUCCESS, result, None))
        return task_id

    async def status(
        self, task_id: str, *, tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> JobStatus | None:
        entry = self._jobs.get(task_id)
        if entry is None:
            return None
        owner_tenant, owner_user, status = entry
        if owner_tenant != tenant_id:
            return None
        if owner_user is not None and owner_user != user_id:
            return None
        return status
```

(`import uuid` and `from src.orchestration.domain.entities import DataSourceProfile` are already present in this file's existing header per the earlier read of its imports; add `JobState`, `JobStatus`, `IngestionResult` to that same import line rather than a second one.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_fake_ingestion_job_dispatcher.py -v`
Expected: 4 passed

- [ ] **Step 5: Fix the `route=None` type-ignore**

The `# type: ignore[arg-type]` above is a real wart, not acceptable to leave under this codebase's `mypy --strict` bar. Fix it properly: give `FakeIngestionJobDispatcher` a real `IngestionRoute` instead of `None`.

```python
        from src.orchestration.domain.entities import IngestionRoute

        result = IngestionResult(source_id=uuid.uuid4(), route=IngestionRoute.RAG_ONLY, changed=True)
```

Remove the `# type: ignore[arg-type]` comment now that the value is well-typed.

- [ ] **Step 6: Run mypy and the test again**

Run: `uv run mypy src && uv run pytest tests/unit/test_fake_ingestion_job_dispatcher.py -v`
Expected: mypy reports no issues; 4 passed

- [ ] **Step 7: Commit**

```bash
git add src/orchestration/domain/entities.py src/orchestration/domain/ports.py tests/unit/fakes.py tests/unit/test_fake_ingestion_job_dispatcher.py
git commit -m "feat(#194): add the IngestionJobDispatcher port and its fake"
```

---

### Task 3: Celery app and `CeleryIngestionJobDispatcher`

**Files:**
- Create: `src/workers/__init__.py`
- Create: `src/workers/celery_app.py`
- Create: `src/workers/celery_ingestion_dispatcher.py`
- Modify: `pyproject.toml` (add `"celery[redis]>=5.4",` to `dependencies`)
- Test: `tests/integration/test_celery_ingestion_dispatcher.py`

**Interfaces:**
- Consumes: `src.orchestration.domain.ports.IngestionJobDispatcher` (Task 2), `src.orchestration.domain.entities.DataSourceProfile/JobState/JobStatus` (Task 2), the `redis_url` TestContainers fixture already in `tests/integration/conftest.py`.
- Produces: `celery_app: Celery` (module-level instance in `src/workers/celery_app.py`, `JOB_TTL_SECONDS: int` constant in the same module); `class CeleryIngestionJobDispatcher(IngestionJobDispatcher)` in `src/workers/celery_ingestion_dispatcher.py`, constructed as `CeleryIngestionJobDispatcher(celery_app, redis_client)`. Task 4's worker task registers against this same `celery_app`. Task 5's `get_ingestion_job_dispatcher()` in `src/api/dependencies.py` constructs one of these.

`celery_app.py` dispatches by task *name* (`"ingest_data_source"`), not by importing the task function directly -- Task 4's `ingestion_worker.py` needs Postgres/Qdrant/Neo4j infrastructure imports that only the worker process should ever load; the API process (which only ever calls `dispatch()`/`status()`) must not import that module. `include=["src.workers.ingestion_worker"]` on the `Celery(...)` constructor is what makes the actual worker process (started via the CLI, never via the API) find and register the task -- the API process never triggers that import at all, since it only ever calls `celery_app.send_task(...)` and `AsyncResult(...)`, neither of which imports task modules.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_celery_ingestion_dispatcher.py
"""CeleryIngestionJobDispatcher against a real Redis instance. No worker runs
here -- that's Task 4's ingestion_worker test and Task 6's full-stack test.
This proves dispatch() actually enqueues onto Redis and writes the ownership
record, and status() reads both back correctly, including denying the wrong
caller -- by priming the Celery result backend directly, the same way Celery's
own test suite proves its result-backend read path without a worker."""
import os
import uuid
from datetime import timedelta

import pytest
import redis.asyncio as redis

from src.orchestration.domain.entities import DataSourceProfile, JobState, SourceScope

TENANT = uuid.uuid4()
OTHER_TENANT = uuid.uuid4()
USER = uuid.uuid4()
PROFILE = DataSourceProfile("return-policy", SourceScope.TENANT, timedelta(days=90))


@pytest.fixture
async def redis_client(redis_url: str):
    client = redis.from_url(redis_url)
    yield client
    await client.aclose()


@pytest.fixture
def dispatcher(redis_url: str, redis_client: redis.Redis):
    # src.workers.celery_app reads REDIS_URL at *import* time (the same
    # module-level-singleton pattern src/api/dependencies.py already uses for
    # APP_DATABASE_URL/QDRANT_URL) -- setting the env var here, before the
    # deferred import below, matches how tests/integration/test_sessions_
    # endpoints.py's _client() helper already handles the identical ordering
    # requirement for src.api.main. A module-level top-of-file import would
    # run before this fixture (or any fixture) executes, and crash with
    # KeyError: 'REDIS_URL' the moment pytest collects this file.
    os.environ["REDIS_URL"] = redis_url
    from src.workers.celery_app import build_celery_app
    from src.workers.celery_ingestion_dispatcher import CeleryIngestionJobDispatcher

    return CeleryIngestionJobDispatcher(build_celery_app(redis_url), redis_client)


@pytest.mark.asyncio
async def test_dispatch_writes_an_ownership_record_and_enqueues_the_task(dispatcher, redis_client):
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=PROFILE, content="policy text", user_id=None
    )
    stored = await redis_client.get(f"ingestion_job:{task_id}")
    assert stored is not None
    assert stored.decode() == f"{TENANT}:"


@pytest.mark.asyncio
async def test_status_reports_pending_for_a_task_with_no_result_yet(dispatcher):
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=PROFILE, content="policy text", user_id=None
    )
    status = await dispatcher.status(task_id, tenant_id=TENANT, user_id=USER)
    assert status is not None
    assert status.state is JobState.PENDING


@pytest.mark.asyncio
async def test_status_denies_another_tenant(dispatcher):
    task_id = await dispatcher.dispatch(
        tenant_id=TENANT, profile=PROFILE, content="policy text", user_id=None
    )
    assert await dispatcher.status(task_id, tenant_id=OTHER_TENANT, user_id=USER) is None


@pytest.mark.asyncio
async def test_status_is_none_for_an_unknown_task_id(dispatcher):
    assert await dispatcher.status(str(uuid.uuid4()), tenant_id=TENANT, user_id=USER) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_celery_ingestion_dispatcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.workers'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/workers/__init__.py
```//empty package marker -- write it as a literally empty file, no content

```python
# src/workers/celery_app.py
from celery import Celery

# Both the Celery result backend and the ownership record CeleryIngestionJobDispatcher
# writes to Redis expire on this same schedule, so neither ever outlives the other.
JOB_TTL_SECONDS = 24 * 60 * 60


def build_celery_app(redis_url: str) -> Celery:
    app = Celery(
        "unified_ai_workers",
        broker=redis_url,
        backend=redis_url,
        # The worker process's own entry point imports this module by name
        # (`celery -A src.workers.celery_app:celery_app worker`) and needs the
        # task registered; the API process, which only ever calls send_task()
        # and AsyncResult() by name, never imports ingestion_worker at all.
        # Keeping that import here rather than at this module's top level is
        # what keeps Postgres/Qdrant/Neo4j infrastructure imports out of the
        # API process's import graph entirely.
        include=["src.workers.ingestion_worker"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        result_expires=JOB_TTL_SECONDS,
        task_track_started=True,
    )
    return app


# The module-level instance both the API process and the `celery` CLI import.
# REDIS_URL is read once, at import time, matching the existing
# _engine/_sessionmaker singleton pattern in src/api/dependencies.py.
import os  # noqa: E402

celery_app = build_celery_app(os.environ["REDIS_URL"])
```

```python
# src/workers/celery_ingestion_dispatcher.py
import asyncio
import uuid

import redis.asyncio as redis
from celery import Celery
from celery.result import AsyncResult

from src.orchestration.domain.entities import DataSourceProfile, IngestionResult, JobState, JobStatus
from src.orchestration.domain.ports import IngestionJobDispatcher
from src.workers.celery_app import JOB_TTL_SECONDS

_OWNERSHIP_KEY_PREFIX = "ingestion_job:"


class CeleryIngestionJobDispatcher(IngestionJobDispatcher):
    """The real IngestionJobDispatcher: Celery for the work, Redis for who owns it.

    The ownership record is checked in status() *before* Celery's result backend is
    ever read -- a task id belonging to another tenant or user reports back exactly
    as if it never existed, the same "missing or not yours" 404 shape
    AnswerInSession already uses for sessions.
    """

    def __init__(self, celery_app: Celery, redis_client: redis.Redis) -> None:
        self._celery_app = celery_app
        self._redis = redis_client

    async def dispatch(
        self,
        *,
        tenant_id: uuid.UUID,
        profile: DataSourceProfile,
        content: str,
        user_id: uuid.UUID | None,
    ) -> str:
        def _send() -> str:
            result = self._celery_app.send_task(
                "ingest_data_source",
                kwargs={
                    "tenant_id": str(tenant_id),
                    "source_key": profile.source_key,
                    "scope": profile.scope.value,
                    "expected_change_interval_seconds": int(
                        profile.expected_change_interval.total_seconds()
                    ),
                    "content": content,
                    "user_id": str(user_id) if user_id is not None else None,
                },
            )
            return str(result.id)

        task_id = await asyncio.to_thread(_send)
        owner = f"{tenant_id}:{user_id if user_id is not None else ''}"
        await self._redis.set(f"{_OWNERSHIP_KEY_PREFIX}{task_id}", owner, ex=JOB_TTL_SECONDS)
        return task_id

    async def status(
        self, task_id: str, *, tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> JobStatus | None:
        owner = await self._redis.get(f"{_OWNERSHIP_KEY_PREFIX}{task_id}")
        if owner is None:
            return None
        owner_tenant_str, _, owner_user_str = owner.decode().partition(":")
        if uuid.UUID(owner_tenant_str) != tenant_id:
            return None
        if owner_user_str and uuid.UUID(owner_user_str) != user_id:
            return None

        def _read() -> JobStatus:
            async_result: AsyncResult = self._celery_app.AsyncResult(task_id)
            if async_result.state in ("PENDING", "STARTED", "RETRY"):
                return JobStatus(JobState.PENDING, None, None)
            if async_result.state == "SUCCESS":
                raw = async_result.result
                return JobStatus(
                    JobState.SUCCESS,
                    IngestionResult(
                        source_id=uuid.UUID(raw["source_id"]),
                        route=raw["route"],
                        changed=raw["changed"],
                    ),
                    None,
                )
            # FAILURE, or any other terminal-but-not-SUCCESS state.
            return JobStatus(JobState.FAILURE, None, str(async_result.result))

        return await asyncio.to_thread(_read)
```

Note: `IngestionResult.route` is typed `IngestionRoute`, not `str` -- `raw["route"]` above needs converting via `IngestionRoute(raw["route"])`, since Task 4's task returns the route as its `.value` string for JSON serialization (Celery's JSON serializer can't carry an `Enum` member directly). Fix this now:

```python
                        route=IngestionRoute(raw["route"]),
```

with `IngestionRoute` added to this file's import line from `src.orchestration.domain.entities`.

Add `"celery[redis]>=5.4",` to `pyproject.toml`'s `dependencies` list (after `"redis>=5.0",`, since Celery's own Redis transport needs the extra), then run `uv sync --extra dev` so the environment picks it up.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv sync --extra dev && uv run pytest tests/integration/test_celery_ingestion_dispatcher.py -v`
Expected: 4 passed

- [ ] **Step 5: Run mypy**

Run: `uv run mypy src`
Expected: no issues (add `types-redis` to `pyproject.toml`'s `dev` extra first if mypy reports missing stubs for `redis.asyncio` -- check whether `redis`'s own inline types already satisfy strict mode before reaching for a stub package, since `RedisRateLimiter` already imports `redis.asyncio` today with no stub package installed).

- [ ] **Step 6: Commit**

```bash
git add src/workers/__init__.py src/workers/celery_app.py src/workers/celery_ingestion_dispatcher.py pyproject.toml uv.lock tests/integration/test_celery_ingestion_dispatcher.py
git commit -m "feat(#194): add the Celery app and CeleryIngestionJobDispatcher"
```

---

### Task 4: The worker's composition root — `ingest_data_source_task`

**Files:**
- Create: `src/workers/ingestion_worker.py`
- Test: `tests/integration/test_ingestion_worker.py`

**Interfaces:**
- Consumes: `src.workers.celery_app.celery_app` (Task 3), `NullFrozenCache` (Task 1), `IngestDataSource` and its existing infrastructure adapters exactly as `tests/integration/freshness_env.py` wires them (`PostgresDataSourceRepository`, `ChunkedRagIndex`, `CacheWarmedRetrieve`, `RecordSemanticFactWriter`, `SearchDocuments`, `FixedSizeChunker`, `CachingEmbeddingModel`, `SentenceTransformersEmbedder`, `QdrantVectorStore`, `QdrantSemanticMemoryIndex`, `Neo4jMemoryGraphRepository`).
- Produces: `@celery_app.task(name="ingest_data_source") def ingest_data_source_task(*, tenant_id: str, source_key: str, scope: str, expected_change_interval_seconds: int, content: str, user_id: str | None) -> dict[str, object]`, returning `{"source_id": str, "route": str, "changed": bool}` (JSON-serializable, matching what Task 3's `CeleryIngestionJobDispatcher.status()` reads back). Task 6's router never imports this module directly -- only `celery_app`'s `include=` (Task 3) does, at real-worker-process startup.

This is the worker process's own composition root -- the one place that builds a *real*, fully-wired `IngestDataSource`, the same way `src/api/dependencies.py` is the API process's composition root. Reads the same environment variables the API process already reads (`APP_DATABASE_URL`, `QDRANT_URL`, `NEO4J_URL` if that's how Neo4j's connection is configured -- check `tests/integration/freshness_env.py`'s `neo4j_url` fixture shape and `src/api/dependencies.py` for whether a `NEO4J_URL`-equivalent environment variable already exists anywhere in this codebase; if not, this task adds one, matching the `APP_DATABASE_URL`/`QDRANT_URL` naming convention).

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_ingestion_worker.py
"""ingest_data_source_task against real Postgres and Qdrant, run synchronously via
Celery's own .apply() (no broker, no worker process) -- this proves the task's
composition root wires IngestDataSource correctly, independent of Task 6's
full HTTP-to-real-worker proof."""
import uuid

import pytest


@pytest.mark.asyncio
async def test_a_new_tenant_scoped_source_is_ingested_and_returns_its_route(
    app_database_url, qdrant_url, redis_url, neo4j_url, monkeypatch
):
    monkeypatch.setenv("APP_DATABASE_URL", app_database_url)
    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    # ingestion_worker imports src.workers.celery_app, which reads REDIS_URL at
    # *import* time -- set before the deferred import below, same reasoning as
    # Task 3's dispatcher fixture and tests/integration/test_sessions_
    # endpoints.py's _client() helper for APP_DATABASE_URL/QDRANT_URL. A
    # module-top-level `from src.workers.ingestion_worker import
    # ingest_data_source_task` would run at collection time, before this test
    # function (or any fixture) ever executes, and crash with KeyError.
    monkeypatch.setenv("REDIS_URL", redis_url)
    url, username, password = neo4j_url
    monkeypatch.setenv("NEO4J_URL", url)
    monkeypatch.setenv("NEO4J_USERNAME", username)
    monkeypatch.setenv("NEO4J_PASSWORD", password)

    from src.workers.ingestion_worker import ingest_data_source_task

    tenant_id = uuid.uuid4()
    result = ingest_data_source_task.apply(
        kwargs={
            "tenant_id": str(tenant_id),
            "source_key": "return-policy",
            "scope": "tenant",
            "expected_change_interval_seconds": 90 * 24 * 60 * 60,
            "content": "Our return policy allows returns within 90 days.",
            "user_id": None,
        }
    ).get()

    assert result["changed"] is True
    assert result["route"] in ("rag_only", "cag_with_rag_backup")
    uuid.UUID(result["source_id"])  # does not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_ingestion_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.workers.ingestion_worker'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/workers/ingestion_worker.py
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from src.identity.infrastructure.db import get_engine, get_sessionmaker
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.ingest_data_source import IngestDataSource
from src.orchestration.domain.entities import DataSourceProfile, SourceScope
from src.orchestration.infrastructure.chunked_rag_index import ChunkedRagIndex
from src.orchestration.infrastructure.null_frozen_cache import NullFrozenCache
from src.orchestration.infrastructure.postgres_data_source_repository import (
    PostgresDataSourceRepository,
)
from src.orchestration.infrastructure.record_semantic_fact_writer import RecordSemanticFactWriter
from src.rag.application.search_documents import SearchDocuments
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.workers.celery_app import celery_app

# CAG's own similarity threshold for a warmed-cache hit -- CacheWarmedRetrieve's
# constructor requires one, but with NullFrozenCache backing it, nothing this
# task does ever produces a hit for the threshold to apply to.
_UNUSED_CAG_THRESHOLD = 0.9


def _build_ingest_data_source() -> IngestDataSource:
    engine = get_engine(os.environ["APP_DATABASE_URL"])
    sessionmaker = get_sessionmaker(engine)
    embedder = CachingEmbeddingModel(SentenceTransformersEmbedder())
    vector_store = QdrantVectorStore(os.environ["QDRANT_URL"])
    semantic_index = QdrantSemanticMemoryIndex(os.environ["QDRANT_URL"])
    graph = Neo4jMemoryGraphRepository(
        os.environ["NEO4J_URL"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )

    repository = PostgresDataSourceRepository(sessionmaker)
    rag_index = ChunkedRagIndex(sessionmaker, vector_store, FixedSizeChunker(), embedder)
    cache = NullFrozenCache()
    search = SearchDocuments(embedder, vector_store)
    warmed = CacheWarmedRetrieve(embedder, cache, search, similarity_threshold=_UNUSED_CAG_THRESHOLD)
    writer = RecordSemanticFactWriter(sessionmaker, semantic_index, embedder, graph)
    return IngestDataSource(repository, rag_index, cache, warmed, writer)


@celery_app.task(name="ingest_data_source")
def ingest_data_source_task(
    *,
    tenant_id: str,
    source_key: str,
    scope: str,
    expected_change_interval_seconds: int,
    content: str,
    user_id: str | None,
) -> dict[str, Any]:
    import asyncio

    async def _run() -> dict[str, Any]:
        ingest = _build_ingest_data_source()
        profile = DataSourceProfile(
            source_key, SourceScope(scope), timedelta(seconds=expected_change_interval_seconds)
        )
        result = await ingest.execute(
            uuid.UUID(tenant_id),
            profile,
            content,
            datetime.now(UTC),
            user_id=uuid.UUID(user_id) if user_id is not None else None,
        )
        return {
            "source_id": str(result.source_id),
            "route": result.route.value,
            "changed": result.changed,
        }

    # A Celery worker process has no already-running event loop the way uvicorn
    # does -- asyncio.run is the bridge from this synchronous task body to
    # IngestDataSource.execute, which is async.
    return asyncio.run(_run())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_ingestion_worker.py -v`
Expected: 1 passed

- [ ] **Step 5: Run mypy**

Run: `uv run mypy src`
Expected: no issues

- [ ] **Step 6: Commit**

```bash
git add src/workers/ingestion_worker.py tests/integration/test_ingestion_worker.py
git commit -m "feat(#194): add the worker's IngestDataSource composition root"
```

---

### Task 5: Request/response schemas

**Files:**
- Create: `src/api/schemas/data_sources.py`
- Test: `tests/unit/test_data_source_schemas.py`

**Interfaces:**
- Consumes: `src.orchestration.domain.entities.IngestionResult`, `JobState`.
- Produces: `class IngestDataSourceRequest(BaseModel)` (`source_key: str`, `scope: Literal["tenant", "user"]`, `expected_change_interval_seconds: int`, `content: str`), `class IngestDataSourceAcceptedResponse(BaseModel)` (`task_id: str`), `class IngestionResultSchema(BaseModel)` (`source_id: uuid.UUID`, `route: str`, `changed: bool`) with `IngestionResultSchema.of(result: IngestionResult) -> IngestionResultSchema`, `class JobStatusResponse(BaseModel)` (`state: Literal["pending", "success", "failure"]`, `result: IngestionResultSchema | None`, `error: str | None`) with `JobStatusResponse.of(status: JobStatus) -> JobStatusResponse`. Task 6's router constructs and returns these directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_data_source_schemas.py
import uuid

import pytest
from pydantic import ValidationError

from src.orchestration.domain.entities import IngestionResult, IngestionRoute, JobState, JobStatus
from src.api.schemas.data_sources import (
    IngestDataSourceRequest,
    IngestionResultSchema,
    JobStatusResponse,
)


def test_a_valid_request_parses():
    request = IngestDataSourceRequest(
        source_key="return-policy",
        scope="tenant",
        expected_change_interval_seconds=90 * 24 * 60 * 60,
        content="Our return policy allows returns within 90 days.",
    )
    assert request.scope == "tenant"


def test_a_non_positive_interval_is_rejected():
    with pytest.raises(ValidationError):
        IngestDataSourceRequest(
            source_key="return-policy", scope="tenant",
            expected_change_interval_seconds=0, content="text",
        )


def test_blank_content_is_rejected():
    with pytest.raises(ValidationError):
        IngestDataSourceRequest(
            source_key="return-policy", scope="tenant",
            expected_change_interval_seconds=3600, content="   ",
        )


def test_an_unknown_field_is_refused_not_ignored():
    with pytest.raises(ValidationError):
        IngestDataSourceRequest(
            source_key="return-policy", scope="tenant",
            expected_change_interval_seconds=3600, content="text",
            tenant_id=str(uuid.uuid4()),
        )


def test_job_status_response_of_a_success_carries_the_result():
    result = IngestionResult(source_id=uuid.uuid4(), route=IngestionRoute.RAG_ONLY, changed=True)
    response = JobStatusResponse.of(JobStatus(JobState.SUCCESS, result, None))
    assert response.state == "success"
    assert response.result is not None
    assert response.result.route == "rag_only"
    assert response.error is None


def test_job_status_response_of_a_failure_carries_the_error():
    response = JobStatusResponse.of(JobStatus(JobState.FAILURE, None, "a tenant-scoped source cannot route to MAG"))
    assert response.state == "failure"
    assert response.result is None
    assert response.error == "a tenant-scoped source cannot route to MAG"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_data_source_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.schemas.data_sources'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/api/schemas/data_sources.py
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.orchestration.domain.entities import IngestionResult, JobState, JobStatus

MAX_CONTENT_CHARS = 2_000_000  # generous text ceiling; MaxBodySizeMiddleware's 11 MiB
                                 # override on this route is the real backstop


class IngestDataSourceRequest(BaseModel):
    # Unknown fields are refused, not ignored: tenant_id/user_id can't be named
    # here -- both come only from the verified Caller, never the request body.
    model_config = ConfigDict(extra="forbid")

    source_key: str = Field(..., min_length=1)
    scope: Literal["tenant", "user"]
    expected_change_interval_seconds: int = Field(..., gt=0)
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_CHARS)

    @field_validator("source_key", "content")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class IngestDataSourceAcceptedResponse(BaseModel):
    task_id: str


class IngestionResultSchema(BaseModel):
    source_id: uuid.UUID
    route: str
    changed: bool

    @classmethod
    def of(cls, result: IngestionResult) -> IngestionResultSchema:
        return cls(source_id=result.source_id, route=result.route.value, changed=result.changed)


class JobStatusResponse(BaseModel):
    state: Literal["pending", "success", "failure"]
    result: IngestionResultSchema | None
    error: str | None

    @classmethod
    def of(cls, status: JobStatus) -> JobStatusResponse:
        return cls(
            state=status.state.value,  # type: ignore[arg-type]
            result=None if status.result is None else IngestionResultSchema.of(status.result),
            error=status.error,
        )
```

`status.state.value` is a plain `str` at runtime (`JobState` is a `str, Enum`), and the `Literal["pending", "success", "failure"]` field accepts it structurally -- but mypy strict doesn't narrow a bare `str` into a `Literal` automatically. Fix properly rather than leaving the `# type: ignore`:

```python
        state: Literal["pending", "success", "failure"] = status.state.value  # type: ignore[assignment]
```

is still not clean. Use `cast` instead, since the value is genuinely guaranteed correct by `JobState`'s own three members:

```python
from typing import Literal, cast
...
            state=cast(Literal["pending", "success", "failure"], status.state.value),
```

Remove the `# type: ignore[arg-type]` comment now that `cast` makes the narrowing explicit rather than silenced.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_data_source_schemas.py -v`
Expected: 6 passed

- [ ] **Step 5: Run mypy**

Run: `uv run mypy src`
Expected: no issues

- [ ] **Step 6: Commit**

```bash
git add src/api/schemas/data_sources.py tests/unit/test_data_source_schemas.py
git commit -m "feat(#194): add data-source ingestion request/response schemas"
```

---

### Task 6: The router, dependency wiring, and the object-level-authorization 404

**Files:**
- Create: `src/api/routers/data_sources.py`
- Modify: `src/api/dependencies.py` (add `get_ingestion_job_dispatcher`)
- Modify: `src/api/rate_limit.py` (add `INGESTION_RATE_LIMIT`)
- Modify: `src/orchestration/domain/errors.py` (add `IngestionJobNotFound`)
- Modify: `src/api/exception_handlers.py` (register a handler for it)
- Modify: `src/api/main.py` (register the router; extend `MaxBodySizeMiddleware`'s `path_overrides`)
- Test: `tests/integration/test_data_sources_endpoints.py`

**Interfaces:**
- Consumes: `Caller`/`get_caller` (existing), `enforce_rate_limit` (existing), `IngestionJobDispatcher`/`get_ingestion_job_dispatcher` (this task, backed by Task 3's `CeleryIngestionJobDispatcher`), `IngestDataSourceRequest`/`IngestDataSourceAcceptedResponse`/`JobStatusResponse` (Task 5).
- Produces: `POST /data-sources` (201... actually 202, see below), `GET /data-sources/jobs/{task_id}`. Nothing downstream depends on this router's own internals; it's the outermost layer.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_data_sources_endpoints.py
"""Full-stack proof: a real Celery worker (Celery's own start_worker test
helper), consuming from the real TestContainers Redis instance, actually runs
IngestDataSource against real Postgres and Qdrant -- reached entirely through
HTTP, the same way every other route in this API is proven."""
import os
import time
import uuid

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="module")
def celery_worker_for_ingestion(redis_url, app_database_url, qdrant_url, neo4j_url):
    os.environ["REDIS_URL"] = redis_url
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["QDRANT_URL"] = qdrant_url
    url, username, password = neo4j_url
    os.environ["NEO4J_URL"] = url
    os.environ["NEO4J_USERNAME"] = username
    os.environ["NEO4J_PASSWORD"] = password

    # Deferred import, not module-top-level: src.workers.celery_app reads
    # REDIS_URL at import time (see the identical note on Task 3's dispatcher
    # fixture), and this module is also imported transitively by src.api.main
    # (via src.api.dependencies, Task 6's own change) -- a top-level import
    # here would run at collection time, before any env var above is set.
    from celery.contrib.testing.worker import start_worker

    from src.workers.celery_app import celery_app as _celery_app

    with start_worker(_celery_app, pool="solo", perform_ping_check=False):
        yield


async def _register_and_login(client: AsyncClient) -> str:
    email = f"ingest-{uuid.uuid4().hex[:8]}@example.com"
    password = "hunter2hunter2"
    await client.post("/auth/register", json={"email": email, "password": password})
    login = await client.post("/auth/login", json={"email": email, "password": password})
    return str(login.json()["access_token"])


@pytest.mark.asyncio
async def test_post_then_poll_reaches_success(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}

        post_response = await client.post(
            "/data-sources",
            json={
                "source_key": "return-policy",
                "scope": "tenant",
                "expected_change_interval_seconds": 90 * 24 * 60 * 60,
                "content": "Our return policy allows returns within 90 days.",
            },
            headers=headers,
        )
        assert post_response.status_code == 202
        task_id = post_response.json()["task_id"]

        deadline = time.monotonic() + 30
        status_body = None
        while time.monotonic() < deadline:
            status_response = await client.get(f"/data-sources/jobs/{task_id}", headers=headers)
            assert status_response.status_code == 200
            status_body = status_response.json()
            if status_body["state"] != "pending":
                break
            time.sleep(0.5)
        assert status_body is not None
        assert status_body["state"] == "success"
        assert status_body["result"]["changed"] is True


@pytest.mark.asyncio
async def test_a_client_cannot_declare_its_own_tenant_id(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.post(
            "/data-sources",
            json={
                "source_key": "return-policy",
                "scope": "tenant",
                "expected_change_interval_seconds": 3600,
                "content": "text",
                "tenant_id": str(uuid.uuid4()),
            },
            headers=headers,
        )
        assert response.status_code == 422  # extra="forbid" rejects the field outright


@pytest.mark.asyncio
async def test_checking_another_users_job_returns_404(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token_a = await _register_and_login(client)
        token_b = await _register_and_login(client)

        post_response = await client.post(
            "/data-sources",
            json={
                "source_key": "return-policy",
                "scope": "tenant",
                "expected_change_interval_seconds": 3600,
                "content": "text",
            },
            headers={"Authorization": f"Bearer {token_a}"},
        )
        task_id = post_response.json()["task_id"]

        cross_response = await client.get(
            f"/data-sources/jobs/{task_id}", headers={"Authorization": f"Bearer {token_b}"}
        )
        assert cross_response.status_code == 404


@pytest.mark.asyncio
async def test_checking_an_unknown_job_id_returns_404(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await _register_and_login(client)
        response = await client.get(
            f"/data-sources/jobs/{uuid.uuid4()}", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_ingestion_is_rate_limited_at_ten_per_minute(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "source_key": "return-policy",
            "scope": "tenant",
            "expected_change_interval_seconds": 3600,
            "content": "text",
        }
        responses = [await client.post("/data-sources", json=payload, headers=headers) for _ in range(11)]
        assert responses[-1].status_code == 429


@pytest.mark.asyncio
async def test_an_oversized_body_is_rejected_before_parsing(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}", "Content-Length": str(12 * 1024 * 1024)}
        oversized = "x" * (12 * 1024 * 1024)
        response = await client.post("/data-sources", content=oversized, headers=headers)
        assert response.status_code == 413
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_data_sources_endpoints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.routers.data_sources'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/orchestration/domain/errors.py`:

```python
class IngestionJobNotFound(Exception):
    """A task id doesn't exist, or belongs to another tenant or user."""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"no ingestion job {task_id} is visible under the current tenant")
        self.task_id = task_id
```

Add to `src/api/rate_limit.py` (alongside `SESSION_CREATE_LIMIT`):

```python
INGESTION_RATE_LIMIT = 10  # per user per window, on POST /data-sources
```

Add to `src/api/dependencies.py`'s top-level imports:

```python
import redis.asyncio as redis

from src.orchestration.domain.ports import IngestionJobDispatcher
```

(`IngestionJobDispatcher` is a plain ABC with no side effects -- safe to import eagerly, unlike the two classes below.)

Add the factory functions themselves further down, alongside `get_rate_limiter`/`get_refresh_token_store`:

```python
@functools.cache
def get_ingestion_redis_client() -> redis.Redis:
    return redis.from_url(os.environ["REDIS_URL"])


@functools.cache
def get_ingestion_job_dispatcher() -> IngestionJobDispatcher:
    # Deferred, not a top-level import: src.workers.celery_app reads REDIS_URL
    # at *import* time (the same module-level-singleton pattern this file's
    # own _engine/_sessionmaker already uses for APP_DATABASE_URL). This
    # module is imported by every test that imports src.api.main -- including
    # ones that never touch ingestion and never set REDIS_URL before doing so
    # -- so making this a top-level import here would turn REDIS_URL into a
    # hard import-time requirement for the whole file, breaking any such test
    # at collection time. get_rate_limiter/get_refresh_token_store below
    # already avoid exactly this by reading os.environ["REDIS_URL"] lazily,
    # inside their own function bodies, not at module import time; this
    # follows the same discipline for the two new classes it needs.
    from src.workers.celery_app import celery_app
    from src.workers.celery_ingestion_dispatcher import CeleryIngestionJobDispatcher

    return CeleryIngestionJobDispatcher(celery_app, get_ingestion_redis_client())
```

`close_redis_clients()` (already in this file) also needs to close `get_ingestion_redis_client()` if it was ever built, matching exactly how it already handles `get_rate_limiter()`/`get_refresh_token_store()`:

```python
    if get_ingestion_redis_client.cache_info().currsize:
        clients.append(get_ingestion_redis_client())
    get_ingestion_redis_client.cache_clear()
```

(added to the existing `clients: list[...]` collection and the existing `.cache_clear()` calls -- widen that list's type annotation to include `redis.Redis` alongside `RedisRateLimiter | RedisRefreshTokenStore`.)

```python
# src/api/routers/data_sources.py
from fastapi import APIRouter, Depends, Request, Response

from src.api.caller import Caller
from src.api.dependencies import get_caller, get_ingestion_job_dispatcher, get_rate_limiter
from src.api.rate_limit import INGESTION_RATE_LIMIT, enforce_rate_limit
from src.api.schemas.data_sources import (
    IngestDataSourceAcceptedResponse,
    IngestDataSourceRequest,
    JobStatusResponse,
)
from src.orchestration.domain.entities import DataSourceProfile, SourceScope
from src.orchestration.domain.errors import IngestionJobNotFound
from src.orchestration.domain.ports import IngestionJobDispatcher
from datetime import timedelta

router = APIRouter(prefix="/data-sources", tags=["data-sources"])


@router.post("", response_model=IngestDataSourceAcceptedResponse, status_code=202)
async def ingest(
    payload: IngestDataSourceRequest,
    request: Request,
    response: Response,
    caller: Caller = Depends(get_caller),
    dispatcher: IngestionJobDispatcher = Depends(get_ingestion_job_dispatcher),
) -> IngestDataSourceAcceptedResponse:
    await enforce_rate_limit(
        request, response,
        limiter=get_rate_limiter(), key=f"data-sources:{caller.user_id}", limit=INGESTION_RATE_LIMIT,
    )
    profile = DataSourceProfile(
        payload.source_key,
        SourceScope(payload.scope),
        timedelta(seconds=payload.expected_change_interval_seconds),
    )
    task_id = await dispatcher.dispatch(
        tenant_id=caller.tenant_id,
        profile=profile,
        content=payload.content,
        user_id=caller.user_id if payload.scope == "user" else None,
    )
    return IngestDataSourceAcceptedResponse(task_id=task_id)


@router.get("/jobs/{task_id}", response_model=JobStatusResponse)
async def job_status(
    task_id: str,
    caller: Caller = Depends(get_caller),
    dispatcher: IngestionJobDispatcher = Depends(get_ingestion_job_dispatcher),
) -> JobStatusResponse:
    status = await dispatcher.status(task_id, tenant_id=caller.tenant_id, user_id=caller.user_id)
    if status is None:
        raise IngestionJobNotFound(task_id)
    return JobStatusResponse.of(status)
```

Add to `src/api/exception_handlers.py`:

```python
from src.orchestration.domain.errors import IngestionJobNotFound  # add to the existing import line


async def ingestion_job_not_found_handler(
    request: Request, exc: IngestionJobNotFound
) -> JSONResponse:
    caller = getattr(request.state, "caller", None)
    security_logger.info(
        "authz_denied",
        tenant_id=str(caller.tenant_id) if caller else None,
        user_id=str(caller.user_id) if caller else None,
        task_id=exc.task_id,
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(status_code=404, content={"detail": "Ingestion job not found"})
```

and register it in `register_exception_handlers`:

```python
    app.add_exception_handler(IngestionJobNotFound, ingestion_job_not_found_handler)  # type: ignore[arg-type]
```

Modify `src/api/main.py`:

```python
from src.api.routers.data_sources import router as data_sources_router
```

```python
app.add_middleware(
    MaxBodySizeMiddleware,
    default_max_bytes=16 * 1024,
    path_overrides={"/documents": 11 * 1024 * 1024, "/data-sources": 11 * 1024 * 1024},
)
```

```python
app.include_router(data_sources_router)
```

(alongside the existing `app.include_router(...)` calls.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_data_sources_endpoints.py -v`
Expected: 6 passed (allow extra time -- this test starts a real Celery worker thread and polls; if it's flaky on a slow machine, raise the 30s deadline in the test rather than the route's own behavior)

- [ ] **Step 5: Run mypy and ruff**

Run: `uv run mypy src && uv run ruff check src tests`
Expected: no issues

- [ ] **Step 6: Commit**

```bash
git add src/api/routers/data_sources.py src/api/dependencies.py src/api/rate_limit.py src/orchestration/domain/errors.py src/api/exception_handlers.py src/api/main.py tests/integration/test_data_sources_endpoints.py
git commit -m "feat(#194): add POST /data-sources and GET /data-sources/jobs/{id}"
```

---

### Task 7: Docker Compose, and documentation updated in the same pass

**Files:**
- Modify: `docker/docker-compose.yml` (new `worker` service)
- Modify: `docs/architecture/OVERVIEW.md`
- Modify: `docs/architecture/CONTEXT_GRAPH.md`
- Modify: `docs/security/SECURITY.md`
- Modify: `docs/database/DATABASE.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: nothing new (infra/docs only).
- Produces: nothing later tasks depend on -- this is the closing task.

- [ ] **Step 1: Add the `worker` service to `docker-compose.yml`**

```yaml
  worker:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    environment:
      APP_DATABASE_URL: postgresql+asyncpg://app_user:${APP_DB_PASSWORD}@postgres:5432/unified_ai
      APP_DB_PASSWORD: ${APP_DB_PASSWORD}
      REDIS_URL: redis://redis:6379/0
      QDRANT_URL: http://qdrant:6333
      NEO4J_URL: bolt://neo4j:7687
      NEO4J_USERNAME: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD:-unified_ai_neo4j}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
      neo4j:
        condition: service_healthy
    command: uv run celery -A src.workers.celery_app:celery_app worker --loglevel=info
```

(Reuses `Dockerfile.api`'s existing image rather than a new `Dockerfile.worker` -- the worker's dependencies are identical to the API's today; a dedicated Dockerfile stays named-but-deferred in the Phase 1 blueprint until the two processes' dependencies actually diverge.)

- [ ] **Step 2: Validate the compose file parses**

Run: `docker compose -f docker/docker-compose.yml config --quiet`
Expected: no output, exit code 0

- [ ] **Step 3: Update `docs/architecture/OVERVIEW.md`**

In the Phase 1 module blueprint's closing paragraph, replace the sentence naming the ingestion endpoint and `src/workers/` as unbuilt:

> What genuinely remains unbuilt, stated plainly rather than left to this paragraph's own out-of-date framing: a scheduler to drive the router's refresh and review, and everything past Phase 1 -- the frontend, and the Kubernetes/production-Docker layer.

(drops "an HTTP endpoint for freshness-routed ingestion" and "`src/workers/`" from what's named unbuilt, since both now exist; the scheduler for `RefreshCachedSources`/`ReviewSourceFreshness` remains genuinely unbuilt and stays named).

Add a new subsection after the Phase 1 blueprint's tree diagram, naming what actually exists under `src/workers/` now: `celery_app.py` (the Celery application, Redis broker and result backend) and `ingestion_worker.py` (`ingest_data_source_task`, the worker process's own `IngestDataSource` composition root) -- both real, both tested against a real Celery worker consuming from a real broker, per `tests/integration/test_data_sources_endpoints.py`.

- [ ] **Step 4: Update `docs/architecture/CONTEXT_GRAPH.md`**

Move `MOD_ORCH_META` (the ingestion-endpoint half of it) and `MOD_WORKERS` out of the `FUTURE` subgraph into a `REAL` one, citing `src/api/routers/data_sources.py`, `src/orchestration/domain/ports.py`'s `IngestionJobDispatcher`, `src/workers/celery_app.py`, and `src/workers/ingestion_worker.py`. Update "What isn't here yet" to drop the ingestion endpoint and worker layer from the list of genuinely-unbuilt things, keeping the frontend and Kubernetes/production-deployment layer.

- [ ] **Step 5: Update `docs/security/SECURITY.md`**

In the rate-limiting section's bulleted list of what's built (`5 requests per minute...`, `100 per minute...`, `20 per minute...`), add:

> - 10 per minute per user, on `POST /data-sources`.

In the object-level-authorization section (the one describing `POST /sessions/{session_id}/answers`'s ownership check), add a parallel bullet for `GET /data-sources/jobs/{task_id}`, describing the same "missing or not yours" 404 shape enforced via the Redis ownership record `CeleryIngestionJobDispatcher` checks before ever reading Celery's result backend.

- [ ] **Step 6: Update `docs/database/DATABASE.md`**

Add the new Redis key shape to wherever this document already lists Redis key namespaces (`tenant:{tenant_id}:` prefixes, rate-limit keys, etc.): `ingestion_job:{task_id}` -> `"{tenant_id}:{user_id or ''}"`, TTL 24 hours, written by `CeleryIngestionJobDispatcher.dispatch()` and read by its `status()`.

- [ ] **Step 7: Update `CLAUDE.md`**

In the opening paragraph's "What's still ahead" sentence, drop "an HTTP endpoint for freshness-routed ingestion, which needs a tenant role model and its own authorization review" (built now) from the list, keeping "the frontend, background workers, and Kubernetes deployment layers" -- except "background workers" is no longer accurate either, since `src/workers/` now has real, tested code. Replace with: "and the frontend and Kubernetes deployment layers" (background workers dropped from the still-ahead list; a scheduler to drive the freshness router's own refresh/review cycle remains a separate, smaller gap, not the same thing as "workers" existing at all).

- [ ] **Step 8: Run the full suite one more time**

Run: `uv run pytest tests/unit -q && uv run pytest tests/integration -q`
Expected: all unit tests pass; the integration suite passes (the 8 GPU-only skips and any Ollama-only skip are expected, matching this project's documented skip pattern)

- [ ] **Step 9: Commit**

```bash
git add docker/docker-compose.yml docs/architecture/OVERVIEW.md docs/architecture/CONTEXT_GRAPH.md docs/security/SECURITY.md docs/database/DATABASE.md CLAUDE.md
git commit -m "docs(#194): record the ingestion endpoint and worker subsystem now built"
```
