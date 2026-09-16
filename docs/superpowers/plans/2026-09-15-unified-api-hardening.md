# Unified API Hardening (#193) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five findings Story #183's security review recorded but didn't fix — cost amplification via unlimited account creation, no request body size limit, proxy-unaware auth rate-limit keys, no security response headers, and unlogged security events — each with a real fix and a test, except CAPTCHA, which stays documented as planned.

**Architecture:**
- `src/api/rate_limit.py` gains a second, global (not per-user, not per-tenant) fixed-window quota on chat and session answers, reusing the existing atomic `RateLimiter` port.
- Two new ASGI/Starlette middlewares — `src/api/middleware/max_body_size.py` (raw ASGI, buffers and caps the body before the app sees it) and `src/api/middleware/security_headers.py` (`BaseHTTPMiddleware`, adds four response headers) — register in `src/api/main.py`.
- `src/api/client_address.py` resolves the real client IP behind a configured number of trusted reverse-proxy hops, replacing the raw `request.client.host` the auth rate limiter uses today.
- `src/api/security_logging.py` configures `structlog` once; `RateLimitExceeded` gains a `key` field and `get_caller` stashes the resolved `Caller` on `request.state`, so both existing exception handlers can log a structured security event without threading new parameters through the routers.

**Tech Stack:** Python 3.11 venv (ruff target py311, mypy strict on `src/`), FastAPI, Starlette middleware (raw ASGI and `BaseHTTPMiddleware`), `structlog` (new dependency), pytest with `asyncio_mode = "auto"`, Redis (TestContainers), httpx `ASGITransport`.

**Spec:** `docs/superpowers/specs/2026-09-15-unified-api-hardening-design.md`

**Issue:** Task #193 under Epic #182. This plan's six tasks together satisfy #193's five acceptance criteria; there are no separate GitHub sub-issues (#193 is itself a single Task, not decomposed further).

## Global Constraints

- **Location.** Work only inside `.worktrees/feature/193-unified-api-hardening/`. Every path below is relative to that directory.
- **Python.** Run it as `../../../.venv/Scripts/python.exe`.
- **Unit tests.** `../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider`. Baseline before this batch: 1171 passed.
- **Integration tests.** `../../../.venv/Scripts/python.exe -m pytest tests/integration/<file> -q -p no:cacheprovider`. They need Docker. Baseline for the whole suite: 277 passed, 8 skipped.
- **Lint and types.** `../../../.venv/Scripts/python.exe -m ruff check <changed files>` and `../../../.venv/Scripts/python.exe -m mypy src`. Line length is 100; wrap any line ruff flags (E501) without changing behaviour.
- **No new database schema.** Every new counter or piece of state in this batch lives in Redis (the existing `RateLimiter` port) or in-memory per-request (`request.state`); nothing here needs an Alembic migration.
- **No ingress, no CAPTCHA, no tenant-role-model work.** Per the spec's non-goals. `docs/security/SECURITY.md`'s CAPTCHA paragraph is read but not changed in a way that claims it's now implemented.
- **Global chat quota.** Key `global:chat` (shared by every caller, not per-user or per-tenant), window 3600 seconds, limit from `CHAT_RATE_LIMIT_GLOBAL_PER_HOUR` (default `1000`, `ValueError` if set below 1) — checked in addition to, and after, the existing per-user `chat:{user_id}` limit on both `POST /chat` and `POST /sessions/{session_id}/answers`.
- **Body size limits.** Default `16 * 1024` bytes for every route; `/documents` gets `11 * 1024 * 1024`. A body over the route's limit gets `413` before the route or Pydantic ever sees it.
- **Trusted proxy hops.** `TRUSTED_PROXY_COUNT` (default `0`) controls how many right-hand entries of `X-Forwarded-For` are trusted; `0` means `X-Forwarded-For` is never read and `request.client.host` is used, unchanged from today.
- **Security headers.** Exactly these four, on every response: `Strict-Transport-Security: max-age=63072000; includeSubDomains`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`.
- **Security event logging.** `structlog`, JSON-rendered. Two event types only: `authz_denied` (on `SessionNotFound`) and `rate_limit_exceeded` (on `RateLimitExceeded`). Never log an `Authorization` header, a token, or request body text (`question`, `title`).
- **Commits.** Messages follow `.gitmessage` (Conventional Commits), reference `#193`, and end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

## File map

| File | Responsibility |
|---|---|
| `src/api/rate_limit.py` (modify) | `GLOBAL_CHAT_WINDOW_SECONDS`, `GLOBAL_CHAT_KEY`, `global_chat_limit()`, `RateLimitExceeded.key` |
| `src/api/routers/chat.py` (modify) | second `enforce_rate_limit` call for the global quota |
| `src/api/routers/sessions.py` (modify) | second `enforce_rate_limit` call for the global quota in `answer()` |
| `src/api/middleware/__init__.py` | empty package marker |
| `src/api/middleware/max_body_size.py` | `MaxBodySizeMiddleware` |
| `src/api/middleware/security_headers.py` | `SecurityHeadersMiddleware` |
| `src/api/client_address.py` | `real_client_ip(request) -> str`, `trusted_proxy_count()` |
| `src/api/routers/auth.py` (modify) | `_enforce_rate_limit` uses `real_client_ip` |
| `src/api/security_logging.py` | `configure_security_logging()`, `security_logger` |
| `src/api/dependencies.py` (modify) | `get_caller` stashes `request.state.caller` |
| `src/api/exception_handlers.py` (modify) | both handlers log a structured event |
| `src/api/main.py` (modify) | registers both new middlewares, calls `configure_security_logging()` |
| `pyproject.toml` (modify) | adds `structlog` |
| `docs/security/SECURITY.md` (modify) | records what's now built |

---

### Task 1: Global chat quota

**Files:**
- Modify: `src/api/rate_limit.py`
- Modify: `src/api/routers/chat.py`
- Modify: `src/api/routers/sessions.py:59-81` (the `answer` route)
- Test: `tests/unit/test_rate_limit.py` (append), `tests/integration/test_sessions_endpoints.py` (append)

**Interfaces:**
- Consumes: `enforce_rate_limit(request, response, *, limiter, key, limit, window_seconds=WINDOW_SECONDS)` (already exists, unchanged signature).
- Produces: `GLOBAL_CHAT_KEY = "global:chat"`, `GLOBAL_CHAT_WINDOW_SECONDS = 3600`, `global_chat_limit() -> int` (reads `CHAT_RATE_LIMIT_GLOBAL_PER_HOUR`, default `1000`, raises `ValueError` if the parsed value is below 1) — the exact shape `chat_rate_limit()` already has, so later tasks and any future caller read it the same way.

- [ ] **Step 1: Write the failing unit tests.** Append to `tests/unit/test_rate_limit.py`:

```python
def test_the_global_chat_limit_defaults_to_1000_and_is_read_from_the_environment_per_call(
    monkeypatch,
):
    monkeypatch.delenv("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", raising=False)
    assert global_chat_limit() == 1000
    monkeypatch.setenv("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", "2")
    assert global_chat_limit() == 2


def test_a_global_chat_limit_below_1_is_refused(monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", "0")
    with pytest.raises(ValueError):
        global_chat_limit()
```

Add `global_chat_limit` to the existing `from src.api.rate_limit import ...` line at the top of the file.

- [ ] **Step 2: Run it and confirm it fails.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_rate_limit.py -q -p no:cacheprovider`. Expected: `ImportError: cannot import name 'global_chat_limit'`.

- [ ] **Step 3: Add the global quota to `rate_limit.py`.** After `_DEFAULT_CHAT_LIMIT = 100` and before `class RateLimitExceeded`, insert:

```python
GLOBAL_CHAT_KEY = "global:chat"  # shared by every caller, not per-user or per-tenant
GLOBAL_CHAT_WINDOW_SECONDS = 3600
_DEFAULT_GLOBAL_CHAT_LIMIT = 1000


def global_chat_limit() -> int:
    """Total requests per hour across every account, shared by POST /chat and answering
    in a session -- the budget that actually bounds paid-model spend when an attacker
    routes around the per-user limit by creating more accounts, since each fresh
    account otherwise gets its own fresh per-user budget. Read per call, like
    chat_rate_limit(), for the same reason.
    """
    limit = int(
        os.environ.get("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", str(_DEFAULT_GLOBAL_CHAT_LIMIT))
    )
    if limit < 1:
        raise ValueError("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR must be at least 1")
    return limit
```

- [ ] **Step 4: Run the unit tests and confirm they pass.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_rate_limit.py -q -p no:cacheprovider`. Expected: all pass.

- [ ] **Step 5: Write the failing integration test.** Append to `tests/integration/test_sessions_endpoints.py`, after `test_the_chat_limit_is_shared_between_answering_and_post_chat`:

```python
async def test_the_global_chat_quota_is_shared_across_every_account(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    os.environ["CHAT_RATE_LIMIT_GLOBAL_PER_HOUR"] = "2"
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    headers_a = _auth(await _user(db_session, tenant_a), tenant_a)
    headers_b = _auth(await _user(db_session, tenant_b), tenant_b)
    await _seed_policy(tenant_a, embedding_model)
    await _seed_policy(tenant_b, embedding_model)
    question = {"question": "What is the return policy?"}
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_a = (await client.post("/sessions", json={}, headers=headers_a)).json()["id"]
        session_b = (await client.post("/sessions", json={}, headers=headers_b)).json()["id"]
        # Two different accounts, well under either one's own 100/minute limit,
        # exhaust the shared global quota of 2 between them.
        first = await client.post(f"/sessions/{session_a}/answers", json=question, headers=headers_a)
        second = await client.post(f"/sessions/{session_b}/answers", json=question, headers=headers_b)
        # A third account's very first request still gets refused: the quota
        # is global, not tied to either account that already used it.
        third = await client.post(f"/sessions/{session_b}/answers", json=question, headers=headers_b)

    assert [first.status_code, second.status_code, third.status_code] == [200, 200, 429]
```

Add a fixture right after `_default_chat_rate_limit` in the same file, so the env var doesn't leak into later tests:

```python
@pytest.fixture(autouse=True)
def _default_global_chat_rate_limit():
    previous = os.environ.pop("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", None)
    yield
    os.environ.pop("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", None)
    if previous is not None:
        os.environ["CHAT_RATE_LIMIT_GLOBAL_PER_HOUR"] = previous
```

- [ ] **Step 6: Run it and confirm it fails.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_sessions_endpoints.py::test_the_global_chat_quota_is_shared_across_every_account -q -p no:cacheprovider`. Expected: FAIL — `second.status_code` (or `third`) is `200`, not the expected refusal, because nothing enforces the global quota yet.

- [ ] **Step 7: Wire the global check into both routes.** In `src/api/routers/chat.py`, change the import line:

```python
from src.api.rate_limit import chat_rate_limit, enforce_rate_limit
```

to:

```python
from src.api.rate_limit import GLOBAL_CHAT_KEY, chat_rate_limit, enforce_rate_limit, global_chat_limit
```

and after the existing `enforce_rate_limit(...)` call for `chat:{caller.user_id}` inside `chat()`, add:

```python
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=GLOBAL_CHAT_KEY,
        limit=global_chat_limit(),
        window_seconds=3600,
    )
```

In `src/api/routers/sessions.py`, change the import line:

```python
from src.api.rate_limit import SESSION_CREATE_LIMIT, chat_rate_limit, enforce_rate_limit
```

to:

```python
from src.api.rate_limit import (
    GLOBAL_CHAT_KEY,
    SESSION_CREATE_LIMIT,
    chat_rate_limit,
    enforce_rate_limit,
    global_chat_limit,
)
```

and inside `answer()`, after its existing `enforce_rate_limit(...)` call for `chat:{caller.user_id}`, add the same second call:

```python
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=GLOBAL_CHAT_KEY,
        limit=global_chat_limit(),
        window_seconds=3600,
    )
```

- [ ] **Step 8: Run the integration test and confirm it passes.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_sessions_endpoints.py -q -p no:cacheprovider`. Expected: all pass, including the new test and every existing test in the file (the default `1000`/hour limit is far above what the rest of the file's tests issue).

- [ ] **Step 9: Lint, type-check, commit.**

```bash
../../../.venv/Scripts/python.exe -m ruff check src/api/rate_limit.py src/api/routers/chat.py src/api/routers/sessions.py tests/unit/test_rate_limit.py tests/integration/test_sessions_endpoints.py
../../../.venv/Scripts/python.exe -m mypy src
git add src/api/rate_limit.py src/api/routers/chat.py src/api/routers/sessions.py tests/unit/test_rate_limit.py tests/integration/test_sessions_endpoints.py
git commit -m "feat(#193): add a global hourly chat quota shared across every account

Cost amplification (API6, API4): the per-user 100/minute limit only
caps one account's spend; an attacker who creates more accounts gets
a fresh budget each time. A second, global fixed-window counter
(global:chat, 1h, CHAT_RATE_LIMIT_GLOBAL_PER_HOUR, default 1000) is
checked after the per-user limit on both POST /chat and answering in
a session, so the two routes' shared per-user budget now has a
shared ceiling too.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Request body size middleware

**Files:**
- Create: `src/api/middleware/__init__.py`
- Create: `src/api/middleware/max_body_size.py`
- Modify: `src/api/main.py`
- Test: `tests/unit/test_max_body_size_middleware.py`, `tests/integration/test_sessions_endpoints.py` (append)

**Interfaces:**
- Produces: `MaxBodySizeMiddleware(app, *, default_max_bytes: int, path_overrides: dict[str, int] | None = None)`, a raw ASGI middleware class (not `BaseHTTPMiddleware`) with `async def __call__(self, scope, receive, send) -> None`.

- [ ] **Step 1: Create the package marker.**

```bash
mkdir -p src/api/middleware
```

Create `src/api/middleware/__init__.py` with no content (an empty file).

- [ ] **Step 2: Write the failing unit tests.** Create `tests/unit/test_max_body_size_middleware.py`:

```python
import json

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.api.middleware.max_body_size import MaxBodySizeMiddleware


async def _echo(request):
    body = await request.body()
    return JSONResponse({"received": len(body)})


def _app(*, default_max_bytes: int, path_overrides: dict[str, int] | None = None) -> Starlette:
    app = Starlette(routes=[Route("/echo", _echo, methods=["POST"])])
    app.add_middleware(
        MaxBodySizeMiddleware, default_max_bytes=default_max_bytes, path_overrides=path_overrides
    )
    return app


def test_a_body_within_the_limit_reaches_the_app_unchanged():
    client = TestClient(_app(default_max_bytes=1024))
    response = client.post("/echo", content=b"x" * 100)
    assert response.status_code == 200
    assert response.json() == {"received": 100}


def test_a_body_over_the_content_length_declared_limit_is_rejected_without_reading_it():
    client = TestClient(_app(default_max_bytes=10))
    response = client.post("/echo", content=b"x" * 1000)
    assert response.status_code == 413


def test_a_chunked_body_over_the_limit_with_no_content_length_is_still_rejected():
    client = TestClient(_app(default_max_bytes=10))

    def _stream():
        for _ in range(5):
            yield b"x" * 5

    response = client.post("/echo", content=_stream())
    assert response.status_code == 413


def test_a_path_override_gets_its_own_limit():
    client = TestClient(_app(default_max_bytes=10, path_overrides={"/echo": 1024}))
    response = client.post("/echo", content=b"x" * 100)
    assert response.status_code == 200
    assert response.json() == {"received": 100}
```

- [ ] **Step 3: Run it and confirm it fails.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_max_body_size_middleware.py -q -p no:cacheprovider`. Expected: `ModuleNotFoundError: No module named 'src.api.middleware.max_body_size'`.

- [ ] **Step 4: Write `src/api/middleware/max_body_size.py`.**

```python
"""Rejects an oversized request body before it ever reaches the app.

Every route today reads and fully parses its body before any size check runs --
Pydantic's own max_length constraints only fire once the whole body is already in
memory (CWE-770). This middleware enforces a byte ceiling itself: on the fast path
(a Content-Length header that already exceeds the limit) it rejects the request
without reading anything at all; on the slow path (no usable Content-Length) it
drains the body itself, bailing the instant the running total crosses the limit, so
the wrapped app never receives an oversized body either way.

A raw ASGI middleware, not starlette.middleware.base.BaseHTTPMiddleware: that base
class has known body-consumption quirks when a middleware needs to inspect or
replace the body, and this needs precise control over the receive() callable.
"""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class MaxBodySizeMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        default_max_bytes: int,
        path_overrides: dict[str, int] | None = None,
    ) -> None:
        self.app = app
        self.default_max_bytes = default_max_bytes
        self.path_overrides = dict(path_overrides or {})

    def _limit_for(self, path: str) -> int:
        for prefix, limit in self.path_overrides.items():
            if path.startswith(prefix):
                return limit
        return self.default_max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self._limit_for(scope["path"])

        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
                if declared is not None and declared > limit:
                    await _reject(scope, receive, send)
                    return
                break

        chunks: list[bytes] = []
        total = 0
        more_body = True
        while more_body:
            message: Message = await receive()
            if message["type"] != "http.request":
                # e.g. a disconnect mid-body: stop collecting and let the real
                # app see it via replay_receive's fallback below.
                break
            body = message.get("body") or b""
            total += len(body)
            if total > limit:
                await _reject(scope, receive, send)
                return
            chunks.append(body)
            more_body = message.get("more_body", False)

        buffered_body = b"".join(chunks)
        sent_once = False

        async def replay_receive() -> Message:
            nonlocal sent_once
            if not sent_once:
                sent_once = True
                return {"type": "http.request", "body": buffered_body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
    response = JSONResponse(status_code=413, content={"detail": "Request body too large"})
    await response(scope, receive, send)
```

- [ ] **Step 5: Run the unit tests and confirm they pass.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_max_body_size_middleware.py -q -p no:cacheprovider`. Expected: all 4 pass.

- [ ] **Step 6: Write the failing integration test.** Append to `tests/integration/test_sessions_endpoints.py`:

```python
async def test_an_oversized_request_body_is_rejected_before_validation_runs(
    app_database_url, redis_url, qdrant_url, embedding_model
):
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        response = await client.post(
            "/sessions", content=json.dumps({"title": "x" * 20_000}).encode(), headers={
                "Content-Type": "application/json", "Authorization": "Bearer not-even-checked",
            }
        )
    assert response.status_code == 413
```

`json` is already imported at the top of this file.

- [ ] **Step 7: Run it and confirm it fails.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_sessions_endpoints.py::test_an_oversized_request_body_is_rejected_before_validation_runs -q -p no:cacheprovider`. Expected: FAIL — the response is `401` (no token check has run yet, but nothing yet caps the body either) or `422`, not `413`.

- [ ] **Step 8: Register the middleware in `src/api/main.py`.** Change the import block:

```python
from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.rate_limit import RateLimitHeadersMiddleware
```

to:

```python
from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.middleware.max_body_size import MaxBodySizeMiddleware
from src.api.rate_limit import RateLimitHeadersMiddleware
```

and after `app.add_middleware(RateLimitHeadersMiddleware)`, add:

```python
app.add_middleware(
    MaxBodySizeMiddleware,
    default_max_bytes=16 * 1024,
    path_overrides={"/documents": 11 * 1024 * 1024},
)
```

- [ ] **Step 9: Run the integration test and confirm it passes.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_sessions_endpoints.py -q -p no:cacheprovider`. Expected: all pass, including this one and every pre-existing test in the file (every legitimate body in this file's other tests is far under 16 KiB).

- [ ] **Step 10: Confirm document upload still works under its own, higher limit.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_documents_endpoints.py -q -p no:cacheprovider`. Expected: all pass, unchanged — the `/documents` override (11 MiB) is comfortably above every upload that file exercises (well under the route's own 10 MiB check).

- [ ] **Step 11: Lint, type-check, commit.**

```bash
../../../.venv/Scripts/python.exe -m ruff check src/api/middleware/max_body_size.py src/api/main.py tests/unit/test_max_body_size_middleware.py tests/integration/test_sessions_endpoints.py
../../../.venv/Scripts/python.exe -m mypy src
git add src/api/middleware/__init__.py src/api/middleware/max_body_size.py src/api/main.py tests/unit/test_max_body_size_middleware.py tests/integration/test_sessions_endpoints.py
git commit -m "feat(#193): reject an oversized request body before it's parsed

No request body size limit (API4, CWE-770): every route read and
parsed the whole body before any size check ran. MaxBodySizeMiddleware
rejects on Content-Length where present, and drains-and-bails on a
chunked body with none, before the app or Pydantic ever sees an
oversized one. 16 KiB by default; /documents keeps its own 11 MiB
ceiling for uploads.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Trusted-proxy client IP for the auth rate limiter

**Files:**
- Create: `src/api/client_address.py`
- Modify: `src/api/routers/auth.py:79-87`
- Test: `tests/unit/test_client_address.py`, `tests/integration/test_auth_endpoints.py` (append)

**Interfaces:**
- Produces: `real_client_ip(request: Request) -> str`.

- [ ] **Step 1: Write the failing unit tests.** Create `tests/unit/test_client_address.py`:

```python
import pytest
from starlette.requests import Request

from src.api.client_address import real_client_ip


def _request(*, client_host: str = "10.0.0.1", forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": (client_host, 12345),
    }
    return Request(scope)


def test_with_no_trusted_proxies_the_header_is_never_read(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_COUNT", raising=False)
    request = _request(client_host="10.0.0.1", forwarded_for="203.0.113.7")
    assert real_client_ip(request) == "10.0.0.1"


def test_with_one_trusted_proxy_the_rightmost_forwarded_entry_is_used(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "1")
    # client, then the one trusted proxy's own upstream address.
    request = _request(client_host="10.0.0.1", forwarded_for="203.0.113.7")
    assert real_client_ip(request) == "203.0.113.7"


def test_with_two_trusted_proxies_the_second_from_the_right_is_used(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "2")
    request = _request(client_host="10.0.0.1", forwarded_for="203.0.113.7, 198.51.100.9")
    assert real_client_ip(request) == "203.0.113.7"


def test_fewer_forwarded_entries_than_trusted_proxies_falls_back_to_client_host(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "3")
    request = _request(client_host="10.0.0.1", forwarded_for="203.0.113.7")
    assert real_client_ip(request) == "10.0.0.1"


def test_a_missing_header_falls_back_to_client_host_even_when_proxies_are_trusted(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "1")
    request = _request(client_host="10.0.0.1", forwarded_for=None)
    assert real_client_ip(request) == "10.0.0.1"


def test_no_client_at_all_returns_unknown(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_COUNT", raising=False)
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "client": None}
    assert real_client_ip(Request(scope)) == "unknown"
```

- [ ] **Step 2: Run it and confirm it fails.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_client_address.py -q -p no:cacheprovider`. Expected: `ModuleNotFoundError: No module named 'src.api.client_address'`.

- [ ] **Step 3: Write `src/api/client_address.py`.**

```python
"""Resolves the real client address behind a configured number of trusted reverse
proxies, so the auth rate limiter's per-IP key isn't the proxy's own address for
every request.

X-Forwarded-For is a client-controlled header by default: anyone can send one. It
only becomes trustworthy for the hops an operator actually put in front of this
app -- each such hop appends the address it received the request from, so exactly
the rightmost TRUSTED_PROXY_COUNT entries were written by infrastructure the
operator controls, and everything to their left (including a first entry someone
might expect to be "the real client") could have been forged by the client itself
before it ever reached the first trusted hop.

TRUSTED_PROXY_COUNT defaults to 0 -- fail closed: no reverse proxy fronts this app
in any environment it runs in today, so the header is never read at all, and
behavior is identical to request.client.host, exactly as before this module
existed.
"""

import os

from fastapi import Request


def trusted_proxy_count() -> int:
    return int(os.environ.get("TRUSTED_PROXY_COUNT", "0"))


def real_client_ip(request: Request) -> str:
    count = trusted_proxy_count()
    if count > 0:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded is not None:
            hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
            if len(hops) >= count:
                return hops[-count]
    return request.client.host if request.client else "unknown"
```

- [ ] **Step 4: Run the unit tests and confirm they pass.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_client_address.py -q -p no:cacheprovider`. Expected: all 6 pass.

- [ ] **Step 5: Write the failing integration test.** Append to `tests/integration/test_auth_endpoints.py` (check the file's existing imports first; it already builds a client via a `_client(...)` helper and issues `/auth/register` or `/auth/login` calls the same way `test_sessions_endpoints.py` does):

```python
async def test_auth_rate_limit_keys_the_trusted_forwarded_address_not_the_proxy(
    app_database_url, redis_url
):
    os.environ["TRUSTED_PROXY_COUNT"] = "1"
    try:
        async with await _client(app_database_url, redis_url) as client:
            # Six requests from one forwarded address exhaust its 5/minute limit --
            # the sixth is refused even though every request arrived from the same
            # httpx test-client "proxy" address, because the trusted header is
            # what keys the limit now, not request.client.host.
            first_address = [
                await client.post(
                    "/auth/login",
                    json={"email": "nobody@example.com", "password": "wrong-password"},
                    headers={"X-Forwarded-For": "203.0.113.7"},
                )
                for _ in range(6)
            ]
            # A different forwarded address gets its own, fresh bucket.
            second_address = await client.post(
                "/auth/login",
                json={"email": "nobody@example.com", "password": "wrong-password"},
                headers={"X-Forwarded-For": "203.0.113.8"},
            )
    finally:
        del os.environ["TRUSTED_PROXY_COUNT"]

    assert [r.status_code for r in first_address] == [401] * 5 + [429]
    assert second_address.status_code == 401
```

This uses the file's existing `_client(app_database_url, redis_url)` helper, the same one every other test in this file already uses — no new client-construction helper needed. Place the new test right after `test_sixth_request_in_a_window_is_rate_limited`, the existing test it sits beside.

- [ ] **Step 6: Run it and confirm it fails.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_auth_endpoints.py::test_auth_rate_limit_keys_the_trusted_forwarded_address_not_the_proxy -q -p no:cacheprovider`. Expected: FAIL — both addresses share one bucket (keyed on the test client's single `request.client.host`), so `second_address.status_code` is `429`, not `401`.

- [ ] **Step 7: Wire it into `auth.py`.** Change the import line:

```python
from src.api.rate_limit import AUTH_LIMIT, enforce_rate_limit
```

to:

```python
from src.api.client_address import real_client_ip
from src.api.rate_limit import AUTH_LIMIT, enforce_rate_limit
```

and change `_enforce_rate_limit`:

```python
async def _enforce_rate_limit(request: Request, response: Response, route_name: str) -> None:
    client_ip = request.client.host if request.client else "unknown"
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=f"{route_name}:{client_ip}",
        limit=AUTH_LIMIT,
    )
```

to:

```python
async def _enforce_rate_limit(request: Request, response: Response, route_name: str) -> None:
    await enforce_rate_limit(
        request,
        response,
        limiter=get_rate_limiter(),
        key=f"{route_name}:{real_client_ip(request)}",
        limit=AUTH_LIMIT,
    )
```

- [ ] **Step 8: Run the integration tests and confirm they pass.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_auth_endpoints.py -q -p no:cacheprovider`. Expected: all pass, including every pre-existing test (they never set `TRUSTED_PROXY_COUNT`, so `real_client_ip` falls back to `request.client.host` exactly as `_enforce_rate_limit` did before).

- [ ] **Step 9: Lint, type-check, commit.**

```bash
../../../.venv/Scripts/python.exe -m ruff check src/api/client_address.py src/api/routers/auth.py tests/unit/test_client_address.py tests/integration/test_auth_endpoints.py
../../../.venv/Scripts/python.exe -m mypy src
git add src/api/client_address.py src/api/routers/auth.py tests/unit/test_client_address.py tests/integration/test_auth_endpoints.py
git commit -m "feat(#193): key auth rate limits on the real client address behind a trusted proxy

Proxy-unaware auth rate-limit keys (API8): _enforce_rate_limit keyed
on request.client.host, so every client behind one reverse proxy
shared a single bucket. real_client_ip() honors X-Forwarded-For only
for the rightmost TRUSTED_PROXY_COUNT entries (default 0, meaning the
header is never read and behavior is unchanged from today) -- an
operator behind N real proxies sets TRUSTED_PROXY_COUNT=N.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Security response headers

**Files:**
- Create: `src/api/middleware/security_headers.py`
- Modify: `src/api/main.py`
- Test: `tests/unit/test_security_headers_middleware.py`, `tests/integration/test_sessions_endpoints.py` (append, or a new small file `tests/integration/test_security_headers.py` — either is fine; this plan uses the latter since the check applies to the whole app, not sessions specifically)

**Interfaces:**
- Produces: `SecurityHeadersMiddleware` (a `starlette.middleware.base.BaseHTTPMiddleware` subclass).

- [ ] **Step 1: Write the failing unit test.** Create `tests/unit/test_security_headers_middleware.py`:

```python
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.api.middleware.security_headers import SecurityHeadersMiddleware


async def _ok(request):
    return PlainTextResponse("ok")


def test_every_response_carries_the_four_security_headers():
    app = Starlette(routes=[Route("/ok", _ok)])
    app.add_middleware(SecurityHeadersMiddleware)
    client = TestClient(app)

    response = client.get("/ok")

    assert response.headers["Strict-Transport-Security"] == "max-age=63072000; includeSubDomains"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
```

- [ ] **Step 2: Run it and confirm it fails.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_security_headers_middleware.py -q -p no:cacheprovider`. Expected: `ModuleNotFoundError: No module named 'src.api.middleware.security_headers'`.

- [ ] **Step 3: Write `src/api/middleware/security_headers.py`.**

```python
"""Adds the security response-header baseline this API is missing (API8, A05):
HSTS, nosniff, a frame-options denial, and a conservative referrer policy.

HSTS is sent unconditionally. A browser only acts on it over an actual HTTPS
connection -- sending it over the plain HTTP this project's own tests and local
development use is inert, not wrong, and TLS termination at a future ingress is
what makes it take effect (docs/security/SECURITY.md already treats TLS as an
infrastructure-layer property this application doesn't itself provide).

Only these four: the finding this fixes names exactly these, and adding a header
nobody asked for is exactly the kind of scope creep a review would flag.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_HEADERS = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.update(_HEADERS)
        return response
```

- [ ] **Step 4: Run the unit test and confirm it passes.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_security_headers_middleware.py -q -p no:cacheprovider`. Expected: pass.

- [ ] **Step 5: Write the failing integration test.** Create `tests/integration/test_security_headers.py`:

```python
import os

from httpx import ASGITransport, AsyncClient


async def _client(app_database_url, redis_url, qdrant_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from src.api.dependencies import get_vector_store
    from src.api.main import app

    await get_vector_store().ensure_collection()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_every_response_carries_the_security_header_baseline(
    app_database_url, redis_url, qdrant_url
):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        response = await client.get("/health")

    assert response.headers["Strict-Transport-Security"] == "max-age=63072000; includeSubDomains"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
```

- [ ] **Step 6: Run it and confirm it fails.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_security_headers.py -q -p no:cacheprovider`. Expected: `KeyError: 'Strict-Transport-Security'`.

- [ ] **Step 7: Register the middleware in `src/api/main.py`.** Add to the import block:

```python
from src.api.middleware.security_headers import SecurityHeadersMiddleware
```

and add, alongside the other two `app.add_middleware(...)` calls:

```python
app.add_middleware(SecurityHeadersMiddleware)
```

- [ ] **Step 8: Run the integration test and confirm it passes.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_security_headers.py -q -p no:cacheprovider`. Expected: pass.

- [ ] **Step 9: Lint, type-check, commit.**

```bash
../../../.venv/Scripts/python.exe -m ruff check src/api/middleware/security_headers.py src/api/main.py tests/unit/test_security_headers_middleware.py tests/integration/test_security_headers.py
../../../.venv/Scripts/python.exe -m mypy src
git add src/api/middleware/security_headers.py src/api/main.py tests/unit/test_security_headers_middleware.py tests/integration/test_security_headers.py
git commit -m "feat(#193): add the HSTS, nosniff, frame-options, and referrer-policy headers

No security response headers (API8, A05). SecurityHeadersMiddleware
adds exactly the four headers the finding names to every response.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Structured security-event logging

**Files:**
- Modify: `pyproject.toml`
- Create: `src/api/security_logging.py`
- Modify: `src/api/rate_limit.py` (`RateLimitExceeded`)
- Modify: `src/api/dependencies.py` (`get_caller`)
- Modify: `src/api/exception_handlers.py` (both handlers)
- Modify: `src/api/main.py`
- Test: `tests/unit/test_security_logging.py`

**Interfaces:**
- Consumes: `Caller` (`src/api/caller.py`, unchanged), `SessionNotFound` (`src/orchestration/domain/errors.py`, unchanged).
- Produces: `configure_security_logging() -> None`, `security_logger` (a `structlog.stdlib.BoundLogger`-compatible logger from `structlog.get_logger("security")`). `RateLimitExceeded.__init__` gains a required `key: str` parameter, so every existing raise site must pass it.

- [ ] **Step 1: Add the dependency.** In `pyproject.toml`, in the `dependencies = [...]` list, add a line (alphabetical-ish placement doesn't matter; put it near the other cross-cutting libraries, e.g. after `"pyyaml>=6.0",`):

```toml
    "structlog>=24.1",
```

Run: `../../../.venv/Scripts/python.exe -m pip install -e .` from the worktree root (installs the new dependency into the shared venv).

- [ ] **Step 2: Write the failing unit tests.** Create `tests/unit/test_security_logging.py`:

```python
import uuid

import structlog

from src.api.caller import Caller
from src.api.exception_handlers import rate_limit_exceeded_handler, session_not_found_handler
from src.api.rate_limit import RateLimitExceeded
from src.api.security_logging import configure_security_logging
from src.orchestration.domain.errors import SessionNotFound


def _request(*, caller: Caller | None = None):
    from starlette.requests import Request

    request = Request({"type": "http", "method": "POST", "path": "/sessions/x/answers", "headers": []})
    if caller is not None:
        request.state.caller = caller
    return request


async def test_a_session_not_found_response_logs_the_callers_identity_and_denied_id():
    configure_security_logging()
    caller = Caller(tenant_id=uuid.uuid4(), user_id=uuid.uuid4())
    session_id = uuid.uuid4()

    with structlog.testing.capture_logs() as captured:
        response = await session_not_found_handler(_request(caller=caller), SessionNotFound(session_id))

    assert response.status_code == 404
    [event] = captured
    assert event["event"] == "authz_denied"
    assert event["tenant_id"] == str(caller.tenant_id)
    assert event["user_id"] == str(caller.user_id)
    assert event["session_id"] == str(session_id)
    assert event["path"] == "/sessions/x/answers"
    assert "question" not in event and "answer" not in event and "authorization" not in event


async def test_a_rate_limit_exceeded_response_logs_which_key_tripped():
    configure_security_logging()
    from datetime import UTC, datetime

    exc = RateLimitExceeded(limit=5, remaining=0, reset_at=datetime.now(UTC), key="register:203.0.113.7")

    with structlog.testing.capture_logs() as captured:
        response = await rate_limit_exceeded_handler(_request(), exc)

    assert response.status_code == 429
    [event] = captured
    assert event["event"] == "rate_limit_exceeded"
    assert event["key"] == "register:203.0.113.7"
    assert event["limit"] == 5
```

- [ ] **Step 3: Run it and confirm it fails.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_security_logging.py -q -p no:cacheprovider`. Expected: `ModuleNotFoundError: No module named 'src.api.security_logging'` (and, once that's created, a `TypeError` for `RateLimitExceeded`'s missing `key` argument — both are expected failures on the way to Step 4).

- [ ] **Step 4: Write `src/api/security_logging.py`.**

```python
"""Configures structlog once, so every part of the API that logs a security event
gets the same JSON-rendered, timestamped output.

Scope, matching #193's own acceptance criteria exactly: session authorization
denials and rate-limit refusals. Not the broader login/logout/token-refresh audit
trail docs/security/SECURITY.md separately describes as still ahead -- that's a
materially larger surface with its own per-event-type test requirement.

Neither event this module's callers log ever includes a token, an Authorization
header, or request body text (a question, an answer, a title): the two call sites
in exception_handlers.py only ever pass a caller's identity, a denied resource id,
a rate-limit key, and request metadata -- fields that were never capable of
carrying that content in the first place.
"""

import structlog


def configure_security_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )


security_logger = structlog.get_logger("security")
```

- [ ] **Step 5: Give `RateLimitExceeded` a `key` field.** In `src/api/rate_limit.py`, change:

```python
class RateLimitExceeded(Exception):
    def __init__(self, limit: int, remaining: int, reset_at: datetime) -> None:
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at
```

to:

```python
class RateLimitExceeded(Exception):
    def __init__(self, limit: int, remaining: int, reset_at: datetime, key: str) -> None:
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at
        self.key = key
```

and its one raise site, inside `enforce_rate_limit`:

```python
        raise RateLimitExceeded(limit=limit, remaining=0, reset_at=reset_at)
```

to:

```python
        raise RateLimitExceeded(limit=limit, remaining=0, reset_at=reset_at, key=key)
```

Exactly one existing test touches `RateLimitExceeded`: `test_a_refused_request_raises_with_the_limit_and_the_reset_time` in `tests/unit/test_rate_limit.py`. It calls `enforce_rate_limit(_request(), Response(), limiter=_FakeLimiter(False, 0), key="k", limit=3)` and asserts on `refused.value.limit`, `.remaining`, `.reset_at` — it never constructs `RateLimitExceeded` directly, so it needs no source change; `key="k"` flows through automatically once `enforce_rate_limit`'s raise site passes `key=key`.

- [ ] **Step 6: Stash the caller on `request.state` in `get_caller`.** In `src/api/dependencies.py`, change:

```python
async def get_caller(claims: dict[str, Any] = Depends(get_current_user_claims)) -> Caller:
    return caller_from_claims(claims)
```

to:

```python
async def get_caller(
    request: Request, claims: dict[str, Any] = Depends(get_current_user_claims)
) -> Caller:
    caller = caller_from_claims(claims)
    # Lets an exception handler running later in the same request (which only
    # ever receives `request` and the raised exception, never a route's own
    # resolved dependencies) recover the authenticated caller's identity for a
    # security-event log -- the same request.state handoff enforce_rate_limit
    # already uses for its own headers.
    request.state.caller = caller
    return caller
```

Add `Request` to the existing `from fastapi import Depends, Header` import line, making it `from fastapi import Depends, Header, Request`.

- [ ] **Step 7: Log from both exception handlers.** In `src/api/exception_handlers.py`, add the import:

```python
from src.api.security_logging import security_logger
```

and change `session_not_found_handler`:

```python
async def session_not_found_handler(request: Request, exc: SessionNotFound) -> JSONResponse:
    # One status and body whether the session is missing, another user's, or another
    # tenant's, so a caller can't learn which session ids exist.
    return JSONResponse(status_code=404, content={"detail": "Session not found"})
```

to:

```python
async def session_not_found_handler(request: Request, exc: SessionNotFound) -> JSONResponse:
    caller = getattr(request.state, "caller", None)
    security_logger.info(
        "authz_denied",
        tenant_id=str(caller.tenant_id) if caller else None,
        user_id=str(caller.user_id) if caller else None,
        session_id=str(exc.session_id),
        path=request.url.path,
        method=request.method,
    )
    # One status and body whether the session is missing, another user's, or another
    # tenant's, so a caller can't learn which session ids exist.
    return JSONResponse(status_code=404, content={"detail": "Session not found"})
```

and change `rate_limit_exceeded_handler`:

```python
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # Reads limit/remaining/reset_at off the exception rather than the response:
    # the raise in enforce_rate_limit happens before any headers are written to
    # the route's injected Response, and FastAPI's exception-handling path builds
    # an entirely new response object for a raised exception, which doesn't
    # inherit anything set on that never-returned Response.
    response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
```

to:

```python
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    security_logger.info(
        "rate_limit_exceeded", key=exc.key, limit=exc.limit, path=request.url.path, method=request.method
    )
    # Reads limit/remaining/reset_at off the exception rather than the response:
    # the raise in enforce_rate_limit happens before any headers are written to
    # the route's injected Response, and FastAPI's exception-handling path builds
    # an entirely new response object for a raised exception, which doesn't
    # inherit anything set on that never-returned Response.
    response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
```

- [ ] **Step 8: Configure logging at startup, in `src/api/main.py`.** Add the import:

```python
from src.api.security_logging import configure_security_logging
```

and call it once, at module scope, right after `app = FastAPI(...)`:

```python
app = FastAPI(title="Unified RAG x CAG x MAG AI System")
configure_security_logging()
register_exception_handlers(app)
```

- [ ] **Step 9: Run the new and existing unit tests and confirm they pass.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_security_logging.py tests/unit/test_rate_limit.py tests/unit/test_caller.py -q -p no:cacheprovider`. Expected: all pass.

- [ ] **Step 10: Run the full unit suite to confirm the `RateLimitExceeded` signature change broke nothing else.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider`. Expected: all pass — `RateLimitExceeded` is only ever constructed at its one raise site in `enforce_rate_limit`, which Step 5 already updated.

- [ ] **Step 11: Run the affected integration suites.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_sessions_endpoints.py tests/integration/test_auth_endpoints.py tests/integration/test_chat_endpoint.py -q -p no:cacheprovider`. Expected: all pass — every `429` and `404` response in these files still gets the same status and body; logging is additive and produces no visible response change.

- [ ] **Step 12: Lint, type-check, commit.**

```bash
../../../.venv/Scripts/python.exe -m ruff check src/api/security_logging.py src/api/rate_limit.py src/api/dependencies.py src/api/exception_handlers.py src/api/main.py tests/unit/test_security_logging.py
../../../.venv/Scripts/python.exe -m mypy src
git add pyproject.toml uv.lock src/api/security_logging.py src/api/rate_limit.py src/api/dependencies.py src/api/exception_handlers.py src/api/main.py tests/unit/test_security_logging.py tests/unit/test_rate_limit.py
git commit -m "feat(#193): log session authorization denials and rate-limit refusals

Security events aren't logged (A09). Adopts structlog, the library
docs/architecture/OVERVIEW.md's Observability table and
docs/security/SECURITY.md's auth-event paragraph already commit to,
as this project's first real consumer of it. get_caller now stashes
the resolved Caller on request.state, letting session_not_found_handler
log the caller's identity and the denied session id as authz_denied;
RateLimitExceeded now carries the key that tripped it, letting
rate_limit_exceeded_handler log it as rate_limit_exceeded. Neither
handler ever sees a token, an Authorization header, or request body
text, so neither can log one.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

If `uv.lock` doesn't exist or isn't tracked in this repository, drop it from the `git add` line — check with `git ls-files uv.lock` before running the commit.

---

### Task 6: Documentation and final verification

**Files:**
- Modify: `docs/security/SECURITY.md`
- Test: none new — this task runs the full suite and closes out the branch.

**Interfaces:**
- Consumes: everything Tasks 1–5 built.
- Produces: nothing new; records what's now built and verifies the whole branch.

- [ ] **Step 1: Update `docs/security/SECURITY.md`.** Find the rate-limiting paragraph (the one describing the fixed window, `chat:{user_id}`, `sessions:{user_id}`, and the auth per-IP limits) and add, in the same connected-prose style as the surrounding paragraph, a sentence recording the global chat quota: what key it uses, its default, and that it closes the registration-driven cost-amplification gap a per-account limit alone can't.

Find the CAPTCHA paragraph (the one already describing reCAPTCHA v3/hCaptcha as planned) and leave its planned-but-unbuilt status exactly as is — don't imply it's now built.

Find (or, if it doesn't exist as its own paragraph, add one near the rate-limiting section) coverage of: the request-body size limit (both the default and the `/documents` override, and that it runs before parsing rather than after), the trusted-proxy client-IP resolution for auth's per-IP limit (default `TRUSTED_PROXY_COUNT=0`, and what an operator sets it to), the four security response headers, and the two structured security-event types now logged (`authz_denied`, `rate_limit_exceeded`) via `structlog` — each with the test file that verifies it, matching how this document already cites test files for its other verified controls.

Reference `#193` as the issue these controls came from, the way the document already references `#193` once (in its current "gaps it didn't fix" sentence) — update that exact sentence, since the gap it names is now closed, to instead describe what got built and where.

Every new sentence traces to a file or a test this plan just created; don't add a claim you can't point at.

- [ ] **Step 2: Run the CI hygiene rehearsal.** From the worktree root:

```bash
grep -rnE '\bTBD\b|\bTODO\b|\bFIXME\b' docs/security/SECURITY.md
grep -rniE 'placeholder' docs/security/SECURITY.md
```

Expected: no output from either command.

- [ ] **Step 3: Commit the documentation.**

```bash
git add docs/security/SECURITY.md
git commit -m "docs(#193): record the Unified API hardening controls now built

Updates the rate-limiting, request-size, and security-event-logging
paragraphs to describe the global chat quota, the body-size
middleware, trusted-proxy client-IP resolution, the security response
headers, and structured authz_denied/rate_limit_exceeded logging --
each cited to the test that verifies it. CAPTCHA stays documented as
planned, unchanged.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Run the full verification pass.**

```bash
../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider
../../../.venv/Scripts/python.exe -m pytest tests/integration -q -p no:cacheprovider
../../../.venv/Scripts/python.exe -m mypy src
../../../.venv/Scripts/python.exe -m ruff check src tests
```

Expected: the unit count is 1171 plus every test this plan added (Task 1: 2 unit + 1 integration; Task 2: 4 unit + 1 integration; Task 3: 6 unit + 1 integration; Task 4: 1 unit + 1 integration; Task 5: 2 unit, no new integration test) — 1186 unit tests — all passing; the integration suite passes at least 277 plus the 4 new integration tests (skips unchanged, since nothing here touches the vLLM- or Ollama-gated tests); mypy and ruff clean.

If any count differs from this arithmetic, work out why before proceeding — don't silently accept a different number without understanding the discrepancy (a fixture collision, a test that got renamed, or an arithmetic slip in this plan itself are the likely causes).

- [ ] **Step 5: Hand off to `superpowers:finishing-a-development-branch`.** Follow that skill: verify tests (already done in Step 4), merge to `develop` locally with a real merge commit (per this project's standing convention and `docs/governance/GIT_WORKFLOW.md`'s settled merge-commit ruling), rerun the unit suite and mypy on the merged result, remove the worktree, delete the branch.

- [ ] **Step 6: Close out the GitHub issue.** With `GH_TOKEN` from `git credential fill` (never printed), from the main checkout after the merge:

```bash
gh issue comment 193 --body "Merged to develop in <merge-sha>. All five findings addressed: a global hourly chat quota closes the registration-driven cost-amplification gap (CAPTCHA stays documented as planned, per SECURITY.md); request bodies are capped before parsing; auth rate limits key on the trusted-proxy-resolved client address; every response carries the HSTS/nosniff/frame-options/referrer-policy baseline; session authorization denials and 429s are logged as structured security events via structlog. <unit count> unit and <integration count> integration tests pass; mypy and ruff are clean. Plan and spec: docs/superpowers/plans/2026-09-15-unified-api-hardening.md, docs/superpowers/specs/2026-09-15-unified-api-hardening-design.md."
gh issue close 193 --reason completed
gh issue comment 182 --body "Task #193 (Unified API hardening) is merged to develop and closed. This epic stays open for the stories still ahead: a freshness-routed ingestion endpoint, which needs a tenant role model, and streaming answers for the frontend."
```

Fill in the actual merge SHA and test counts from Step 4/5's real output before running these commands — don't leave the placeholders in the literal command text.
