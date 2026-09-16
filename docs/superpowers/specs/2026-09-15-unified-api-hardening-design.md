# Unified API Hardening (#193) Design Spec

**Status:** Approved (design decisions made under the standing full-control grant; recorded here with reasoning rather than left implicit).

**Parent:** Epic #182 (Unified API). **Task:** #193, filed from Story #183's security review.

## Why

Story #183's security review recorded five findings rather than fixing them, because each belongs to the abuse-control or deployment layer rather than the endpoint code itself. This spec designs the fix for each — a real one with a test, or an explicit documentation of the layer that owns it when no such layer exists in this repository yet.

## Non-goals

- No HTTP ingress, TLS termination, or Kubernetes deployment exists in this repository. Nothing in this spec assumes one. Where a control is more naturally an ingress property (a WAF rule, a load-balancer body-size cap), it's built as application middleware instead, since that's the layer this repository actually has, and it's noted that an equivalent ingress control would also satisfy the finding once one exists.
- No CAPTCHA integration. `docs/security/SECURITY.md` already commits to reCAPTCHA v3 or hCaptcha at registration, and integrating a live third-party service (plus the frontend that would render its challenge) is out of scope for a backend-only task. This stays documented as planned, unchanged from its current state.
- No change to the tenant model. A per-tenant quota would be nearly redundant with today's per-user limit, since every registration currently creates its own fresh tenant (documented in `docs/database/DATABASE.md` and confirmed by the live check in #183's plan) — a multi-user tenant role model is explicitly Epic #182's own still-ahead item, not this task's.

## Finding 1 — cost amplification via registration (API6, API4)

**The threat, precisely:** an attacker isn't limited by any one account's 100/minute chat budget, because registration is nearly free (a 5/minute per-IP limit, spreadable across many IPs or simply waited out) and every new account gets its own fresh budget. The per-user limit that already exists caps what one account can spend; nothing caps what an unbounded number of accounts can spend in aggregate.

**Decision: a global chat quota, not a per-tenant one.** A per-tenant quota doesn't close this gap — since one tenant is created per registration today, it just becomes a second name for the same per-account limit the attacker routes around by creating more accounts. A quota shared by every account, regardless of how many exist, is what actually bounds total paid-model spend, which is the resource the finding is about.

**Design:**
- A new Redis-backed fixed-window counter, reusing the existing `RateLimiter` port and `RedisRateLimiter` implementation (already atomic — `incr` + `expire(nx=True)` + `ttl` in one pipeline) rather than inventing a second rate-limiting mechanism.
- Key: a single constant string, `global:chat` — not per-user, per-tenant, or per-IP, since the whole point is one shared counter.
- Window: one hour (`GLOBAL_CHAT_WINDOW_SECONDS`, 3600 seconds). A per-minute global window is too tight for a legitimate multi-tenant deployment with more than a couple of active users; a per-day window reacts too slowly to a burst. An hour is the smallest window that still meaningfully separates "normal aggregate usage" from "someone is trying to run up the bill." Only the limit is env-configurable, matching the existing pattern in this file (`chat_rate_limit()`'s own window is likewise a hardcoded constant passed at its call sites); an operator retuning the window itself needs a code change, not just an environment variable.
- Limit: `CHAT_RATE_LIMIT_GLOBAL_PER_HOUR`, default `10000` — roughly 10x a single user's own theoretical per-hour maximum (100/minute sustained is 6,000/hour), comfortably covering several concurrently active tenants in an early multi-tenant deployment while still bounding aggregate registration-driven cost to a fixed, budgetable number per hour — read per call exactly like `chat_rate_limit()` already reads `CHAT_RATE_LIMIT_PER_MINUTE` — same validation (`ValueError` if below 1), same reason (an operator or a test changes it without controlling import order).
- Enforcement point: both `POST /chat` and `POST /sessions/{session_id}/answers` already call `enforce_rate_limit` once for the per-user limit. Add a second `enforce_rate_limit` call, after the per-user check, for the global limit. Checking per-user first means a single abusive account still gets its own 429 before it can be blamed for exhausting everyone else's shared budget; checking global second means one runaway account can still trip the global limit and start rejecting other tenants' legitimate traffic — an explicit, accepted trade-off of a shared budget over per-tenant isolation, and the reason the global window and limit need to be generous enough for normal multracode use, not tight enough to bite on ordinary traffic.
- `X-RateLimit-*` headers on a 429 reflect whichever limit tripped (the existing exception-carries-its-own-values design in `RateLimitExceeded` already does this correctly with no change needed).

## Finding 2 — no request body size limit (API4, CWE-770)

**The threat:** every route currently reads and JSON-parses (or multipart-parses) the entire request body into memory before any size check runs — the 4,000-character `question` limit is a Pydantic field constraint that only fires after the whole body already exists in memory. A large body costs real memory and CPU on every request, win or lose.

**Decision: an ASGI middleware that buffers the body itself, capped, before the app ever sees it — not `BaseHTTPMiddleware`.** Starlette's `BaseHTTPMiddleware` has known body-consumption quirks when a middleware needs to inspect or replace the body; a raw ASGI middleware class controls the `receive` callable directly, which is what this needs.

**Design (`src/api/middleware/max_body_size.py`):**
- On each HTTP scope, first check the `Content-Length` header if present. If it declares more bytes than the route's limit, respond `413` immediately, without reading anything — the cheap, common case (every real client that isn't deliberately evading detection sends `Content-Length`).
- If `Content-Length` is absent, or the app has no way to trust it, drain the body itself via the ASGI `receive()` callable, chunk by chunk, counting bytes as they arrive. The instant the running total exceeds the limit, respond `413` and stop — the app underneath never receives a byte of the oversized body.
- If the body stays within the limit, replay the fully-buffered body to the wrapped app as a single `http.request` message with `more_body: False`, via a `receive` closure that returns the buffered body once and then falls through to the real `receive()` for anything after (a disconnect message, chiefly). This means the real app still receives exactly the ASGI protocol it expects; the middleware is transparent to everything downstream except an oversized body, which it never lets through at all.
- **Two limits, not one**, because the routes have genuinely different legitimate sizes:
  - A default of 16 KiB for every route. The largest legitimate JSON body today is `AnswerRequest`'s or `ChatRequest`'s 4,000-character question plus routing/session-title overhead — nowhere near 16 KiB, so this is generous headroom for real traffic and still small enough to make an oversized-body attack cheap to reject.
  - `/documents` gets its own override: 11 MiB, just above the existing `_MAX_UPLOAD_BYTES` (10 MiB) in `src/api/routers/documents.py`, so a legitimate maximum-size upload (plus multipart boundary/header overhead) still clears the middleware and reaches the route's own, more precise, per-file check. That existing check stays as defense-in-depth; this task doesn't touch `documents.py`.
- Registered in `src/api/main.py` via `app.add_middleware(MaxBodySizeMiddleware, default_max_bytes=16 * 1024, path_overrides={"/documents": 11 * 1024 * 1024})`.

## Finding 3 — proxy-unaware auth rate-limit keys (API8)

**The threat:** `_enforce_rate_limit` in `src/api/routers/auth.py` keys the 5/minute registration and login limits on `request.client.host`, which is the TCP peer address. Behind any reverse proxy, that's the proxy's own address for every request, so every client sharing that proxy shares one rate-limit bucket — one attacker's failed logins lock out everyone else behind the same proxy.

**Design (`src/api/client_address.py`):**
- `real_client_ip(request: Request) -> str`, using the standard "N trusted hops" model: an operator declares how many reverse proxies sit in front of the app (`TRUSTED_PROXY_COUNT`, default `0`), and only that many entries from the **right-hand end** of `X-Forwarded-For` are trusted — because each hop appends the address it received the request from, so the rightmost `N` entries were appended by hops the operator actually controls, while everything to their left (including the first entry, conventionally "the client") could be forged by the client itself when there's no proxy validating it.
- `TRUSTED_PROXY_COUNT=0` (the default, matching every environment this repository actually runs in today, since none has a reverse proxy in front of it) never reads `X-Forwarded-For` at all and falls straight back to `request.client.host` — today's exact behavior, unchanged, so nothing regresses for a deployment with no proxy. This is the fail-closed default: trusting a client-supplied header by default would be strictly worse than today's bug, not a fix for it.
- With `TRUSTED_PROXY_COUNT=N` (N ≥ 1) and an `X-Forwarded-For` header present with at least `N` comma-separated entries, the real client address is the entry `N` positions from the right. Fewer entries than `N`, a missing header, or an unparseable value all fall back to `request.client.host` — a proxy that isn't holding up its end of the contract shouldn't silently produce a wrong-but-plausible address.
- Read per call, like `chat_rate_limit()` and `_cookie_secure()`, for the same reason: an operator or a test can set it without controlling import order.
- Applied at the one call site that currently uses raw `client.host` for a rate-limit key: `_enforce_rate_limit` in `src/api/routers/auth.py`. No other route keys a limit on IP.

## Finding 4 — no security response headers (API8, A05)

**Design (`src/api/middleware/security_headers.py`):** a small `BaseHTTPMiddleware` (safe here — it only adds response headers after `call_next`, none of the body-handling hazards that ruled it out for Finding 2) that sets exactly the four headers the finding names, on every response:
- `Strict-Transport-Security: max-age=63072000; includeSubDomains` — sent unconditionally. Browsers only act on it over an actual HTTPS connection, so sending it over the plain-HTTP connections this repository's own tests and local development use is inert, not wrong; TLS termination at a future ingress is what makes it take effect, consistent with `SECURITY.md`'s existing framing of TLS as an infrastructure-layer property.
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY` — this is a pure JSON API with no pages of its own to frame, but FastAPI's own `/docs` and `/redoc` are real HTML pages this same app serves, and they're exactly the kind of page clickjacking targets.
- `Referrer-Policy: no-referrer` — the most conservative option; nothing in this API benefits from a referrer leaking to a downstream origin.

No CSP, no `Permissions-Policy`, no other header: the finding names four, and adding more than what a filed, reviewed finding actually asks for is exactly the kind of scope creep this project's own review process would flag.

Registered in `main.py` alongside the other middleware.

## Finding 5 — security events aren't logged (A09)

**Decision: adopt `structlog` now**, rather than stdlib `logging`. `docs/architecture/OVERVIEW.md`'s Observability table and `docs/security/SECURITY.md`'s own auth-event-logging paragraph already commit to structlog as this project's structured-logging library; nothing in the codebase uses it yet. This task is the first real consumer, and using the already-documented choice is simpler than introducing a second, undocumented one.

**Design:**
- Add `structlog>=24.1` to `pyproject.toml`.
- `src/api/security_logging.py`: one small module that configures structlog once (JSON renderer, ISO-8601 timestamps, so log lines are machine-parseable from day one) and exposes `security_logger = structlog.get_logger("security")`.
- **Scope, precisely matching the acceptance criterion:** session authorization denials and 429s. Not the broader auth-event trail (login, logout, token refresh) `SECURITY.md` separately describes as still-ahead — that's a materially larger surface with its own test-per-event-type requirement, and #193's own acceptance criteria name only these two event types.
- **Authorization denial (404 on `SessionNotFound`):** the caller was authenticated (they hold a valid token) but the session they asked for isn't theirs, isn't in their tenant, or doesn't exist. `get_caller` in `src/api/dependencies.py` gains one line — `request.state.caller = caller` — before returning, the same pattern `enforce_rate_limit` already uses (`request.state.rate_limit_headers`) to hand a value from deep in the dependency chain to a handler that runs later. `session_not_found_handler` in `src/api/exception_handlers.py` reads `request.state.caller` (present, since `SessionNotFound` can only be raised after `get_caller` already resolved successfully) and logs `event="authz_denied"`, the caller's `tenant_id` and `user_id`, the denied `session_id` (a UUID identifying which resource was probed — useful for investigation, not sensitive on its own), `path`, and `method`. Never the `Authorization` header, never a question or answer.
- **Rate limit exceeded (429):** `RateLimitExceeded` gains a `key: str` field, set at its one raise site in `enforce_rate_limit` (`src/api/rate_limit.py`) from the `key` parameter already passed in. The key alone (`chat:{user_id}`, `register:{client_ip}`, `sessions:{user_id}`, …) already carries exactly the caller-or-IP identity an investigation needs, with no separate lookup required. `rate_limit_exceeded_handler` logs `event="rate_limit_exceeded"`, `key=exc.key`, `limit=exc.limit`, `path`, and `method`.
- Neither handler logs the request body, so no code path here can ever log `question` text or a token — the design makes that impossible rather than merely avoided by convention.

## Files touched

- `src/api/middleware/max_body_size.py` (new), `src/api/middleware/security_headers.py` (new), `src/api/middleware/__init__.py` (new, empty)
- `src/api/client_address.py` (new)
- `src/api/security_logging.py` (new)
- `src/api/rate_limit.py` (modify: `GLOBAL_CHAT_LIMIT_WINDOW_SECONDS`, `global_chat_limit()`, `RateLimitExceeded.key`)
- `src/api/routers/chat.py`, `src/api/routers/sessions.py` (modify: second `enforce_rate_limit` call for the global chat quota)
- `src/api/routers/auth.py` (modify: `_enforce_rate_limit` uses `real_client_ip`)
- `src/api/dependencies.py` (modify: `get_caller` stashes `request.state.caller`)
- `src/api/exception_handlers.py` (modify: both handlers log)
- `src/api/main.py` (modify: register both new middlewares, configure structlog at startup)
- `pyproject.toml` (modify: add `structlog`)
- `docs/security/SECURITY.md` (modify: record what's now built, matching the acceptance criteria's own "or documented" escape hatch for CAPTCHA specifically)
- Tests: one new unit or integration test per finding, plus updates to any existing test whose behavior this changes (none expected to need it — every change here is additive).

## Verification Matrix

| Finding | Endpoint / Surface | Mechanism | Test |
|---|---|---|---|
| 1 — cost amplification | `POST /chat`, `POST /sessions/{id}/answers` | Global fixed-window Redis counter, `global:chat`, 1h window | Integration: exhaust the global limit (low value via env override) and assert 429 on the next call from a *different* account, proving the limit is shared rather than per-account |
| 1 — CAPTCHA | `POST /auth/register` | Documented as planned in `SECURITY.md` (unchanged) | None (out of scope; no code) |
| 2 — body size | Every route | ASGI middleware, Content-Length fast path + streamed hard cap | Integration: send a body over the default limit to a JSON route and assert 413 before any 422 could fire; a body just over 10 MiB to `/documents` still gets the route's own 413 |
| 3 — proxy-aware keys | `POST /auth/register`, `POST /auth/login` | `real_client_ip` honors `X-Forwarded-For` only up to `TRUSTED_PROXY_COUNT` | Unit: `TRUSTED_PROXY_COUNT=0` ignores a spoofed header; integration: `TRUSTED_PROXY_COUNT=1` derives the trusted entry and two different forwarded IPs get independent rate-limit buckets |
| 4 — security headers | Every route | `SecurityHeadersMiddleware` | Integration: `GET /health` response carries all four headers with the exact values above |
| 5 — security logging | `SessionNotFound`, `RateLimitExceeded` | `structlog` JSON events via the two exception handlers | Unit (with `caplog`/structlog's test helpers): a 404 on another user's session logs `authz_denied` with the caller's identity and no token/question text; a 429 logs `rate_limit_exceeded` with the tripped key |
