# Auth Foundation — Design Spec

## Why this is a sub-project on its own

`docs/architecture/OVERVIEW.md`'s Phase 1 blueprint scopes the repository's first slice of code as "a working API with basic RAG" — an upload endpoint, chunking, embeddings, a vector search endpoint, and a chat endpoint. Layering `docs/security/SECURITY.md`'s auth requirements (JWT issuance, Argon2id hashing, row-level tenant isolation, rate limiting) on top of that roughly doubles the slice, and the two halves are naturally independent: nothing about issuing a JWT depends on chunking a document, and nothing about vector search depends on how a password got hashed. This spec covers only the first half — a running, tested, protected API with no paradigm logic behind it yet — so it can be built, reviewed, and merged before the RAG pipeline sub-project starts on top of it. It is also, concretely, the first code this repository will contain, which means the choices below aren't only about auth; they're the pattern every later module inherits.

## Scope

In scope: user registration, login, access/refresh token issuance, refresh rotation, logout, the `Users`/`Sessions` Alembic migration with row-level security, Redis-backed rate limiting on the auth endpoints, and the Docker Compose stack needed to run all of that locally. Out of scope, deferred to later work and named here so the boundary is explicit rather than implied: OAuth2/OIDC single sign-on, CAPTCHA/bot protection, TLS termination and ingress (SECURITY.md itself treats that as infrastructure to verify once something is deployed, not application code), and anything paradigm-specific — no `rag/`, `cag/`, `mag/`, or `orchestration/` code, no Qdrant, no Neo4j, no vLLM. The `Sessions` table is created by this sub-project's migration because it lives in the same `Users`-adjacent group in `docs/database/DATABASE.md`'s schema, but nothing here populates it with real conversation data — that starts in the RAG sub-project.

## Module layout

`OVERVIEW.md`'s Phase 1 tree names `rag/`, `cag/`, `mag/`, `orchestration/`, and `workers/` — one directory per paradigm plus the background-job layer. Auth isn't a paradigm, so it needs a home the existing tree doesn't provide. Two new top-level modules are added, following the domain/application/infrastructure split CLAUDE.md's Hexagonal Architecture rule requires project-wide and ADR-0004 already establishes as the pattern for MAG:

```text
src/
├── api/
│   ├── main.py              # FastAPI app factory, router mounting, middleware wiring
│   ├── dependencies.py      # get_current_user, get_db_session (RLS-scoped), rate-limit dependency
│   ├── routers/
│   │   └── auth.py          # POST /auth/register, /auth/login, /auth/refresh, /auth/logout
│   └── exception_handlers.py
│
├── identity/
│   ├── domain/
│   │   ├── entities.py      # User, RefreshToken value objects — no framework imports
│   │   ├── ports.py         # PasswordHasher, TokenIssuer, UserRepository, RefreshTokenStore, RateLimiter (all ABCs)
│   │   └── errors.py        # InvalidCredentials, TokenExpired, TokenAlreadyUsed, etc.
│   ├── application/
│   │   ├── register_user.py
│   │   ├── authenticate_user.py
│   │   ├── refresh_access_token.py
│   │   └── revoke_refresh_token.py
│   └── infrastructure/
│       ├── postgres_user_repository.py   # SQLAlchemy, implements UserRepository
│       ├── argon2_password_hasher.py     # implements PasswordHasher
│       ├── jwt_token_issuer.py           # implements TokenIssuer
│       ├── redis_refresh_token_store.py  # implements RefreshTokenStore
│       └── redis_rate_limiter.py         # implements RateLimiter
│
alembic/
├── env.py
└── versions/
    └── <rev>_users_sessions.py   # Users, Sessions tables + RLS policies

docker/
├── Dockerfile.api
└── docker-compose.yml            # api, postgres (pgvector image), redis

tests/
├── unit/
│   ├── test_identity_domain.py
│   └── test_identity_application.py   # use cases against fake in-memory repositories
├── integration/
│   ├── test_auth_endpoints.py         # TestContainers Postgres + Redis, real HTTP flow
│   ├── test_rls_tenant_isolation.py   # the exact cross-tenant test SECURITY.md specifies
│   └── test_rate_limiting.py
└── conftest.py
```

`src/identity/domain/` never imports FastAPI, SQLAlchemy, or Redis — it defines what a `User` is and what a `PasswordHasher` must be able to do, nothing about how. `src/identity/application/` orchestrates those domain objects against the port interfaces, so a use case like `AuthenticateUser` can be unit-tested against a fake repository with no database at all. `src/identity/infrastructure/` is the only layer that imports `argon2`, `sqlalchemy`, or `redis` directly. `src/api/` is the driving adapter — it depends on `identity/application/`, never the other way around.

The Postgres image is `pgvector/pgvector:pg16` even though nothing in this sub-project's schema uses a vector column — `Users` and `Sessions` have none. Provisioning the extension now avoids a second image swap and a second `CREATE EXTENSION` migration when the RAG sub-project's `Documents`/`Chunks` tables need it on the same instance.

## Data model

Both tables come directly from `docs/database/DATABASE.md`'s schema table, unchanged:

| Table | Column | Notes |
|---|---|---|
| Users | id | UUID, primary key |
| Users | email | unique, not null |
| Users | hashed_password | Argon2id hash, never the raw password |
| Users | tenant_id | tenant-scoping root |
| Users | created_at | — |
| Users | updated_at | — |
| Sessions | id | UUID, primary key |
| Sessions | user_id | foreign key → Users |
| Sessions | tenant_id | — |
| Sessions | title | — |
| Sessions | context_budget | JSONB; unused until the RAG/orchestration sub-project writes to it |
| Sessions | created_at | — |

Refresh tokens are **not** a table. Ruling, made without a human available to confirm: DATABASE.md states PostgreSQL holds exactly seven tables, and adding an eighth for token revocation would contradict that documented count for a need Redis already covers — a revocable, expiring key is exactly Redis's stated role. Refresh tokens live at `identity:refresh:{token_id}` → `{"user_id": ..., "expires_at": ...}`, TTL 7 days, matching the key-pattern convention DATABASE.md already documents for Redis (`session:{session_id}:working_memory`, `user:{user_id}:preferences`). Deleting the key on rotation or logout is what makes reuse of an already-rotated token fail — the key is gone, so the lookup finds nothing and the request is rejected — which is the exact behavior SECURITY.md's replay test checks for.

Both `Users` and `Sessions` get a row-level security policy: `CREATE POLICY tenant_isolation ON <table> USING (tenant_id = current_setting('app.current_tenant_id')::uuid)`. The FastAPI dependency that opens a DB session for a request issues `SET LOCAL app.current_tenant_id = '<uuid>'` as the first statement in that transaction, scoped from the authenticated user's own `tenant_id` — never from a client-supplied value, since trusting a request body or header for tenant scoping would make the RLS policy decorative rather than load-bearing.

> **Corrected during implementation.** Three details of the paragraph above did not survive contact with PostgreSQL, and what shipped differs. The full reasoning is in the plan's Task 5 section (`docs/superpowers/plans/2026-08-22-auth-foundation.md`); the short version:
>
> - **RLS landed on `Sessions` only, not on `Users`.** `Users` is the tenant-identity root rather than tenant-scoped data, and a tenant policy on it makes the two flows that have to run *before* a tenant is known structurally impossible: login looks a user up by email across all tenants precisely to discover which tenant they belong to, and registration inserts a row for a tenant that does not exist yet. `Sessions` is where this project's row-level tenant isolation actually lives, and `tests/integration/test_rls_tenant_isolation.py` proves it there.
> - **`current_setting(..., true)`, with the second argument.** Without `true` (the `missing_ok` flag), `current_setting` raises rather than returning NULL when the setting has never been assigned on that connection — so any query on an unscoped session would error out instead of simply matching no rows.
> - **`set_config('app.current_tenant_id', $1, true)`, not `SET LOCAL`.** PostgreSQL's `SET`/`SET LOCAL` grammar accepts only a literal, never a bound parameter, and asyncpg always binds server-side via the extended query protocol — so Postgres sees a literal `$1` where it requires a constant and raises a syntax error. `set_config()` is an ordinary function, so it takes a normal bound parameter; its third argument `true` gives it `SET LOCAL`'s transaction-local scope. Interpolating the UUID into the SQL text instead would have worked, but only by reintroducing exactly the string-built SQL this project's own rules forbid. See `src/identity/infrastructure/db.py`.

## Request flows

**Register** (`POST /auth/register`): a Pydantic `RegisterRequest` (email, password) is validated at the boundary, `RegisterUser` checks the email isn't already taken, `Argon2PasswordHasher.hash()` produces the `$argon2id$`-prefixed hash, a new `tenant_id` is generated for the user (this sub-project has no invite/join-existing-tenant flow — every registration starts its own tenant), and `PostgresUserRepository.save()` inserts the row. Returns 201 with the new user's id; it does not issue tokens — registration and login are separate steps, so a client always exercises the same login path regardless of how the account was created.

**Login** (`POST /auth/login`): `AuthenticateUser` looks up the user by email, and `Argon2PasswordHasher.verify()` checks the password against the stored hash — a failure here and a "user not found" failure return the same generic `401 Invalid credentials`, deliberately, so the endpoint doesn't leak which emails are registered. On success, `JWTTokenIssuer` issues a 15-minute access token and a 7-day refresh token (rotation id in Redis), and the refresh token is set as an httpOnly cookie rather than returned in the JSON body, per SECURITY.md's reasoning that a token reachable from page JavaScript is a token reachable by an XSS payload.

**Refresh** (`POST /auth/refresh`): reads the refresh token from the httpOnly cookie, `RefreshAccessToken` looks up its Redis key; if present, it deletes that key, issues a new access token and a new refresh token (new Redis key, new cookie), and returns the new access token. If the key isn't found — expired, already used, or never valid — the endpoint returns `401` and the client is forced back to `/auth/login`.

**Logout** (`POST /auth/logout`): deletes the current refresh token's Redis key and clears the cookie. The access token itself isn't revocable before its own 15-minute expiry — that's the accepted tradeoff of a short-lived stateless access token, which is exactly why SECURITY.md keeps that lifetime short.

**Every other protected endpoint** (none exist yet in this sub-project, but the mechanism is what later endpoints build on): `get_current_user` extracts the access token from the `Authorization: Bearer` header, verifies its signature and expiry via `JWTTokenIssuer`, and the `tenant_id` it carries drives the tenant-context call described above before any query runs (`set_config(..., true)` as shipped, not the `SET LOCAL` form this document originally named — see the correction note above).

## Rate limiting

`RedisRateLimiter` implements a sliding-window counter, applied only to `/auth/register` and `/auth/login` in this sub-project (the 100/min-chat and 10/min-upload tiers SECURITY.md specifies don't apply yet — those endpoints don't exist until the RAG sub-project). Ruling: 5 requests/minute per IP on both endpoints. SECURITY.md says only that auth endpoints need "tighter limits" without a number; 5/minute is this project's own choice, not a cited figure, and is recorded as such rather than presented as sourced. Every rate-limited response carries `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers on both the accepted and the `429`-rejected paths, per SECURITY.md's stated convention.

## Security-control traceability

Every control this sub-project claims to implement maps to a specific, automatable check — the same discipline `docs/security/SECURITY.md` itself is written to:

| Control | Verified by |
|---|---|
| Argon2id password hashing | Unit test: hash a password, assert the stored value starts with `$argon2id$`; hash the same password twice, assert the two hashes differ (salt) |
| 15-minute access token expiry | Integration test: issue a token, advance past 15 minutes (via a fake clock injected into `JWTTokenIssuer`), assert the protected dependency rejects it |
| Refresh rotation invalidates reuse | Integration test: refresh once, then replay the original (now-deleted) refresh cookie, assert `401` |
| Row-level tenant isolation | Integration test (`test_rls_tenant_isolation.py`), the exact shape SECURITY.md specifies: seed two tenants, run a query against the raw connection with no application-level tenant filter, assert RLS still returns zero cross-tenant rows |
| Parameterized queries / no raw SQL | Every repository method uses SQLAlchemy's query builder or bound parameters — no f-string or `%`-formatted SQL anywhere in `infrastructure/`; a targeted test feeds an adversarial string (`"'; DROP TABLE users; --"`) through `PostgresUserRepository.find_by_email` and asserts it's treated as inert data |
| Pydantic input validation at the boundary | Each router's request model gets a test asserting a set of invalid payloads (missing field, oversized string, wrong type) is rejected with `422` and valid ones are accepted |
| Rate limiting with correct headers | Integration test against real Redis: fire six requests in one window, assert the sixth is `429` and every response carries the three `X-RateLimit-*` headers |
| Secrets never hardcoded | `JWT_SECRET_KEY` is read from the environment via the same loader convention `docs/security/SECRETS_MANAGEMENT.md` already documents; added to `.env.example` (name only) and to that document's "where each key lives" list |

## Testing strategy

Per `docs/testing/TESTING.md`'s pyramid: `tests/unit/` covers domain entities and application use cases against fake, in-memory implementations of the `identity/domain/ports.py` interfaces — no database, no network, fast and mock-free in the sense that matters (mocking a port you own is not the same failure mode as mocking a real dependency's behavior). `tests/integration/` runs against TestContainers-provisioned PostgreSQL and Redis — real RLS, real Argon2 hashing, real token issuance and expiry, real rate limiting — because, as `TESTING.md` argues for MAG and CAG's own core promises, these are exactly the claims a mock cannot actually verify. Coverage target is ≥80%, checked via `uv run pytest tests/ --cov=src --cov-report=term-missing` — the whole suite, not `tests/unit/` alone as this document originally specified. Measuring only the unit tests against a `--cov=src` denominator that includes the FastAPI router, the dependency wiring, and every infrastructure adapter counts code the unit tier deliberately never touches as uncovered, which understates the real figure badly enough to be useless as a gate.

## Docker and local dev

`docker/docker-compose.yml` runs three services: `api` (built from `docker/Dockerfile.api`, the FastAPI app under `uvicorn`), `postgres` (`pgvector/pgvector:pg16`, with a healthcheck), and `redis` (with a healthcheck). `api` depends on both healthchecks passing before it starts. Environment variables — `DATABASE_URL`, `REDIS_URL`, `JWT_SECRET_KEY`, plus `ANTHROPIC_API_KEY` declared but unused until the RAG sub-project's chat endpoint needs it — are read from `.env`, which already exists and is already protected by the working pre-commit hook and CI secret-scan; no changes to that protocol are needed beyond adding `JWT_SECRET_KEY`'s name to `.env.example`.

> **Corrected during implementation.** The variable list above is incomplete, because the design had not yet worked out that a single database URL cannot serve both roles. Two more shipped, both named in `.env.example` and in the `api` service's own `environment` block:
>
> - **`APP_DATABASE_URL`** — the connection the running application actually uses. `DATABASE_URL` is now only the bootstrap superuser connection for `alembic upgrade head`, which needs `CREATE TABLE`/`CREATE EXTENSION` and provisions the `app_user` role. They have to be different connections because PostgreSQL exempts both a superuser and a table's owner from row-level security no matter how correct the policy is — an API connecting as the bootstrap role would have made every RLS policy in this system silently inert while every test still passed. `src/api/dependencies.py` reads `APP_DATABASE_URL` at import time, so a missing value fails fast with a `KeyError` rather than quietly falling back.
> - **`APP_DB_PASSWORD`** — the password the migration assigns to that `app_user` role, kept out of the migration file as a literal.
>
> One variable was also added that is not a secret: **`COOKIE_SECURE`**, optional and defaulting to `true`, which exists only so local HTTP development can opt out of the refresh cookie's `Secure` flag. Full reasoning for the superuser/`app_user` split is in the plan's Task 5 section (`docs/superpowers/plans/2026-08-22-auth-foundation.md`).

## Explicit non-goals

No React frontend — `OVERVIEW.md`'s own Phase 1 tree is backend-only, and nothing about auth needs a UI to be tested or verified. No OAuth2/OIDC or CAPTCHA — both are named in SECURITY.md but scoped there as additions to a working password-auth flow, not a prerequisite for one. No multi-tenant invite/join flow — every registration creates its own tenant; joining an existing tenant is a real feature this sub-project doesn't need to unblock the RAG work that follows it. No TLS/ingress/Traefik configuration — SECURITY.md itself treats that as infrastructure to confirm once something is deployed, not application code this plan produces.
