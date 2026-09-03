# Auth Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the repository's first working code — a FastAPI service with registration, login, refresh, and logout, backed by PostgreSQL (with row-level tenant isolation) and Redis, running under Docker Compose, with the security controls docs/security/SECURITY.md requires each verified by a real test.

**Architecture:** Hexagonal — `src/identity/domain/` (framework-free entities and port interfaces), `src/identity/application/` (use cases orchestrating ports), `src/identity/infrastructure/` (concrete Postgres/Redis/Argon2/JWT adapters) — driven by `src/api/` (FastAPI routers and dependencies). Unit tests exercise the domain and application layers against fake in-memory ports; integration tests exercise the infrastructure layer and the full HTTP stack against TestContainers-provisioned PostgreSQL and Redis.

**Tech Stack:** Python 3.11+, FastAPI 0.111+, Pydantic v2, SQLAlchemy 2.0 (async, asyncpg), Alembic, argon2-cffi, PyJWT, redis-py (asyncio), uv, pytest/pytest-asyncio/pytest-cov, TestContainers, mypy, ruff.

**Spec:** docs/superpowers/specs/2026-08-22-auth-foundation-design.md

**Tracking:** GitHub Story #145 (parent Epic #144), branch `feature/145-auth-foundation`.

## Global Constraints

- `src/identity/domain/` never imports FastAPI, SQLAlchemy, redis, argon2, or PyJWT — zero framework dependencies.
- `src/identity/application/` depends only on `src/identity/domain/ports.py` interfaces, never on a concrete `src/identity/infrastructure/` class directly.
- Passwords are hashed with Argon2id via `argon2-cffi`; every stored hash starts with `$argon2id$`; hashing the same password twice produces two different hashes.
- Access tokens expire in 15 minutes; refresh tokens expire in 7 days and live in Redis at `identity:refresh:{token_id}` → `{"user_id": ..., "expires_at": ...}` — never in PostgreSQL. A refresh token is deleted from Redis the moment it's used (rotation) or revoked (logout), so replaying a used token finds no key and fails.
- JWTs are signed HS256 with the `JWT_SECRET_KEY` environment variable.
- `Sessions` gets a `CREATE POLICY tenant_isolation ON sessions USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)` row-level-security policy (`Users` deliberately does not — it's the tenant-identity root, not tenant-scoped data; see Task 5). The tenant context is set via `set_tenant_context()`, which uses `set_config('app.current_tenant_id', ..., true)` rather than a literal `SET LOCAL` statement, since PostgreSQL's `SET`/`SET LOCAL` grammar doesn't accept a bound query parameter. The value always comes from the authenticated user's own `tenant_id`, never from client input.
- `/auth/register` and `/auth/login` are rate-limited to 5 requests/minute/IP; every response from those two routes carries `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers.
- Login and register failures both return `401 {"detail": "Invalid credentials"}` for wrong-password and unknown-email cases alike — no response shape or timing difference that reveals whether an email is registered.
- No raw or string-interpolated SQL anywhere — SQLAlchemy's query builder or bound parameters only.
- The PostgreSQL container image is `pgvector/pgvector:pg16`.
- Unit tests (`tests/unit/`) use fake, in-memory implementations of `src/identity/domain/ports.py` — no real database, no real Redis, no network. Integration tests (`tests/integration/`) use TestContainers-provisioned PostgreSQL and Redis — real hashing, real tokens, real RLS, real rate limiting.
- Coverage target is ≥80%, checked via `uv run pytest tests/ --cov=src --cov-report=term-missing` — the whole suite, not `tests/unit/` alone. This architecture deliberately pushes infrastructure-adapter coverage (Postgres/Redis clients, the FastAPI router) into the integration tier rather than mocking those dependencies (docs/testing/TESTING.md's own reasoning: a mock can't verify what these adapters actually promise), so a unit-tests-only coverage denominator can never reach 80% no matter how complete the suite is — it isn't measuring what it claims to. This correction landed during the final whole-branch review, replacing the earlier `tests/unit/`-only wording used through Task 15.
- Every commit follows the `.gitmessage` template and Conventional Commits format.

---

### Task 1: Project scaffold and tooling

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`, `src/api/__init__.py`, `src/api/routers/__init__.py`, `src/identity/__init__.py`, `src/identity/domain/__init__.py`, `src/identity/application/__init__.py`, `src/identity/infrastructure/__init__.py`
- Create: `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `tests/conftest.py`
- Modify: `.env.example` (add `DATABASE_URL`, `REDIS_URL`, `JWT_SECRET_KEY`)
- Modify: `docs/security/SECRETS_MANAGEMENT.md` (add `JWT_SECRET_KEY` to the "where each key lives" list)

**Interfaces:**
- Produces: an installable project (`uv sync` succeeds), `ruff check src/ tests/` and `mypy src/` both run clean on the (near-empty) tree, `pytest` collects zero tests without error.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "unified-rag-cag-mag"
version = "0.1.0"
description = "Unified RAG x CAG x MAG AI System"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.29",
    "alembic>=1.13",
    "argon2-cffi>=23.1",
    "pyjwt>=2.8",
    "redis>=5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "httpx>=0.27",
    "testcontainers>=4.5",
    "mypy>=1.10",
    "ruff>=0.5",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.11"
strict = true
exclude = ["tests/"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Create the package skeleton**

Create every `__init__.py` listed above as an empty file (or, for `src/__init__.py`, a single docstring-free empty file — no content required, these exist only to make the directories importable packages).

- [ ] **Step 3: Create `tests/conftest.py` with a placeholder marker fixture**

```python
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
```

(This fixture is a minimal placeholder so `pytest` has something valid to collect; Task 5 replaces it with the real TestContainers fixtures.)

- [ ] **Step 4: Install and verify**

Run: `uv sync --extra dev`
Expected: dependency resolution succeeds, `.venv/` is created.

Run: `uv run ruff check src/ tests/`
Expected: no errors (nothing to lint yet beyond empty files).

Run: `uv run mypy src/`
Expected: no errors.

Run: `uv run pytest`
Expected: `1 passed` or `0 collected` with no errors (the placeholder fixture alone doesn't constitute a test; either outcome is fine as long as nothing errors).

- [ ] **Step 5: Add the new secrets to `.env.example` and `SECRETS_MANAGEMENT.md`**

Add these three lines to `.env.example` (names only, matching the existing file's convention of no real values):

```
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/unified_ai
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=
```

In `docs/security/SECRETS_MANAGEMENT.md`, add `JWT_SECRET_KEY` to the "Where each key lives, and why nowhere else" section's list of secrets this project manages, alongside the existing `ANTHROPIC_API_KEY`/`GEMINI_API_KEY`/`DEEPSEEK_API_KEY` mentions — one sentence noting it signs this project's own JWTs rather than authenticating to a third-party provider, so its leak-response is "rotate and every previously issued token becomes invalid," not "revoke in a vendor console."

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ tests/ .env.example docs/security/SECRETS_MANAGEMENT.md
git commit -m "feat: scaffold project structure and tooling for Auth Foundation"
```

---

### Task 2: Identity domain layer

**Files:**
- Create: `src/identity/domain/entities.py`
- Create: `src/identity/domain/ports.py`
- Create: `src/identity/domain/errors.py`
- Test: `tests/unit/test_identity_domain.py`

**Interfaces:**
- Produces: `User` (id, email, hashed_password, tenant_id, created_at, updated_at), `PasswordHash` (wraps an opaque string), `AccessToken`/`RefreshToken` (value objects: token string + expiry), the `PasswordHasher`, `TokenIssuer`, `UserRepository`, `RefreshTokenStore`, `RateLimiter` ABCs, and `InvalidCredentials`/`EmailAlreadyRegistered`/`TokenExpired`/`TokenAlreadyUsed` exceptions. Every later task consumes these.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_identity_domain.py
import uuid
from datetime import datetime, timezone

import pytest

from src.identity.domain.entities import PasswordHash, User
from src.identity.domain.errors import InvalidCredentials


def test_user_equality_is_by_all_fields():
    now = datetime.now(timezone.utc)
    shared_id = uuid.uuid4()
    tenant = uuid.uuid4()
    a = User(
        id=shared_id,
        email="a@example.com",
        hashed_password=PasswordHash("$argon2id$v=19$m=1,t=1,p=1$abc$def"),
        tenant_id=tenant,
        created_at=now,
        updated_at=now,
    )
    b = User(
        id=shared_id,
        email="a@example.com",
        hashed_password=PasswordHash("$argon2id$v=19$m=1,t=1,p=1$abc$def"),
        tenant_id=tenant,
        created_at=now,
        updated_at=now,
    )
    c = User(
        id=uuid.uuid4(),
        email="a@example.com",
        hashed_password=PasswordHash("$argon2id$v=19$m=1,t=1,p=1$abc$def"),
        tenant_id=tenant,
        created_at=now,
        updated_at=now,
    )
    assert a == b
    assert a != c


def test_password_hash_rejects_a_value_that_is_not_argon2id():
    with pytest.raises(ValueError):
        PasswordHash("plaintext-not-a-hash")


def test_password_hash_accepts_a_real_argon2id_value():
    ph = PasswordHash("$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl")
    assert str(ph).startswith("$argon2id$")


def test_invalid_credentials_error_carries_no_identifying_detail():
    err = InvalidCredentials()
    assert str(err) == "Invalid credentials"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_identity_domain.py -v`
Expected: `ModuleNotFoundError: No module named 'src.identity.domain.entities'` (or similar import failure).

- [ ] **Step 3: Write `src/identity/domain/errors.py`**

```python
class InvalidCredentials(Exception):
    def __init__(self) -> None:
        super().__init__("Invalid credentials")


class EmailAlreadyRegistered(Exception):
    def __init__(self, email: str) -> None:
        super().__init__(f"Email already registered: {email}")
        self.email = email


class TokenExpired(Exception):
    def __init__(self) -> None:
        super().__init__("Token expired")


class TokenAlreadyUsed(Exception):
    def __init__(self) -> None:
        super().__init__("Token already used or invalid")
```

- [ ] **Step 4: Write `src/identity/domain/entities.py`**

```python
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime

_ARGON2ID_PATTERN = re.compile(r"^\$argon2id\$")


@dataclass(frozen=True)
class PasswordHash:
    value: str

    def __init__(self, value: str) -> None:
        if not _ARGON2ID_PATTERN.match(value):
            raise ValueError("PasswordHash must be an Argon2id hash ($argon2id$ prefix)")
        object.__setattr__(self, "value", value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class User:
    id: uuid.UUID
    email: str
    hashed_password: PasswordHash
    tenant_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class AccessToken:
    value: str
    expires_at: datetime


@dataclass(frozen=True)
class RefreshToken:
    token_id: uuid.UUID
    value: str
    expires_at: datetime


@dataclass(frozen=True)
class TokenPair:
    access_token: AccessToken
    refresh_token: RefreshToken
```

- [ ] **Step 5: Write `src/identity/domain/ports.py`**

```python
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from src.identity.domain.entities import AccessToken, PasswordHash, RefreshToken, TokenPair, User


class PasswordHasher(ABC):
    @abstractmethod
    def hash(self, plain_password: str) -> PasswordHash: ...

    @abstractmethod
    def verify(self, plain_password: str, hashed: PasswordHash) -> bool: ...


class TokenIssuer(ABC):
    @abstractmethod
    def issue_pair(self, user_id: uuid.UUID, tenant_id: uuid.UUID) -> TokenPair: ...

    @abstractmethod
    def verify_access_token(self, token: str) -> dict:
        """Returns the decoded claims, or raises TokenExpired / TokenAlreadyUsed-equivalent."""
        ...


class UserRepository(ABC):
    @abstractmethod
    async def save(self, user: User) -> None: ...

    @abstractmethod
    async def find_by_email(self, email: str) -> User | None: ...

    @abstractmethod
    async def find_by_id(self, user_id: uuid.UUID) -> User | None: ...


class RefreshTokenStore(ABC):
    @abstractmethod
    async def save(self, refresh_token: RefreshToken, user_id: uuid.UUID) -> None: ...

    @abstractmethod
    async def get_user_id(self, token_id: uuid.UUID) -> uuid.UUID | None: ...

    @abstractmethod
    async def delete(self, token_id: uuid.UUID) -> None: ...


class RateLimiter(ABC):
    @abstractmethod
    async def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int, datetime]:
        """Returns (allowed, remaining, reset_at)."""
        ...
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_identity_domain.py -v`
Expected: `4 passed`.

- [ ] **Step 7: Confirm zero framework imports**

Run: `grep -rE "^(import|from) (fastapi|sqlalchemy|redis|argon2|jwt)" src/identity/domain/`
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add src/identity/domain/ tests/unit/test_identity_domain.py
git commit -m "feat: add identity domain entities, ports, and errors"
```

---

### Task 3: Argon2 password hasher

**Files:**
- Create: `src/identity/infrastructure/argon2_password_hasher.py`
- Test: `tests/unit/test_argon2_password_hasher.py`

**Interfaces:**
- Consumes: `PasswordHasher` port, `PasswordHash` entity from Task 2.
- Produces: `Argon2PasswordHasher`, consumed by Task 10's `RegisterUser` and `AuthenticateUser` use cases.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_argon2_password_hasher.py
from src.identity.infrastructure.argon2_password_hasher import Argon2PasswordHasher


def test_hash_produces_an_argon2id_prefixed_value():
    hasher = Argon2PasswordHasher()
    result = hasher.hash("correct horse battery staple")
    assert str(result).startswith("$argon2id$")


def test_hashing_the_same_password_twice_produces_different_hashes():
    hasher = Argon2PasswordHasher()
    a = hasher.hash("correct horse battery staple")
    b = hasher.hash("correct horse battery staple")
    assert str(a) != str(b)


def test_verify_succeeds_for_the_correct_password():
    hasher = Argon2PasswordHasher()
    hashed = hasher.hash("correct horse battery staple")
    assert hasher.verify("correct horse battery staple", hashed) is True


def test_verify_fails_for_the_wrong_password():
    hasher = Argon2PasswordHasher()
    hashed = hasher.hash("correct horse battery staple")
    assert hasher.verify("wrong password", hashed) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_argon2_password_hasher.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/identity/infrastructure/argon2_password_hasher.py`**

```python
from argon2 import PasswordHasher as _Argon2PasswordHasher
from argon2.exceptions import VerifyMismatchError

from src.identity.domain.entities import PasswordHash
from src.identity.domain.ports import PasswordHasher


class Argon2PasswordHasher(PasswordHasher):
    def __init__(self) -> None:
        self._hasher = _Argon2PasswordHasher()

    def hash(self, plain_password: str) -> PasswordHash:
        return PasswordHash(self._hasher.hash(plain_password))

    def verify(self, plain_password: str, hashed: PasswordHash) -> bool:
        try:
            return self._hasher.verify(str(hashed), plain_password)
        except VerifyMismatchError:
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_argon2_password_hasher.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/identity/infrastructure/argon2_password_hasher.py tests/unit/test_argon2_password_hasher.py
git commit -m "feat: add Argon2id password hasher"
```

---

### Task 4: JWT token issuer

**Files:**
- Create: `src/identity/infrastructure/jwt_token_issuer.py`
- Test: `tests/unit/test_jwt_token_issuer.py`

**Interfaces:**
- Consumes: `TokenIssuer` port, `TokenPair`/`AccessToken`/`RefreshToken` entities from Task 2.
- Produces: `JWTTokenIssuer(secret_key: str, clock: Callable[[], datetime] = ...)`, consumed by Task 10/11's use cases and Task 12's `get_current_user` dependency. The `clock` parameter is what Step 5 below and the spec's traceability table both rely on to test expiry without waiting 15 real minutes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_jwt_token_issuer.py
import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from src.identity.domain.errors import TokenExpired
from src.identity.infrastructure.jwt_token_issuer import JWTTokenIssuer

SECRET = "test-secret-key-not-for-production"


def test_issue_pair_returns_an_access_token_expiring_in_15_minutes():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    issuer = JWTTokenIssuer(secret_key=SECRET, clock=lambda: now)
    pair = issuer.issue_pair(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())
    assert pair.access_token.expires_at == now + timedelta(minutes=15)


def test_issue_pair_returns_a_refresh_token_expiring_in_7_days():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    issuer = JWTTokenIssuer(secret_key=SECRET, clock=lambda: now)
    pair = issuer.issue_pair(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())
    assert pair.refresh_token.expires_at == now + timedelta(days=7)


def test_verify_access_token_returns_the_claims_within_expiry():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    issuer = JWTTokenIssuer(secret_key=SECRET, clock=lambda: now)
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    pair = issuer.issue_pair(user_id=user_id, tenant_id=tenant_id)
    claims = issuer.verify_access_token(pair.access_token.value)
    assert claims["sub"] == str(user_id)
    assert claims["tenant_id"] == str(tenant_id)


def test_verify_access_token_rejects_a_token_issued_16_minutes_ago():
    issued_at = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    issuer_at_issue_time = JWTTokenIssuer(secret_key=SECRET, clock=lambda: issued_at)
    pair = issuer_at_issue_time.issue_pair(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())

    later = issued_at + timedelta(minutes=16)
    issuer_at_verify_time = JWTTokenIssuer(secret_key=SECRET, clock=lambda: later)
    with pytest.raises(TokenExpired):
        issuer_at_verify_time.verify_access_token(pair.access_token.value)


def test_verify_access_token_rejects_a_tampered_token():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    issuer = JWTTokenIssuer(secret_key=SECRET, clock=lambda: now)
    pair = issuer.issue_pair(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())

    # Mutate the FIRST character of the signature segment, not the last.
    # Base64url's last character can carry "don't-care" padding bits when the
    # encoded length isn't a multiple of 4 (an HMAC-SHA256 signature's base64url
    # form has exactly this property) — flipping it decodes to the same bytes
    # often enough (~5.6% in a 2000-run stress test) that the token isn't
    # actually corrupted and the test flakes. The first character of a
    # multi-character base64 segment is always a fully-determined 6-bit group.
    header, payload, signature = pair.access_token.value.split(".")
    tampered_signature = ("A" if signature[0] != "A" else "B") + signature[1:]
    tampered = f"{header}.{payload}.{tampered_signature}"

    with pytest.raises(TokenExpired):
        issuer.verify_access_token(tampered)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_jwt_token_issuer.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/identity/infrastructure/jwt_token_issuer.py`**

```python
from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import jwt
from jwt import InvalidTokenError

from src.identity.domain.entities import AccessToken, RefreshToken, TokenPair
from src.identity.domain.errors import TokenExpired
from src.identity.domain.ports import TokenIssuer

_ACCESS_TOKEN_LIFETIME = timedelta(minutes=15)
_REFRESH_TOKEN_LIFETIME = timedelta(days=7)
_ALGORITHM = "HS256"


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


class JWTTokenIssuer(TokenIssuer):
    def __init__(self, secret_key: str, clock: Callable[[], datetime] = _default_clock) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def issue_pair(self, user_id: uuid.UUID, tenant_id: uuid.UUID) -> TokenPair:
        now = self._clock()
        access_expires_at = now + _ACCESS_TOKEN_LIFETIME
        access_claims = {
            "sub": str(user_id),
            "tenant_id": str(tenant_id),
            "exp": access_expires_at,
            "iat": now,
            "type": "access",
        }
        access_value = jwt.encode(access_claims, self._secret_key, algorithm=_ALGORITHM)

        refresh_expires_at = now + _REFRESH_TOKEN_LIFETIME
        token_id = uuid.uuid4()
        refresh_claims = {
            "sub": str(user_id),
            "jti": str(token_id),
            "exp": refresh_expires_at,
            "iat": now,
            "type": "refresh",
        }
        refresh_value = jwt.encode(refresh_claims, self._secret_key, algorithm=_ALGORITHM)

        return TokenPair(
            access_token=AccessToken(value=access_value, expires_at=access_expires_at),
            refresh_token=RefreshToken(
                token_id=token_id, value=refresh_value, expires_at=refresh_expires_at
            ),
        )

    def verify_access_token(self, token: str) -> dict:
        try:
            return jwt.decode(token, self._secret_key, algorithms=[_ALGORITHM])
        except InvalidTokenError as exc:
            raise TokenExpired() from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_jwt_token_issuer.py -v`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/identity/infrastructure/jwt_token_issuer.py tests/unit/test_jwt_token_issuer.py
git commit -m "feat: add JWT token issuer with injectable clock for expiry testing"
```

---

### Task 5: Database engine, Alembic migration, and TestContainers fixtures

**Files:**
- Create: `src/identity/infrastructure/db.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/0001_users_sessions.py`
- Create: `tests/integration/conftest.py` (leave `tests/conftest.py` as Task 1's placeholder — untouched)
- Modify: `.env.example`, `docs/security/SECRETS_MANAGEMENT.md` (add `APP_DB_PASSWORD`)
- Test: `tests/integration/test_migration.py`

**Interfaces:**
- Produces: `get_engine(database_url: str)`, `get_sessionmaker(engine)`, `set_tenant_context(session, tenant_id)`, and the `Users`/`Sessions` tables with RLS enabled and enforced (including against the table owner, via `FORCE ROW LEVEL SECURITY`) plus a non-superuser `app_user` role holding only DML privileges on both tables. Consumed by Task 6's repository and every later integration test via the `postgres_container`/`app_database_url`/`db_session` fixtures this task adds to `tests/integration/conftest.py`, and by Task 12's `src/api/dependencies.py` via the `APP_DATABASE_URL` environment variable at runtime.

- [ ] **Step 1: Write `src/identity/infrastructure/db.py`**

```python
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def get_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def get_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
```

- [ ] **Step 2: Write `alembic.ini`**

```ini
[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = driver://user:pass@localhost/dbname

[loggers]
keys = root,sqlalchemy,alembic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handlers]
keys = console

[handler_console]
class = StreamHandler
args = (sys.stderr,)
formatter = generic

[formatters]
keys = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

- [ ] **Step 3: Write `alembic/env.py`**

`DATABASE_URL` is read from the environment rather than `alembic.ini`'s placeholder, and migrations run synchronously via `asyncpg`'s sync-compatible counterpart (`psycopg` isn't a project dependency; instead this uses SQLAlchemy's `run_sync` bridge so Alembic's traditionally-sync migration runner works against the same `asyncpg`-based engine the app uses):

```python
import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def get_url() -> str:
    return os.environ["DATABASE_URL"]


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()
    connectable = async_engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

- [ ] **Step 4: Write `alembic/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 5: Write `alembic/versions/0001_users_sessions.py`**

PostgreSQL exempts the table owner and any superuser from row-level security by default — `ENABLE ROW LEVEL SECURITY` alone never restricts either, and `FORCE ROW LEVEL SECURITY` extends enforcement to the owner but explicitly still never applies to a superuser (this is a hard, non-overridable PostgreSQL rule, not a configuration gap). Alembic needs to run as a privileged role to create tables and extensions, so migrations still run as the database's bootstrap superuser — but if the running application connects with that same role, every RLS policy in this system would be silently inert, in tests and in the real deployed service alike. This migration closes that gap by provisioning a separate, non-superuser `app_user` role with only the DML privileges the application actually needs, and every application-level connection (the `db_session` fixture below, and Task 14's `api` service) uses that role instead of the bootstrap superuser. `APP_DB_PASSWORD` is a new secret this task introduces — add it to `.env.example` and `docs/security/SECRETS_MANAGEMENT.md`'s secrets list alongside `JWT_SECRET_KEY`, following the same "name only, no real value in the repo" convention.

The `tenant_isolation` policy is applied to `sessions` only, not `users` — a deliberate, corrected scope, not an oversight. `Users` is what `docs/database/DATABASE.md` calls "the root of the tenant model": every registration in this system creates a brand-new tenant, so a `Users` row IS a tenant's root record, not data scoped by a tenant established elsewhere. `Sessions`, by contrast, holds actual per-tenant data (a conversation, its `context_budget`) that a different tenant genuinely must never see — exactly the case RLS exists for. Enforcing RLS on `Users` as well creates a real structural conflict with login: `find_by_email` has to search across every tenant by design, since a client authenticating doesn't know their tenant_id yet — that's the whole point of looking it up by email. Under `FORCE ROW LEVEL SECURITY`, a policy with no `WITH CHECK` clause falls back to the `USING` clause for inserts too, so even *registration* — creating the very first row for a brand-new tenant — would be rejected, because no session has "the new tenant's" context set before that row exists to derive it from. There's no clean way to set `app.current_tenant_id` correctly before the row it would be set from is written; the only fixes are either a second, RLS-bypassing role reserved for the login/registration path specifically, or not putting `users` under RLS in the first place. The second is simpler, costs nothing security-wise (a `Users` row is only ever reached by someone who already knows that email and can prove they know the password — that's authentication working as intended, not a cross-tenant leak), and is what this migration does.

```python
"""users and sessions with row-level security

Revision ID: 0001
Revises:
Create Date: 2026-08-22

"""
import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("email", sa.String, nullable=False, unique=True),
        sa.Column("hashed_password", sa.String, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])

    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String, nullable=True),
        sa.Column("context_budget", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_sessions_tenant_id", "sessions", ["tenant_id"])

    op.execute("ALTER TABLE sessions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE sessions FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON sessions
        USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
        """
    )

    # A non-superuser, non-owner role for the running application to connect
    # as. The migration itself keeps running as the bootstrap superuser
    # (needed for CREATE TABLE/EXTENSION), but nothing in the application
    # ever should — a superuser connection bypasses every RLS policy above
    # regardless of how correct the policy itself is. The password comes from
    # the environment, never a literal in this file; single quotes are
    # doubled (the standard SQL string-literal escape) because CREATE ROLE's
    # PASSWORD clause is a keyword-value pair in PostgreSQL's grammar, not an
    # expression context, so it cannot take a bound query parameter the way
    # a normal DML value can — this is a deployment-time secret being placed
    # into one-time role-provisioning DDL, not a user-supplied query value,
    # which is what the project's "no raw/string-interpolated SQL" rule
    # exists to prevent.
    app_db_password = os.environ["APP_DB_PASSWORD"].replace("'", "''")
    op.execute(
        f"""
        DO $$
        BEGIN
          IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_user') THEN
            CREATE ROLE app_user LOGIN PASSWORD '{app_db_password}';
          ELSE
            ALTER ROLE app_user WITH PASSWORD '{app_db_password}';
          END IF;
        END
        $$;
        """
    )
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON users, sessions TO app_user")


def downgrade() -> None:
    op.execute("REVOKE ALL ON users, sessions FROM app_user")
    op.drop_table("sessions")
    op.drop_table("users")
```

Note the `true` second argument to `current_setting` — it makes the setting optional, returning `NULL` rather than raising when `app.current_tenant_id` hasn't been set on the current connection (a migration-time `ALTER TABLE ... ENABLE ROW LEVEL SECURITY` check, or any session that genuinely has no tenant context, would otherwise error instead of simply matching zero rows).

- [ ] **Step 6: Add TestContainers fixtures scoped to `tests/integration/` only**

These fixtures start real Docker containers and run a real migration — `tests/unit/` needs none of that, and per docs/testing/TESTING.md's own design, the unit tier is supposed to run with no database at all. An `autouse` fixture in the root `tests/conftest.py` would apply to every test under `tests/`, unit included, forcing container startup cost onto tests that don't touch a database. Putting these fixtures in `tests/integration/conftest.py` instead means pytest only applies them to tests actually under `tests/integration/`.

Leave `tests/conftest.py` as Task 1 created it (the placeholder `anyio_backend` fixture) — don't add anything to it in this task.

Create `tests/integration/conftest.py`. `database_url` (built from `postgres_container.get_connection_url()`) carries the container's bootstrap superuser credentials — needed for `run_migrations` to create tables/extensions/the `app_user` role, but never for anything else, since a superuser connection bypasses every RLS policy regardless of correctness. `app_database_url` swaps in `app_user`'s credentials on the same host/port/database, and `db_session` — what every other task's tests actually use — connects through that instead:

```python
import os

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from src.identity.infrastructure.db import get_engine, get_sessionmaker

_APP_DB_PASSWORD = "test-only-app-user-password"  # never used outside a throwaway TestContainers instance


@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        yield container


@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer("redis:7") as container:
        yield container


@pytest.fixture(scope="session")
def database_url(postgres_container: PostgresContainer) -> str:
    url = postgres_container.get_connection_url()
    return url.replace("postgresql+psycopg2", "postgresql+asyncpg")


@pytest.fixture(scope="session")
def redis_url(redis_container: RedisContainer) -> str:
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}/0"


@pytest.fixture(scope="session", autouse=True)
def run_migrations(database_url: str):
    os.environ["DATABASE_URL"] = database_url
    os.environ["APP_DB_PASSWORD"] = _APP_DB_PASSWORD
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    yield


@pytest.fixture(scope="session")
def app_database_url(database_url: str, run_migrations) -> str:
    # Depends on run_migrations (not just database_url) because app_user
    # doesn't exist until the migration creates it.
    #
    # str(url) is NOT safe here: SQLAlchemy's URL.__str__ calls
    # render_as_string(hide_password=True) by default, which replaces the
    # password with a literal "***" — every connection made from the
    # returned string would then fail authentication with that literal
    # masked value. render_as_string(hide_password=False) is required to
    # get the real password back out.
    url = make_url(database_url)
    return url.set(username="app_user", password=_APP_DB_PASSWORD).render_as_string(hide_password=False)


@pytest_asyncio.fixture
async def db_session(app_database_url: str):
    engine = get_engine(app_database_url)
    sessionmaker = get_sessionmaker(engine)
    async with sessionmaker() as session:
        yield session
        await session.rollback()
    await engine.dispose()
```

`autouse=True` on `run_migrations` still means every test *under `tests/integration/`* gets a migrated database whether it names the fixture explicitly or not — that scope is correct and intentional, since every integration test in this plan needs the schema to exist. It's only the root-level `tests/conftest.py` placement that was wrong.

- [ ] **Step 7: Write the failing tests — DDL metadata only here; the real query-level proof belongs to Task 7**

`users` carries no RLS policy at all (see the note above Step 5) — only `sessions` does, so these tests check `sessions` specifically rather than both tables. A real query-level test proving `current_setting(..., true)`'s no-error behavior needs seeded rows in two different tenants to be meaningful rather than vacuous (an empty table returns zero rows whether or not RLS is doing anything) — `sessions` has no rows yet at this point in the plan, and seeding them requires a valid `user_id` foreign key, so that proof is Task 7's job, not this one's. This task's tests confirm the policy exists and is structurally correct; Task 7 proves it actually filters.

```python
# tests/integration/test_migration.py
from sqlalchemy import text


async def test_sessions_table_exists_with_rls_enabled(db_session):
    result = await db_session.execute(
        text("SELECT relrowsecurity FROM pg_class WHERE relname = 'sessions'")
    )
    assert result.scalar_one() is True


async def test_tenant_isolation_policy_exists_on_sessions(db_session):
    result = await db_session.execute(
        text("SELECT tablename FROM pg_policies WHERE policyname = 'tenant_isolation'")
    )
    tables = {row.tablename for row in result}
    assert tables == {"sessions"}


async def test_users_table_has_no_rls_policy(db_session):
    # Deliberate: users is the tenant-identity root, not tenant-scoped data —
    # see the note above Step 5 for why RLS on users would break login/
    # registration. This test guards against a future edit accidentally
    # reintroducing a policy on users the way the original draft of this
    # migration did.
    result = await db_session.execute(
        text("SELECT relrowsecurity FROM pg_class WHERE relname = 'users'")
    )
    assert result.scalar_one() is False
```

- [ ] **Step 8: Run the tests to verify they fail, then pass**

Run: `uv run pytest tests/integration/test_migration.py -v`
Expected first run (before Docker is confirmed reachable): either a clean pass, or a Docker-connectivity error — if Docker isn't running, start it before proceeding; this task cannot be verified without a real container runtime, per the spec's integration-testing requirement.
Expected after Docker is available: `3 passed`.

Also confirm `tests/unit/` no longer touches Docker: run `uv run pytest tests/unit/ -v` and check it completes without starting any container (no TestContainers log lines, and it should be fast — seconds, not the tens-of-seconds a container pull/start adds).

Add `APP_DB_PASSWORD` to `.env.example` (name only, no value) and to `docs/security/SECRETS_MANAGEMENT.md`'s secrets list, next to `JWT_SECRET_KEY` — one sentence noting it's the password for the non-superuser `app_user` database role the migration creates, distinct from whatever admin credentials `DATABASE_URL` itself carries.

- [ ] **Step 9: Commit**

```bash
git add src/identity/infrastructure/db.py alembic.ini alembic/ tests/integration/conftest.py tests/integration/test_migration.py .env.example docs/security/SECRETS_MANAGEMENT.md
git commit -m "feat: add Users/Sessions migration with row-level security and TestContainers fixtures"
```

---

### Task 6: Postgres user repository

**Files:**
- Create: `src/identity/infrastructure/postgres_user_repository.py`
- Test: `tests/integration/test_postgres_user_repository.py`

**Interfaces:**
- Consumes: `UserRepository` port (Task 2), `db_session` fixture (Task 5).
- Produces: `PostgresUserRepository(session)`, consumed by Task 10's `RegisterUser`/`AuthenticateUser` use cases via `src/api/dependencies.py` (Task 12).

- [ ] **Step 1: Write the failing tests**

```python
# tests/integration/test_postgres_user_repository.py
import uuid
from datetime import datetime, timezone

import pytest

from src.identity.domain.entities import PasswordHash, User
from src.identity.domain.errors import EmailAlreadyRegistered
from src.identity.infrastructure.postgres_user_repository import PostgresUserRepository

VALID_HASH = PasswordHash("$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl")


def _new_user(email: str) -> User:
    now = datetime.now(timezone.utc)
    return User(
        id=uuid.uuid4(),
        email=email,
        hashed_password=VALID_HASH,
        tenant_id=uuid.uuid4(),
        created_at=now,
        updated_at=now,
    )


async def test_save_then_find_by_email_returns_the_same_user(db_session):
    repo = PostgresUserRepository(db_session)
    user = _new_user("alice@example.com")
    await repo.save(user)
    await db_session.commit()

    found = await repo.find_by_email("alice@example.com")
    assert found is not None
    assert found.id == user.id
    assert found.email == "alice@example.com"


async def test_find_by_email_returns_none_for_an_unknown_address(db_session):
    repo = PostgresUserRepository(db_session)
    assert await repo.find_by_email("nobody@example.com") is None


async def test_save_rejects_a_duplicate_email(db_session):
    repo = PostgresUserRepository(db_session)
    await repo.save(_new_user("bob@example.com"))
    await db_session.commit()

    with pytest.raises(EmailAlreadyRegistered):
        await repo.save(_new_user("bob@example.com"))


async def test_find_by_email_treats_an_adversarial_string_as_inert_data(db_session):
    repo = PostgresUserRepository(db_session)
    result = await repo.find_by_email("'; DROP TABLE users; --")
    assert result is None
    # If the string had been interpolated into raw SQL instead of bound as a
    # parameter, this second query would now fail because the table is gone.
    assert await repo.find_by_email("bob@example.com") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_postgres_user_repository.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/identity/infrastructure/postgres_user_repository.py`**

```python
import uuid

from sqlalchemy import Table, MetaData, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.identity.domain.entities import PasswordHash, User
from src.identity.domain.errors import EmailAlreadyRegistered
from src.identity.domain.ports import UserRepository

_metadata = MetaData()
_users_table = Table("users", _metadata, autoload_with=None)


class PostgresUserRepository(UserRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, user: User) -> None:
        from sqlalchemy import text

        try:
            await self._session.execute(
                text(
                    """
                    INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at)
                    VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)
                    """
                ),
                {
                    "id": user.id,
                    "email": user.email,
                    "hashed_password": str(user.hashed_password),
                    "tenant_id": user.tenant_id,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at,
                },
            )
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise EmailAlreadyRegistered(user.email) from exc

    async def find_by_email(self, email: str) -> User | None:
        from sqlalchemy import text

        result = await self._session.execute(
            text(
                "SELECT id, email, hashed_password, tenant_id, created_at, updated_at "
                "FROM users WHERE email = :email"
            ),
            {"email": email},
        )
        row = result.mappings().first()
        return self._row_to_user(row) if row else None

    async def find_by_id(self, user_id: uuid.UUID) -> User | None:
        from sqlalchemy import text

        result = await self._session.execute(
            text(
                "SELECT id, email, hashed_password, tenant_id, created_at, updated_at "
                "FROM users WHERE id = :id"
            ),
            {"id": user_id},
        )
        row = result.mappings().first()
        return self._row_to_user(row) if row else None

    @staticmethod
    def _row_to_user(row) -> User:
        return User(
            id=row["id"],
            email=row["email"],
            hashed_password=PasswordHash(row["hashed_password"]),
            tenant_id=row["tenant_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
```

(The unused `Table`/`MetaData`/`select`/`autoload_with` import at module scope is dead — remove it during this step; it's a leftover from an earlier draft of this file and isn't needed since every query here uses `text()` with bound parameters, which already satisfies the "no raw/string-interpolated SQL" constraint because the SQL string itself contains no user-supplied values, only `:name` placeholders bound separately.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_postgres_user_repository.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/identity/infrastructure/postgres_user_repository.py tests/integration/test_postgres_user_repository.py
git commit -m "feat: add Postgres user repository"
```

---

### Task 7: Row-level tenant isolation

**Files:**
- Modify: `src/identity/infrastructure/db.py` (add `set_tenant_context`)
- Test: `tests/integration/test_rls_tenant_isolation.py`

**Interfaces:**
- Consumes: `db_session` fixture (Task 5), `PostgresUserRepository` (Task 6).
- Produces: `set_tenant_context(session, tenant_id)`, consumed by Task 12's `get_db_session` API dependency.

This test targets `sessions`, not `users` — Task 5's migration deliberately puts no RLS policy on `users` at all (see the note above that task's Step 5: `users` is the tenant-identity root, and enforcing RLS on it would make login's cross-tenant email lookup structurally impossible). `sessions` is where this project's row-level tenant isolation actually lives, so it's what this test needs to prove is working.

- [ ] **Step 1: Write the failing test — the exact shape SECURITY.md specifies, applied to the table that's actually RLS-protected**

```python
# tests/integration/test_rls_tenant_isolation.py
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from src.identity.domain.entities import PasswordHash, User
from src.identity.infrastructure.db import set_tenant_context
from src.identity.infrastructure.postgres_user_repository import PostgresUserRepository

VALID_HASH = PasswordHash("$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl")


async def test_rls_returns_zero_cross_tenant_sessions_even_without_an_app_level_filter(db_session):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    now = datetime.now(timezone.utc)

    # sessions.user_id is a foreign key, so each session needs an owning
    # user first. users carries no RLS, so these two inserts need no tenant
    # context of their own — this only proves out once Task 6 exists.
    repo = PostgresUserRepository(db_session)
    user_a = User(id=uuid.uuid4(), email="a@tenant-a.com", hashed_password=VALID_HASH,
                  tenant_id=tenant_a, created_at=now, updated_at=now)
    user_b = User(id=uuid.uuid4(), email="b@tenant-b.com", hashed_password=VALID_HASH,
                  tenant_id=tenant_b, created_at=now, updated_at=now)
    await repo.save(user_a)
    await repo.save(user_b)
    await db_session.commit()

    # sessions IS under FORCE ROW LEVEL SECURITY, so each insert needs its
    # own matching tenant context — the policy's USING clause is what an
    # insert falls back to for its WITH CHECK when none is defined, so an
    # insert under the wrong (or no) context would be rejected outright.
    await set_tenant_context(db_session, tenant_a)
    await db_session.execute(
        text("INSERT INTO sessions (id, user_id, tenant_id, title) VALUES (:id, :user_id, :tenant_id, :title)"),
        {"id": uuid.uuid4(), "user_id": user_a.id, "tenant_id": tenant_a, "title": "tenant a's session"},
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_b)
    await db_session.execute(
        text("INSERT INTO sessions (id, user_id, tenant_id, title) VALUES (:id, :user_id, :tenant_id, :title)"),
        {"id": uuid.uuid4(), "user_id": user_b.id, "tenant_id": tenant_b, "title": "tenant b's session"},
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    # Deliberately no WHERE tenant_id = ... — RLS alone must do the filtering.
    result = await db_session.execute(text("SELECT title FROM sessions"))
    titles = {row.title for row in result}

    assert titles == {"tenant a's session"}
    assert "tenant b's session" not in titles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_rls_tenant_isolation.py -v`
Expected: `ImportError: cannot import name 'set_tenant_context'`.

- [ ] **Step 3: Add `set_tenant_context` to `src/identity/infrastructure/db.py`**

```python
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def set_tenant_context(session: AsyncSession, tenant_id: uuid.UUID) -> None:
    # NOT `SET LOCAL app.current_tenant_id = :tenant_id` — PostgreSQL's SET
    # statement grammar doesn't accept a bind parameter there at all (this
    # isn't an asyncpg quirk; psycopg2 only appears to allow it because it
    # mogrifies parameters into the query string client-side before sending
    # it, which asyncpg's server-side extended-query binding doesn't do).
    # set_config() is a normal SQL function, so it takes a bound parameter
    # like any other call; its third argument (`true` = "is_local") gives
    # the identical transaction-scoped behavior SET LOCAL would have.
    await session.execute(
        text("SELECT set_config('app.current_tenant_id', :tenant_id, true)"), {"tenant_id": str(tenant_id)}
    )
```

(Append this function and the two new imports to the existing `db.py` from Task 5 — don't replace the file, add to it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_rls_tenant_isolation.py -v`
Expected: `1 passed`. If the tenant-A insert or tenant-B insert itself fails (not the final assertion), the most likely cause is `set_tenant_context` not being called before it, since `sessions` rejects an insert whose context doesn't match the row's own `tenant_id` under `FORCE ROW LEVEL SECURITY`. If the final assertion fails with both titles returned, the RLS policy from Task 5 isn't actually being enforced on this connection — check that the Postgres role `db_session` connects as (`app_user`) isn't accidentally a superuser or the table owner, both of which bypass RLS by default regardless of policy.

- [ ] **Step 5: Commit**

```bash
git add src/identity/infrastructure/db.py tests/integration/test_rls_tenant_isolation.py
git commit -m "test: verify row-level tenant isolation on sessions with no application-level filter"
```

---

### Task 8: Redis refresh token store

**Files:**
- Create: `src/identity/infrastructure/redis_refresh_token_store.py`
- Test: `tests/integration/test_redis_refresh_token_store.py`

**Interfaces:**
- Consumes: `RefreshTokenStore` port, `RefreshToken` entity (Task 2), `redis_url` fixture (Task 5).
- Produces: `RedisRefreshTokenStore(redis_url)`, consumed by Task 11's `RefreshAccessToken`/`RevokeRefreshToken` use cases.

- [ ] **Step 1: Write the failing tests**

```python
# tests/integration/test_redis_refresh_token_store.py
import uuid
from datetime import datetime, timedelta, timezone

from src.identity.domain.entities import RefreshToken
from src.identity.infrastructure.redis_refresh_token_store import RedisRefreshTokenStore


async def test_save_then_get_user_id_returns_the_owning_user(redis_url):
    store = RedisRefreshTokenStore(redis_url)
    token_id = uuid.uuid4()
    user_id = uuid.uuid4()
    token = RefreshToken(token_id=token_id, value="opaque", expires_at=datetime.now(timezone.utc) + timedelta(days=7))

    await store.save(token, user_id)
    assert await store.get_user_id(token_id) == user_id


async def test_get_user_id_returns_none_for_an_unknown_token(redis_url):
    store = RedisRefreshTokenStore(redis_url)
    assert await store.get_user_id(uuid.uuid4()) is None


async def test_delete_makes_the_token_unfindable(redis_url):
    store = RedisRefreshTokenStore(redis_url)
    token_id = uuid.uuid4()
    user_id = uuid.uuid4()
    token = RefreshToken(token_id=token_id, value="opaque", expires_at=datetime.now(timezone.utc) + timedelta(days=7))

    await store.save(token, user_id)
    await store.delete(token_id)
    assert await store.get_user_id(token_id) is None


async def test_save_sets_a_ttl_matching_the_token_lifetime(redis_url):
    import redis.asyncio as redis

    store = RedisRefreshTokenStore(redis_url)
    token_id = uuid.uuid4()
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    token = RefreshToken(token_id=token_id, value="opaque", expires_at=expires_at)
    await store.save(token, uuid.uuid4())

    client = redis.from_url(redis_url)
    ttl = await client.ttl(f"identity:refresh:{token_id}")
    await client.aclose()

    seven_days_seconds = 7 * 24 * 60 * 60
    assert seven_days_seconds - 60 < ttl <= seven_days_seconds
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_redis_refresh_token_store.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/identity/infrastructure/redis_refresh_token_store.py`**

```python
import uuid
from datetime import datetime, timezone

import redis.asyncio as redis

from src.identity.domain.entities import RefreshToken
from src.identity.domain.ports import RefreshTokenStore

_KEY_PREFIX = "identity:refresh:"


class RedisRefreshTokenStore(RefreshTokenStore):
    def __init__(self, redis_url: str) -> None:
        self._client = redis.from_url(redis_url, decode_responses=True)

    async def save(self, refresh_token: RefreshToken, user_id: uuid.UUID) -> None:
        ttl_seconds = int((refresh_token.expires_at - datetime.now(timezone.utc)).total_seconds())
        await self._client.set(
            f"{_KEY_PREFIX}{refresh_token.token_id}",
            str(user_id),
            ex=max(ttl_seconds, 1),
        )

    async def get_user_id(self, token_id: uuid.UUID) -> uuid.UUID | None:
        value = await self._client.get(f"{_KEY_PREFIX}{token_id}")
        return uuid.UUID(value) if value else None

    async def delete(self, token_id: uuid.UUID) -> None:
        await self._client.delete(f"{_KEY_PREFIX}{token_id}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_redis_refresh_token_store.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/identity/infrastructure/redis_refresh_token_store.py tests/integration/test_redis_refresh_token_store.py
git commit -m "feat: add Redis-backed refresh token store"
```

---

### Task 9: Redis rate limiter

**Files:**
- Create: `src/identity/infrastructure/redis_rate_limiter.py`
- Test: `tests/integration/test_redis_rate_limiter.py`

**Interfaces:**
- Consumes: `RateLimiter` port (Task 2), `redis_url` fixture (Task 5).
- Produces: `RedisRateLimiter(redis_url)`, consumed by Task 13's rate-limit dependency on `/auth/register` and `/auth/login`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/integration/test_redis_rate_limiter.py
from src.identity.infrastructure.redis_rate_limiter import RedisRateLimiter


async def test_allows_requests_under_the_limit(redis_url):
    limiter = RedisRateLimiter(redis_url)
    for _ in range(5):
        allowed, remaining, _ = await limiter.check("test-key-a", limit=5, window_seconds=60)
        assert allowed is True
    assert remaining == 0


async def test_blocks_the_request_that_exceeds_the_limit(redis_url):
    limiter = RedisRateLimiter(redis_url)
    for _ in range(5):
        await limiter.check("test-key-b", limit=5, window_seconds=60)
    allowed, remaining, _ = await limiter.check("test-key-b", limit=5, window_seconds=60)
    assert allowed is False
    assert remaining == 0


async def test_different_keys_have_independent_limits(redis_url):
    limiter = RedisRateLimiter(redis_url)
    for _ in range(5):
        await limiter.check("test-key-c", limit=5, window_seconds=60)
    allowed, _, _ = await limiter.check("test-key-d", limit=5, window_seconds=60)
    assert allowed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_redis_rate_limiter.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/identity/infrastructure/redis_rate_limiter.py`**

A fixed-window counter: `INCR` a per-key, per-window counter, set its expiry on first increment.

```python
from datetime import datetime, timedelta, timezone

import redis.asyncio as redis

from src.identity.domain.ports import RateLimiter

_KEY_PREFIX = "identity:ratelimit:"


class RedisRateLimiter(RateLimiter):
    def __init__(self, redis_url: str) -> None:
        self._client = redis.from_url(redis_url, decode_responses=True)

    async def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int, datetime]:
        redis_key = f"{_KEY_PREFIX}{key}"
        count = await self._client.incr(redis_key)
        if count == 1:
            await self._client.expire(redis_key, window_seconds)

        ttl = await self._client.ttl(redis_key)
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=max(ttl, 0))

        if count > limit:
            return False, 0, reset_at
        return True, limit - count, reset_at
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_redis_rate_limiter.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/identity/infrastructure/redis_rate_limiter.py tests/integration/test_redis_rate_limiter.py
git commit -m "feat: add Redis-backed sliding-window rate limiter"
```

---

### Task 10: Register and authenticate use cases

**Files:**
- Create: `src/identity/application/register_user.py`
- Create: `src/identity/application/authenticate_user.py`
- Create: `tests/unit/fakes.py` (fake in-memory ports, shared by this task and Task 11)
- Test: `tests/unit/test_register_user.py`
- Test: `tests/unit/test_authenticate_user.py`

**Interfaces:**
- Consumes: every port from Task 2 (`UserRepository`, `PasswordHasher`, `TokenIssuer`).
- Produces: `RegisterUser`, `AuthenticateUser`, consumed by Task 13's `/auth/register` and `/auth/login` routes. `tests/unit/fakes.py` is also consumed by Task 11.

- [ ] **Step 1: Write `tests/unit/fakes.py`**

```python
import uuid
from datetime import datetime, timedelta, timezone

from src.identity.domain.entities import AccessToken, PasswordHash, RefreshToken, TokenPair, User
from src.identity.domain.ports import PasswordHasher, RefreshTokenStore, TokenIssuer, UserRepository
from src.identity.domain.errors import EmailAlreadyRegistered


class FakeUserRepository(UserRepository):
    def __init__(self) -> None:
        self._by_email: dict[str, User] = {}
        self._by_id: dict[uuid.UUID, User] = {}

    async def save(self, user: User) -> None:
        if user.email in self._by_email:
            raise EmailAlreadyRegistered(user.email)
        self._by_email[user.email] = user
        self._by_id[user.id] = user

    async def find_by_email(self, email: str) -> User | None:
        return self._by_email.get(email)

    async def find_by_id(self, user_id: uuid.UUID) -> User | None:
        return self._by_id.get(user_id)


class FakePasswordHasher(PasswordHasher):
    """Not real Argon2 — deterministic and fast, for use-case-level tests only."""

    def hash(self, plain_password: str) -> PasswordHash:
        return PasswordHash(f"$argon2id$fake${plain_password}")

    def verify(self, plain_password: str, hashed: PasswordHash) -> bool:
        return str(hashed) == f"$argon2id$fake${plain_password}"


class FakeTokenIssuer(TokenIssuer):
    def issue_pair(self, user_id: uuid.UUID, tenant_id: uuid.UUID) -> TokenPair:
        now = datetime.now(timezone.utc)
        return TokenPair(
            access_token=AccessToken(value=f"access-{user_id}", expires_at=now + timedelta(minutes=15)),
            refresh_token=RefreshToken(
                token_id=uuid.uuid4(), value=f"refresh-{user_id}", expires_at=now + timedelta(days=7)
            ),
        )

    def verify_access_token(self, token: str) -> dict:
        raise NotImplementedError("not exercised by the application-layer tests")


class FakeRefreshTokenStore(RefreshTokenStore):
    def __init__(self) -> None:
        self._store: dict[uuid.UUID, uuid.UUID] = {}

    async def save(self, refresh_token: RefreshToken, user_id: uuid.UUID) -> None:
        self._store[refresh_token.token_id] = user_id

    async def get_user_id(self, token_id: uuid.UUID) -> uuid.UUID | None:
        return self._store.get(token_id)

    async def delete(self, token_id: uuid.UUID) -> None:
        self._store.pop(token_id, None)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_register_user.py
import pytest

from src.identity.application.register_user import RegisterUser
from src.identity.domain.errors import EmailAlreadyRegistered
from tests.unit.fakes import FakePasswordHasher, FakeUserRepository


async def test_register_creates_a_user_with_a_hashed_password():
    repo = FakeUserRepository()
    use_case = RegisterUser(user_repository=repo, password_hasher=FakePasswordHasher())

    user = await use_case.execute(email="new@example.com", plain_password="hunter2")

    stored = await repo.find_by_email("new@example.com")
    assert stored is not None
    assert stored.id == user.id
    assert str(stored.hashed_password) != "hunter2"


async def test_register_rejects_a_duplicate_email():
    repo = FakeUserRepository()
    use_case = RegisterUser(user_repository=repo, password_hasher=FakePasswordHasher())
    await use_case.execute(email="dup@example.com", plain_password="hunter2")

    with pytest.raises(EmailAlreadyRegistered):
        await use_case.execute(email="dup@example.com", plain_password="different")


async def test_register_gives_each_new_user_their_own_tenant():
    repo = FakeUserRepository()
    use_case = RegisterUser(user_repository=repo, password_hasher=FakePasswordHasher())

    first = await use_case.execute(email="a@example.com", plain_password="hunter2")
    second = await use_case.execute(email="b@example.com", plain_password="hunter2")

    assert first.tenant_id != second.tenant_id
```

```python
# tests/unit/test_authenticate_user.py
import pytest

from src.identity.application.authenticate_user import AuthenticateUser
from src.identity.application.register_user import RegisterUser
from src.identity.domain.errors import InvalidCredentials
from tests.unit.fakes import FakePasswordHasher, FakeTokenIssuer, FakeUserRepository


async def test_authenticate_succeeds_with_correct_credentials_and_returns_the_user_and_tokens():
    repo = FakeUserRepository()
    hasher = FakePasswordHasher()
    registered = await RegisterUser(user_repository=repo, password_hasher=hasher).execute(
        email="a@example.com", plain_password="hunter2"
    )
    use_case = AuthenticateUser(user_repository=repo, password_hasher=hasher, token_issuer=FakeTokenIssuer())

    user, pair = await use_case.execute(email="a@example.com", plain_password="hunter2")

    assert user.id == registered.id
    assert pair.access_token.value.startswith("access-")


async def test_authenticate_rejects_a_wrong_password():
    repo = FakeUserRepository()
    hasher = FakePasswordHasher()
    await RegisterUser(user_repository=repo, password_hasher=hasher).execute(
        email="a@example.com", plain_password="hunter2"
    )
    use_case = AuthenticateUser(user_repository=repo, password_hasher=hasher, token_issuer=FakeTokenIssuer())

    with pytest.raises(InvalidCredentials):
        await use_case.execute(email="a@example.com", plain_password="wrong")


async def test_authenticate_rejects_an_unknown_email_with_the_same_error_as_a_wrong_password():
    repo = FakeUserRepository()
    hasher = FakePasswordHasher()
    use_case = AuthenticateUser(user_repository=repo, password_hasher=hasher, token_issuer=FakeTokenIssuer())

    with pytest.raises(InvalidCredentials) as unknown_email_exc:
        await use_case.execute(email="nobody@example.com", plain_password="whatever")

    assert str(unknown_email_exc.value) == "Invalid credentials"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_register_user.py tests/unit/test_authenticate_user.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Write `src/identity/application/register_user.py`**

```python
import uuid
from datetime import datetime, timezone

from src.identity.domain.entities import User
from src.identity.domain.ports import PasswordHasher, UserRepository


class RegisterUser:
    def __init__(self, user_repository: UserRepository, password_hasher: PasswordHasher) -> None:
        self._users = user_repository
        self._hasher = password_hasher

    async def execute(self, email: str, plain_password: str) -> User:
        now = datetime.now(timezone.utc)
        user = User(
            id=uuid.uuid4(),
            email=email,
            hashed_password=self._hasher.hash(plain_password),
            tenant_id=uuid.uuid4(),
            created_at=now,
            updated_at=now,
        )
        await self._users.save(user)
        return user
```

- [ ] **Step 5: Write `src/identity/application/authenticate_user.py`**

```python
from src.identity.domain.entities import TokenPair, User
from src.identity.domain.errors import InvalidCredentials
from src.identity.domain.ports import PasswordHasher, TokenIssuer, UserRepository


class AuthenticateUser:
    def __init__(
        self, user_repository: UserRepository, password_hasher: PasswordHasher, token_issuer: TokenIssuer
    ) -> None:
        self._users = user_repository
        self._hasher = password_hasher
        self._tokens = token_issuer

    async def execute(self, email: str, plain_password: str) -> tuple[User, TokenPair]:
        user = await self._users.find_by_email(email)
        if user is None or not self._hasher.verify(plain_password, user.hashed_password):
            raise InvalidCredentials()
        pair = self._tokens.issue_pair(user_id=user.id, tenant_id=user.tenant_id)
        return user, pair
```

The router (Task 13) is the caller that actually needs `user.id` — it uses the returned `User` to persist the refresh token against its real owner, not the token's own id. Returning the `TokenPair` alone here was this plan's original draft and was wrong: nothing about a `TokenPair` identifies whose tokens they are, and the router has no other way to learn it without a second lookup.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_register_user.py tests/unit/test_authenticate_user.py -v`
Expected: `6 passed`.

- [ ] **Step 7: Commit**

```bash
git add src/identity/application/register_user.py src/identity/application/authenticate_user.py tests/unit/fakes.py tests/unit/test_register_user.py tests/unit/test_authenticate_user.py
git commit -m "feat: add register and authenticate use cases"
```

---

### Task 11: Refresh and revoke use cases

**Files:**
- Create: `src/identity/application/refresh_access_token.py`
- Create: `src/identity/application/revoke_refresh_token.py`
- Test: `tests/unit/test_refresh_access_token.py`
- Test: `tests/unit/test_revoke_refresh_token.py`

**Interfaces:**
- Consumes: `TokenIssuer`, `RefreshTokenStore`, `UserRepository` ports (Task 2); `FakeRefreshTokenStore`, `FakeTokenIssuer`, `FakeUserRepository` (Task 10's `tests/unit/fakes.py`).
- Produces: `RefreshAccessToken`, `RevokeRefreshToken`, consumed by Task 13's `/auth/refresh` and `/auth/logout` routes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_refresh_access_token.py
import uuid

import pytest

from src.identity.application.refresh_access_token import RefreshAccessToken
from src.identity.domain.errors import TokenAlreadyUsed
from tests.unit.fakes import FakeRefreshTokenStore, FakeTokenIssuer


async def test_refresh_issues_a_new_pair_and_invalidates_the_old_token():
    store = FakeRefreshTokenStore()
    issuer = FakeTokenIssuer()
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    original_pair = issuer.issue_pair(user_id, tenant_id)
    await store.save(original_pair.refresh_token, user_id)

    use_case = RefreshAccessToken(refresh_token_store=store, token_issuer=issuer)
    new_pair = await use_case.execute(refresh_token_id=original_pair.refresh_token.token_id, tenant_id=tenant_id)

    assert new_pair.access_token.value.startswith("access-")
    assert await store.get_user_id(original_pair.refresh_token.token_id) is None


async def test_refresh_rejects_an_already_used_token():
    store = FakeRefreshTokenStore()
    issuer = FakeTokenIssuer()
    use_case = RefreshAccessToken(refresh_token_store=store, token_issuer=issuer)

    with pytest.raises(TokenAlreadyUsed):
        await use_case.execute(refresh_token_id=uuid.uuid4(), tenant_id=uuid.uuid4())


async def test_refresh_rejects_the_same_token_replayed_a_second_time():
    store = FakeRefreshTokenStore()
    issuer = FakeTokenIssuer()
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    original_pair = issuer.issue_pair(user_id, tenant_id)
    await store.save(original_pair.refresh_token, user_id)
    use_case = RefreshAccessToken(refresh_token_store=store, token_issuer=issuer)

    await use_case.execute(refresh_token_id=original_pair.refresh_token.token_id, tenant_id=tenant_id)

    with pytest.raises(TokenAlreadyUsed):
        await use_case.execute(refresh_token_id=original_pair.refresh_token.token_id, tenant_id=tenant_id)
```

```python
# tests/unit/test_revoke_refresh_token.py
import uuid

from src.identity.application.revoke_refresh_token import RevokeRefreshToken
from tests.unit.fakes import FakeRefreshTokenStore, FakeTokenIssuer


async def test_revoke_deletes_the_token_so_it_can_no_longer_be_used():
    store = FakeRefreshTokenStore()
    issuer = FakeTokenIssuer()
    user_id = uuid.uuid4()
    pair = issuer.issue_pair(user_id, uuid.uuid4())
    await store.save(pair.refresh_token, user_id)

    use_case = RevokeRefreshToken(refresh_token_store=store)
    await use_case.execute(refresh_token_id=pair.refresh_token.token_id)

    assert await store.get_user_id(pair.refresh_token.token_id) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_refresh_access_token.py tests/unit/test_revoke_refresh_token.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/identity/application/refresh_access_token.py`**

```python
import uuid

from src.identity.domain.entities import TokenPair
from src.identity.domain.errors import TokenAlreadyUsed
from src.identity.domain.ports import RefreshTokenStore, TokenIssuer


class RefreshAccessToken:
    def __init__(self, refresh_token_store: RefreshTokenStore, token_issuer: TokenIssuer) -> None:
        self._store = refresh_token_store
        self._tokens = token_issuer

    async def execute(self, refresh_token_id: uuid.UUID, tenant_id: uuid.UUID) -> TokenPair:
        user_id = await self._store.get_user_id(refresh_token_id)
        if user_id is None:
            raise TokenAlreadyUsed()

        await self._store.delete(refresh_token_id)
        new_pair = self._tokens.issue_pair(user_id=user_id, tenant_id=tenant_id)
        await self._store.save(new_pair.refresh_token, user_id)
        return new_pair
```

- [ ] **Step 4: Write `src/identity/application/revoke_refresh_token.py`**

```python
import uuid

from src.identity.domain.ports import RefreshTokenStore


class RevokeRefreshToken:
    def __init__(self, refresh_token_store: RefreshTokenStore) -> None:
        self._store = refresh_token_store

    async def execute(self, refresh_token_id: uuid.UUID) -> None:
        await self._store.delete(refresh_token_id)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_refresh_access_token.py tests/unit/test_revoke_refresh_token.py -v`
Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/identity/application/refresh_access_token.py src/identity/application/revoke_refresh_token.py tests/unit/test_refresh_access_token.py tests/unit/test_revoke_refresh_token.py
git commit -m "feat: add refresh and revoke use cases with rotation-invalidates-reuse behavior"
```

---

### Task 12: FastAPI app skeleton and dependencies

**Files:**
- Create: `src/api/main.py`
- Create: `src/api/dependencies.py`
- Create: `src/api/exception_handlers.py`
- Test: `tests/integration/test_app_dependencies.py`

**Interfaces:**
- Consumes: `get_engine`/`get_sessionmaker`/`set_tenant_context` (Task 5/7), `JWTTokenIssuer` (Task 4), `PostgresUserRepository` (Task 6), `RedisRefreshTokenStore`/`RedisRateLimiter` (Task 8/9).
- Produces: the FastAPI `app` instance, `get_db_session`, `get_current_user` dependencies, consumed by Task 13's router and Task 14's Dockerfile entrypoint.

- [ ] **Step 1: Write `src/api/exception_handlers.py`**

```python
from fastapi import Request
from fastapi.responses import JSONResponse

from src.identity.domain.errors import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    TokenAlreadyUsed,
    TokenExpired,
)


async def invalid_credentials_handler(request: Request, exc: InvalidCredentials) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})


async def email_already_registered_handler(request: Request, exc: EmailAlreadyRegistered) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": "Email already registered"})


async def token_expired_handler(request: Request, exc: TokenExpired) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "Token expired"})


async def token_already_used_handler(request: Request, exc: TokenAlreadyUsed) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})


def register_exception_handlers(app) -> None:
    app.add_exception_handler(InvalidCredentials, invalid_credentials_handler)
    app.add_exception_handler(EmailAlreadyRegistered, email_already_registered_handler)
    app.add_exception_handler(TokenExpired, token_expired_handler)
    app.add_exception_handler(TokenAlreadyUsed, token_already_used_handler)
```

Note: `token_already_used_handler` also returns the generic `"Invalid credentials"` detail rather than `"Token already used or invalid"`, deliberately — the `/auth/refresh` endpoint (Task 13) uses this to avoid telling a caller whether a stale refresh token was well-formed-but-reused versus simply garbage, the same enumeration-avoidance reasoning the spec applies to login.

- [ ] **Step 2: Write `src/api/dependencies.py`**

This reads `APP_DATABASE_URL`, not `DATABASE_URL` — Task 5's migration set up a non-superuser `app_user` role specifically because a superuser or table-owner connection bypasses every row-level-security policy regardless of whether the policy itself is correct. `DATABASE_URL` (the bootstrap superuser) is for Alembic migrations only; every application-level connection, including this one, uses `APP_DATABASE_URL`.

```python
import os
import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from src.identity.domain.errors import TokenExpired
from src.identity.infrastructure.db import get_engine, get_sessionmaker, set_tenant_context
from src.identity.infrastructure.jwt_token_issuer import JWTTokenIssuer
from src.identity.infrastructure.postgres_user_repository import PostgresUserRepository
from src.identity.infrastructure.redis_rate_limiter import RedisRateLimiter
from src.identity.infrastructure.redis_refresh_token_store import RedisRefreshTokenStore

_engine = get_engine(os.environ["APP_DATABASE_URL"])
_sessionmaker = get_sessionmaker(_engine)


def get_token_issuer() -> JWTTokenIssuer:
    return JWTTokenIssuer(secret_key=os.environ["JWT_SECRET_KEY"])


def get_refresh_token_store() -> RedisRefreshTokenStore:
    return RedisRefreshTokenStore(os.environ["REDIS_URL"])


def get_rate_limiter() -> RedisRateLimiter:
    return RedisRateLimiter(os.environ["REDIS_URL"])


async def get_raw_db_session() -> AsyncGenerator[AsyncSession, None]:
    """A session with no tenant context set — only for pre-auth flows like register/login."""
    async with _sessionmaker() as session:
        yield session


async def get_current_user_claims(
    authorization: str | None = Header(default=None), token_issuer: JWTTokenIssuer = Depends(get_token_issuer)
) -> dict:
    # Header(default=None), not Header(...) (a required field): the required
    # form makes FastAPI raise its own RequestValidationError and return a
    # generic 422 the instant the header is absent, before this function body
    # ever runs — bypassing every handler in exception_handlers.py entirely.
    # Accepting None and checking it explicitly keeps the missing-header case
    # on the same path as the malformed-header case below, so both surface
    # as the domain's TokenExpired through the registered handler.
    if authorization is None or not authorization.startswith("Bearer "):
        raise TokenExpired()
    token = authorization.removeprefix("Bearer ")
    return token_issuer.verify_access_token(token)


async def get_db_session(
    claims: dict = Depends(get_current_user_claims),
) -> AsyncGenerator[AsyncSession, None]:
    """A tenant-scoped session for any endpoint behind auth — SET LOCAL runs before the caller sees it."""
    async with _sessionmaker() as session:
        await set_tenant_context(session, uuid.UUID(claims["tenant_id"]))
        yield session


def get_user_repository_unscoped(session: AsyncSession = Depends(get_raw_db_session)) -> PostgresUserRepository:
    return PostgresUserRepository(session)


def get_user_repository_scoped(session: AsyncSession = Depends(get_db_session)) -> PostgresUserRepository:
    return PostgresUserRepository(session)
```

- [ ] **Step 3: Write `src/api/main.py`**

```python
from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers

app = FastAPI(title="Unified RAG x CAG x MAG AI System")
register_exception_handlers(app)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
```

(Task 13 imports and mounts the auth router onto this `app`; this task's own test only checks the dependency chain and the health check.)

- [ ] **Step 4: Write the failing test**

`from src.api.main import app` must NOT be a module-level import in this file. `dependencies.py` reads `APP_DATABASE_URL` at import time to build a module-level singleton engine, and Python caches that import for the rest of the pytest process — whichever test file happens to import `src.api.main` *first* in the session permanently pins that engine to whatever connection string was set at that moment. A module-level import here would break collection outright (the env var isn't set yet at collection time) if nothing is set, or — worse, if a placeholder/fake connection string were set just to dodge that — would silently break every *other* test file that imports `src.api.main` later in the same session (e.g. Task 13's real register/login tests) with a connection to nothing. The fix is to defer the import inside each test function, after setting the environment variables from the real TestContainers fixtures every other integration test in this plan already uses — that way, whichever test file imports first, it's always the same working connection.

```python
# tests/integration/test_app_dependencies.py
import os


async def test_health_endpoint_returns_ok(app_database_url, redis_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from httpx import ASGITransport, AsyncClient

    from src.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_missing_authorization_header_returns_domain_token_expired(app_database_url, redis_url):
    # No auth-gated route exists on `app` yet -- Task 13 adds the router. This
    # throwaway probe app+route exercises get_current_user_claims in isolation
    # to prove a fully-absent Authorization header surfaces as the domain's
    # TokenExpired via the registered handler, not FastAPI's default 422
    # RequestValidationError shape (which is what Header(...)'s required-field
    # marker would produce, bypassing exception_handlers.py entirely).
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from fastapi import Depends, FastAPI
    from httpx import ASGITransport, AsyncClient

    from src.api.dependencies import get_current_user_claims
    from src.api.exception_handlers import register_exception_handlers

    probe_app = FastAPI()
    register_exception_handlers(probe_app)

    @probe_app.get("/protected")
    async def protected(claims: dict = Depends(get_current_user_claims)) -> dict:
        return {"claims": claims}

    transport = ASGITransport(app=probe_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/protected")

    assert response.status_code == 401
    assert response.json() == {"detail": "Token expired"}
```

- [ ] **Step 5: Run test, fix, verify green**

Run: `uv run pytest tests/integration/test_app_dependencies.py -v`
Expected: `2 passed`. Then run the whole suite with no path argument (`uv run pytest`) and confirm it still collects and passes cleanly — this is the specific failure mode the deferred-import fix above closes: a bare, no-argument `pytest` run collects every test file up front, and a module-level import anywhere in that set that needs an unset environment variable breaks collection for the entire session, not just its own file.

- [ ] **Step 6: Commit**

```bash
git add src/api/main.py src/api/dependencies.py src/api/exception_handlers.py tests/integration/test_app_dependencies.py
git commit -m "feat: add FastAPI app skeleton with tenant-scoped session and current-user dependencies"
```

---

### Task 13: Auth router

**Files:**
- Create: `src/api/schemas/__init__.py`
- Create: `src/api/schemas/auth.py`
- Create: `src/api/routers/auth.py`
- Modify: `src/api/main.py` (mount the router and the rate-limit-headers middleware)
- Modify: `src/api/exception_handlers.py` (add the `_RateLimitExceeded` handler)
- Modify: `pyproject.toml` (add `email-validator` — `pydantic.EmailStr`, used by `RegisterRequest`/`LoginRequest` below, raises an import-time error without it; this wasn't identified as a dependency until a schema actually used `EmailStr`, so it lands here rather than in Task 1's original list)
- Test: `tests/integration/test_auth_endpoints.py`

**Interfaces:**
- Consumes: every use case from Task 10/11, every dependency from Task 12.
- Produces: `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh`, `POST /auth/logout` — the full slice this Story delivers.

- [ ] **Step 1: Write `src/api/schemas/auth.py`**

```python
import uuid

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)


class RegisterResponse(BaseModel):
    id: uuid.UUID
    email: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

- [ ] **Step 2: Write `src/api/routers/auth.py`**

```python
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.dependencies import (
    get_raw_db_session,
    get_rate_limiter,
    get_refresh_token_store,
    get_token_issuer,
)
from src.api.schemas.auth import LoginRequest, RegisterRequest, RegisterResponse, TokenResponse
from src.identity.application.authenticate_user import AuthenticateUser
from src.identity.application.refresh_access_token import RefreshAccessToken
from src.identity.application.register_user import RegisterUser
from src.identity.application.revoke_refresh_token import RevokeRefreshToken
from src.identity.domain.errors import TokenAlreadyUsed
from src.identity.infrastructure.argon2_password_hasher import Argon2PasswordHasher
from src.identity.infrastructure.postgres_user_repository import PostgresUserRepository

router = APIRouter(prefix="/auth", tags=["auth"])

_RATE_LIMIT = 5
_RATE_LIMIT_WINDOW_SECONDS = 60
_REFRESH_COOKIE_NAME = "refresh_token"


class _RateLimitExceeded(Exception):
    def __init__(self, limit: int, remaining: int, reset_at) -> None:
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """Reapplies the X-RateLimit-* headers onto whatever response the app ends up returning.

    `_enforce_rate_limit` below sets these same headers directly on its injected
    `response` for the common case, but that only reaches the client when the route
    returns normally. When the rate limiter allows the request and the route then
    raises a DIFFERENT domain exception anyway (e.g. login's `AuthenticateUser.execute()`
    raising `InvalidCredentials` for a wrong password — not just the 429 case),
    FastAPI's exception-handling path builds an entirely new Response from the
    registered handler — one that never saw the injected `response` and so never
    inherits headers written to it. This isn't only the 429 path's problem: ANY
    exception raised after `_enforce_rate_limit` succeeds loses those headers the
    same way, which is why a per-exception fix (carrying the values on
    `_RateLimitExceeded` alone, as the 429 handler below still does for its own
    case) doesn't generalize — every exception type a rate-limited route can raise
    would need its own copy of the same plumbing. Middleware sidesteps that:
    Starlette dispatches registered exception handlers inside `ExceptionMiddleware`,
    which sits *below* any middleware added via `add_middleware`, so `call_next`
    here always hands back the final response — success or handled-exception alike
    — letting this middleware attach the headers stashed on `request.state`
    regardless of which path produced that response.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        headers = getattr(request.state, "rate_limit_headers", None)
        if headers is not None:
            response.headers.update(headers)
        return response


async def _enforce_rate_limit(request: Request, response: Response, route_name: str) -> None:
    limiter = get_rate_limiter()
    client_ip = request.client.host if request.client else "unknown"
    allowed, remaining, reset_at = await limiter.check(
        key=f"{route_name}:{client_ip}", limit=_RATE_LIMIT, window_seconds=_RATE_LIMIT_WINDOW_SECONDS
    )
    if not allowed:
        # Raising here means the route's own successful-response path never
        # runs, so headers set on the injected `response` below would never
        # reach the client — FastAPI's exception handler builds an entirely
        # new JSONResponse for the 429 and does not inherit them. Carry the
        # values on the exception itself instead, and let the handler set
        # them on the response it actually returns.
        raise _RateLimitExceeded(limit=_RATE_LIMIT, remaining=0, reset_at=reset_at)

    headers = {
        "X-RateLimit-Limit": str(_RATE_LIMIT),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": reset_at.isoformat(),
    }
    # Written to both places: directly on `response` covers the normal
    # successful-response path with no extra hop through the middleware, and
    # stashed on `request.state` is what lets RateLimitHeadersMiddleware
    # recover these same values if the route raises a domain exception
    # afterward (see the middleware's own docstring above).
    response.headers.update(headers)
    request.state.rate_limit_headers = headers


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_raw_db_session),
) -> RegisterResponse:
    await _enforce_rate_limit(request, response, "register")
    use_case = RegisterUser(
        user_repository=PostgresUserRepository(session), password_hasher=Argon2PasswordHasher()
    )
    user = await use_case.execute(email=payload.email, plain_password=payload.password)
    await session.commit()
    return RegisterResponse(id=user.id, email=user.email)


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_raw_db_session),
) -> TokenResponse:
    await _enforce_rate_limit(request, response, "login")
    use_case = AuthenticateUser(
        user_repository=PostgresUserRepository(session),
        password_hasher=Argon2PasswordHasher(),
        token_issuer=get_token_issuer(),
    )
    user, pair = await use_case.execute(email=payload.email, plain_password=payload.password)
    await get_refresh_token_store().save(pair.refresh_token, user_id=user.id)
    response.set_cookie(
        _REFRESH_COOKIE_NAME, str(pair.refresh_token.token_id), httponly=True, max_age=7 * 24 * 60 * 60
    )
    return TokenResponse(access_token=pair.access_token.value)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: Request, response: Response, session: AsyncSession = Depends(get_raw_db_session)
) -> TokenResponse:
    import uuid

    token_id_str = request.cookies.get(_REFRESH_COOKIE_NAME)
    if not token_id_str:
        raise TokenAlreadyUsed()

    store = get_refresh_token_store()
    user_id = await store.get_user_id(uuid.UUID(token_id_str))
    if user_id is None:
        raise TokenAlreadyUsed()

    # tenant_id isn't stored alongside the refresh token in this sub-project's schema
    # (deliberately, per the spec — refresh tokens carry no tenant claim); the new
    # access token is issued against the user's own tenant, looked up fresh via the
    # same injected session register/login already use — no ad-hoc engine here.
    user = await PostgresUserRepository(session).find_by_id(user_id)
    if user is None:
        raise TokenAlreadyUsed()

    use_case = RefreshAccessToken(refresh_token_store=store, token_issuer=get_token_issuer())
    new_pair = await use_case.execute(refresh_token_id=uuid.UUID(token_id_str), tenant_id=user.tenant_id)
    response.set_cookie(
        _REFRESH_COOKIE_NAME, str(new_pair.refresh_token.token_id), httponly=True, max_age=7 * 24 * 60 * 60
    )
    return TokenResponse(access_token=new_pair.access_token.value)


@router.post("/logout", status_code=204)
async def logout(request: Request, response: Response) -> None:
    token_id_str = request.cookies.get(_REFRESH_COOKIE_NAME)
    if token_id_str:
        import uuid

        await RevokeRefreshToken(refresh_token_store=get_refresh_token_store()).execute(
            refresh_token_id=uuid.UUID(token_id_str)
        )
    response.delete_cookie(_REFRESH_COOKIE_NAME)
```

Notice the refresh cookie carries `str(pair.refresh_token.token_id)` — the raw UUID — not `pair.refresh_token.value`, the signed JWT `JWTTokenIssuer.issue_pair` also produces. That JWT is computed and then never sent anywhere. This is deliberate, not a leftover: `RedisRefreshTokenStore` keys purely on `token_id` (Task 8), so revocability comes from the key either existing or not in Redis, and a UUIDv4's 122 bits of randomness already make the cookie itself unguessable without a signature on top. The unused `.value` is inert extra work, not a security gap — it's called out here so it doesn't read as a bug during review.

Register a handler for `_RateLimitExceeded` in `src/api/exception_handlers.py` (add this to the file from Task 12). It reads the limit/remaining/reset values off the exception rather than the response, since those were never written to a response object on this path:

```python
async def rate_limit_exceeded_handler(request: Request, exc) -> JSONResponse:
    response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    response.headers["X-RateLimit-Limit"] = str(exc.limit)
    response.headers["X-RateLimit-Remaining"] = str(exc.remaining)
    response.headers["X-RateLimit-Reset"] = exc.reset_at.isoformat()
    return response
```

In `register_exception_handlers`, import `_RateLimitExceeded` from `src.api.routers.auth` **inside the function body** (a deferred import, not a module-level one) and add `app.add_exception_handler(_RateLimitExceeded, rate_limit_exceeded_handler)`. The deferred import matters here specifically: `src/api/routers/auth.py` does not import anything from `exception_handlers.py`, so there's no cycle either way, but keeping this one import local to the function keeps `exception_handlers.py` important-free of the router module at load time, consistent with `main.py` being the only place that imports the router eagerly.

- [ ] **Step 3: Mount the router and the rate-limit-headers middleware in `src/api/main.py`**

Add `from src.api.routers.auth import RateLimitHeadersMiddleware` and `from src.api.routers.auth import router as auth_router`, then `app.add_middleware(RateLimitHeadersMiddleware)` and `app.include_router(auth_router)`, to the existing file from Task 12. The middleware registration matters on its own: without it, `_enforce_rate_limit`'s `request.state.rate_limit_headers` stash (added above) is written but never read, and the 429 case still works (its own exception handler reads the values off the exception directly), but the "allowed, then a later exception" case — e.g. a rate-limit-permitted request to `/auth/login` that then fails with a wrong password — would silently drop its `X-RateLimit-*` headers again.

- [ ] **Step 4: Write the failing integration tests**

```python
# tests/integration/test_auth_endpoints.py
import os

import pytest
import pytest_asyncio
import redis.asyncio as redis
from httpx import ASGITransport, AsyncClient

# src.api.dependencies builds its Postgres engine (and asyncpg connection pool)
# once, at first import, and every test in this module reuses that same cached
# `src.api.main` module (see `_client` below) rather than re-importing it fresh.
# pytest-asyncio's default is a brand-new event loop per test function, but an
# asyncpg connection pool is bound to whichever loop was running when its
# connections were opened — reusing pooled connections from a prior test's
# (now-closed) loop raises "Event loop is closed" deep in asyncpg's Windows
# ProactorEventLoop transport, since the pool's connections outlive the loop
# they were created on. Pinning every test in this module to one shared event
# loop matches the engine's actual lifetime (created once, reused for the
# module) and avoids that mismatch — mirroring how a real uvicorn process runs
# on a single, persistent event loop.
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(autouse=True)
async def _clean_redis_between_tests(redis_url):
    # The rate limiter keys by client IP, and every test in this file calls
    # /auth/login from the same test-client IP against the same session-scoped
    # Redis container — without this, earlier tests' login attempts count
    # against later tests' rate-limit budget, making test order matter. A full
    # flushdb before each test is safe here because this Redis instance exists
    # only for this test session.
    client = redis.from_url(redis_url)
    await client.flushdb()
    yield
    await client.aclose()


async def _client(app_database_url, redis_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from src.api.main import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_register_then_login_returns_a_working_access_token(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        register_response = await client.post(
            "/auth/register", json={"email": "flow@example.com", "password": "hunter2hunter2"}
        )
        assert register_response.status_code == 201

        login_response = await client.post(
            "/auth/login", json={"email": "flow@example.com", "password": "hunter2hunter2"}
        )
        assert login_response.status_code == 200
        assert "access_token" in login_response.json()
        assert "refresh_token" in login_response.cookies


async def test_login_with_wrong_password_returns_generic_401(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        await client.post("/auth/register", json={"email": "wp@example.com", "password": "hunter2hunter2"})
        response = await client.post("/auth/login", json={"email": "wp@example.com", "password": "wrongwrong"})
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid credentials"}


async def test_login_with_unknown_email_returns_the_same_generic_401(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        response = await client.post("/auth/login", json={"email": "ghost@example.com", "password": "whatever1"})
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid credentials"}


async def test_register_rejects_an_invalid_payload(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        response = await client.post("/auth/register", json={"email": "not-an-email", "password": "short"})
        assert response.status_code == 422


async def test_refresh_rotates_the_token_and_the_old_one_cannot_be_replayed(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        await client.post("/auth/register", json={"email": "rot@example.com", "password": "hunter2hunter2"})
        login_response = await client.post(
            "/auth/login", json={"email": "rot@example.com", "password": "hunter2hunter2"}
        )
        old_cookie = login_response.cookies.get("refresh_token")

        refresh_response = await client.post("/auth/refresh", cookies={"refresh_token": old_cookie})
        assert refresh_response.status_code == 200

        replay_response = await client.post("/auth/refresh", cookies={"refresh_token": old_cookie})
        assert replay_response.status_code == 401


async def test_sixth_request_in_a_window_is_rate_limited(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        for i in range(5):
            response = await client.post(
                "/auth/login", json={"email": f"rl{i}@example.com", "password": "wrongwrong"}
            )
            assert "X-RateLimit-Remaining" in response.headers
        sixth = await client.post("/auth/login", json={"email": "rl-sixth@example.com", "password": "wrongwrong"})
        assert sixth.status_code == 429
        assert sixth.headers["X-RateLimit-Remaining"] == "0"
```

- [ ] **Step 5: Run tests to verify they fail, then implement until green**

Run: `uv run pytest tests/integration/test_auth_endpoints.py -v`
Iterate on `src/api/routers/auth.py` and `src/api/exception_handlers.py` until all tests pass. Expected final state: `6 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/api/schemas/ src/api/routers/auth.py src/api/main.py src/api/exception_handlers.py tests/integration/test_auth_endpoints.py pyproject.toml uv.lock
git commit -m "feat: add auth router with register, login, refresh, and logout endpoints"
```

---

### Task 14: Docker Compose stack and end-to-end smoke test

**Files:**
- Create: `docker/Dockerfile.api`
- Create: `docker/docker-compose.yml`
- Test: `tests/integration/test_docker_compose_smoke.py` (documented as a manual-run script, not part of the `pytest` suite CI executes — see Step 4)

**Interfaces:**
- Consumes: the full `src/api/` app built in Tasks 12-13.
- Produces: a runnable local stack (`docker compose up`), the deliverable this Story's acceptance criteria measure end to end.

- [ ] **Step 1: Write `docker/Dockerfile.api`**

```dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv sync --no-dev

COPY src/ src/
COPY alembic.ini .
COPY alembic/ alembic/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write `docker/docker-compose.yml`**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: unified_ai
      POSTGRES_PASSWORD: unified_ai
      POSTGRES_DB: unified_ai
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U unified_ai"]
      interval: 5s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 10

  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    environment:
      # DATABASE_URL is the bootstrap superuser connection — used only for
      # `alembic upgrade head` below, which needs to CREATE TABLE/EXTENSION
      # and provision the app_user role itself. APP_DATABASE_URL is what
      # src/api/dependencies.py actually connects with at runtime: a
      # non-superuser, non-owner role, because PostgreSQL exempts both a
      # superuser and a table owner from row-level security regardless of
      # how correct the policy is (docs/superpowers/plans/2026-08-22-auth-
      # foundation.md, Task 5). Running the API itself as the superuser
      # would make every RLS policy in this system silently inert.
      DATABASE_URL: postgresql+asyncpg://unified_ai:unified_ai@postgres:5432/unified_ai
      APP_DATABASE_URL: postgresql+asyncpg://app_user:${APP_DB_PASSWORD}@postgres:5432/unified_ai
      APP_DB_PASSWORD: ${APP_DB_PASSWORD}
      REDIS_URL: redis://redis:6379/0
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: >
      sh -c "uv run alembic upgrade head && uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
```

- [ ] **Step 3: Bring the stack up and smoke-test it manually**

Run: `JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") APP_DB_PASSWORD=$(python3 -c "import secrets; print(secrets.token_hex(16))") docker compose -f docker/docker-compose.yml up --build -d`
Expected: all three services report healthy (`docker compose -f docker/docker-compose.yml ps`).

Run:
```bash
curl -s -X POST http://localhost:8000/auth/register -H "Content-Type: application/json" \
  -d '{"email":"smoke@example.com","password":"hunter2hunter2"}'
curl -s -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" \
  -d '{"email":"smoke@example.com","password":"hunter2hunter2"}' -c /tmp/cookies.txt
curl -s -X POST http://localhost:8000/auth/refresh -b /tmp/cookies.txt
curl -s -X POST http://localhost:8000/auth/logout -b /tmp/cookies.txt -o /dev/null -w "%{http_code}\n"
```
Expected: register returns `201` with an id/email; login returns `200` with an `access_token` and sets a `refresh_token` cookie; refresh returns `200` with a new `access_token`; logout returns `204`.

- [ ] **Step 4: Record the smoke test as a documented manual script**

Write the four `curl` commands from Step 3 into `tests/integration/test_docker_compose_smoke.py` as a module-level docstring and a `if __name__ == "__main__":` block using `httpx` to run the same sequence against `http://localhost:8000`, explicitly not collected by `pytest` (name it without a `test_` prefix on the file itself being auto-run against a stack `pytest` doesn't manage — keep the `test_` filename for discoverability by humans, but guard its contents so `pytest`'s default collection finds no `test_*` functions inside, only the `__main__` block). This keeps the manual end-to-end proof in the repository without making the CI's `pytest tests/` invocation depend on a Docker Compose stack the CI doesn't stand up in this sub-project.

Run: `docker compose -f docker/docker-compose.yml down -v` once the manual verification above is complete, to leave a clean state.

- [ ] **Step 5: Commit**

```bash
git add docker/ tests/integration/test_docker_compose_smoke.py
git commit -m "feat: add Docker Compose stack and end-to-end smoke test script"
```

---

### Task 15: Documentation sync

**Files:**
- Modify: `docs/architecture/OVERVIEW.md` (Phase 1 module blueprint section)

**Interfaces:**
- None — this task produces no code, only brings the documentation in line with what now exists, per CLAUDE.md's "Documentation stays in sync with the repository" rule.

- [ ] **Step 1: Update the Phase 1 blueprint's opening framing**

In `docs/architecture/OVERVIEW.md`'s "## The Phase 1 module blueprint" section, the sentence "**None of the directories below exist in this repository as of this writing.** There is no `src/`, no `tests/`, no `docker/`, no `k8s/`..." is no longer accurate for the auth slice. Replace it with a sentence that states plainly what now exists — `src/api/`, `src/identity/`, `tests/unit/`, `tests/integration/`, `docker/` (the auth-scoped subset: `Dockerfile.api` and `docker-compose.yml`, not yet `Dockerfile.worker`/`Dockerfile.vllm`/`docker-compose.prod.yml`) — and what's still pending (`src/rag/`, `src/cag/`, `src/mag/`, `src/orchestration/`, `src/workers/`, `k8s/`, `notebooks/`), rather than treating the whole tree as equally hypothetical.

- [ ] **Step 2: Note the identity/api modules the original blueprint didn't name**

Add one sentence after the existing tree diagram (or inline in the surrounding prose) noting that `src/api/` and `src/identity/` exist alongside the paradigm modules the original blueprint diagram shows, added during the Auth Foundation sub-project per `docs/superpowers/specs/2026-08-22-auth-foundation-design.md` because authentication isn't itself one of the three paradigms and needed its own home.

- [ ] **Step 3: Verify the doc-map and placeholder-scan CI checks still pass**

Run: `grep -rnE '\bTBD\b|\bTODO\b|\bFIXME\b' CLAUDE.md README.md docs/ --include='*.md' | grep -vE 'docs/inputs/concepts/|docs/superpowers/'`
Expected: no output.

Run the link-check logic from `.github/workflows/repo-hygiene.yml` locally against `CLAUDE.md` and `docs/README.md` (both files are unmodified by this task, so this should already pass, but confirm rather than assume).

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/OVERVIEW.md
git commit -m "docs: update Phase 1 blueprint to reflect the Auth Foundation code that now exists"
```

---

## After all tasks: finishing this branch

Once Task 15 is complete and the final whole-branch review (per `subagent-driven-development`) is clean:

1. Update GitHub Story #145 and Epic #144: check off the acceptance criteria and Definition-of-Done items that are now genuinely true, and comment the PR/commit range that closes them.
2. Move Story #145's card to `Done` on whichever project board tracks it — none of the six paradigm-combination boards is really the right home for cross-cutting auth infrastructure, which is a real gap worth naming rather than forcing a mismatched fit (see `docs/governance/KANBAN_AUTOMATION.md`'s own precedent for naming automation gaps plainly instead of silently working around them).
3. Run `/graphify` over the repository so `docs/architecture/CONTEXT_GRAPH.md`'s Module/Class/Test File levels — named but left empty since this repository was pre-code — get their first real entries, now that `src/api/` and `src/identity/` exist.
4. Use `superpowers:finishing-a-development-branch` to run the test suite, present the merge/PR/keep options, and — since this is a `feature/*` branch — squash-merge into `develop` per `docs/governance/GIT_WORKFLOW.md`, never into `main` directly.
