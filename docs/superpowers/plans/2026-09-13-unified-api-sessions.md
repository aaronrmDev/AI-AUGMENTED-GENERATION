# Unified API Batch A: Session-Scoped Unified Answer Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `UnifiedAnswerQuestion` over HTTP behind chat sessions the caller owns: `POST /sessions`, `GET /sessions`, and `POST /sessions/{session_id}/answers`, with object-level authorization and per-user rate limiting.

**Architecture:**
- `ChatSession` and its repository port live in `src/identity/domain/`, with two thin use cases in `src/identity/application/` and a Postgres adapter behind migration 0007.
- `AnswerInSession` in `src/orchestration/application/` checks that the session is the caller's before delegating to `UnifiedAnswerQuestion`.
- `src/api/` gains:
  - a shared rate-limit module, moved out of the auth router;
  - a caller type read from the token;
  - request and response schemas;
  - a process-wide pipeline composition with MAG and RAG tiers;
  - the sessions router.

**Tech Stack:** Python 3.14 venv (ruff target py311, mypy strict on `src/`), FastAPI, Pydantic v2, pytest with `asyncio_mode = "auto"`, SQLAlchemy async + asyncpg, Alembic, Redis, TestContainers (`pgvector/pgvector:pg16`, `qdrant/qdrant:v1.16.2`, Redis), httpx `ASGITransport`, sentence-transformers MiniLM.

**Spec:** `docs/superpowers/specs/2026-09-13-unified-api-sessions-design.md`

**Issues:** Story #183 under Epic #182; Tasks #184 to #192, one per task below.

## Global Constraints

- **Location.** Work only inside `.worktrees/feature/183-unified-answer-endpoint/`. Every path below is relative to that directory.
- **Python.** Run it as `../../../.venv/Scripts/python.exe`.
- **Unit tests.** `../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider`. Baseline before this batch: 1137 passed.
- **Integration tests.** `../../../.venv/Scripts/python.exe -m pytest tests/integration/<file> -q -p no:cacheprovider`. They need Docker. Baseline for the whole suite: 257 passed, 8 skipped.
- **Lint and types.** `../../../.venv/Scripts/python.exe -m ruff check <changed files>` and `../../../.venv/Scripts/python.exe -m mypy src`. Line length is 100; wrap any line ruff flags (E501) without changing behaviour.
- **Identity.** The tenant is the access token's `tenant_id` and the user is its `sub`. No request body or query parameter carries either.
- **Isolation.** A session that is missing, another user's, or another tenant's raises `SessionNotFound`, which becomes `404 {"detail": "Session not found"}` in every case.
- **Rate limits.** Answering and `POST /chat` share the key `chat:{user_id}`, 100 per 60 seconds by default (`CHAT_RATE_LIMIT_PER_MINUTE`). `POST /sessions` gets 20 per 60 seconds under `sessions:{user_id}`. Auth keeps 5 per 60 seconds per client IP.
- **Tiers.** The API composes `MagTier` and `RagTier` only, with Concept 5's default `TierTimeouts()`.
- **Units of work.** A repository or recorder called from the answer path opens its own short transaction and sets the tenant context inside it.
- **Commits.** Messages follow `.gitmessage` (Conventional Commits), cite `Refs #183` plus the task's issue, and end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## File map

| File | Responsibility |
|---|---|
| `alembic/versions/0007_sessions_user_index.py` | `ix_sessions_user_id_created_at` |
| `src/identity/domain/entities.py` (modify) | `ChatSession`, `MAX_SESSION_TITLE_CHARS` |
| `src/identity/domain/ports.py` (modify) | `ChatSessionRepository` |
| `src/identity/application/start_chat_session.py` | `StartChatSession` |
| `src/identity/application/list_chat_sessions.py` | `ListChatSessions`, `MAX_SESSIONS_PER_PAGE` |
| `src/identity/infrastructure/postgres_chat_session_repository.py` | `PostgresChatSessionRepository` |
| `src/orchestration/application/answer_in_session.py` | `AnswerInSession`, `SessionQuestionAnswerer` |
| `src/api/rate_limit.py` | `enforce_rate_limit`, `RateLimitExceeded`, `RateLimitHeadersMiddleware`, limits |
| `src/api/caller.py` | `Caller`, `caller_from_claims` |
| `src/api/schemas/sessions.py` | request and response models, `answer_response` |
| `src/api/unified_pipeline.py` | `UnifiedPipeline`, `build_unified_pipeline`, serving thresholds |
| `src/api/dependencies.py` (modify) | `get_caller`, `get_chat_session_repository`, `get_unified_pipeline`, `get_answer_in_session` |
| `src/api/routers/sessions.py` | the three routes |
| `src/api/routers/auth.py` (modify) | uses `src/api/rate_limit.py` |
| `src/api/routers/chat.py` (modify) | `Caller` and the shared chat limit |
| `src/api/exception_handlers.py` (modify) | `SessionNotFound` → 404, `QueryExceedsBudget` → 422 |
| `src/api/main.py` (modify) | sessions router, middleware import, cascade drain on shutdown |
| `tests/unit/session_fakes.py` | `FakeChatSessionRepository` |

---

### Task 1: Migration 0007 (#184)

**Files:**
- Create: `alembic/versions/0007_sessions_user_index.py`
- Test: `tests/integration/test_migration.py` (append)

**Interfaces:**
- Produces: index `ix_sessions_user_id_created_at` on `sessions (user_id, created_at DESC)`.

- [ ] **Step 1: Write the failing test.** Append to `tests/integration/test_migration.py`:

```python
async def test_sessions_are_indexed_by_owner_newest_first(db_session):
    result = await db_session.execute(
        text(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = 'sessions' AND indexname = 'ix_sessions_user_id_created_at'"
        )
    )
    assert result.scalar_one() == (
        "CREATE INDEX ix_sessions_user_id_created_at ON public.sessions "
        "USING btree (user_id, created_at DESC)"
    )
```

- [ ] **Step 2: Run it and confirm it fails.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_migration.py -q -p no:cacheprovider`. Expected: the new test fails with `NoResultFound`.

- [ ] **Step 3: Write the migration.**

```python
"""sessions: index each user's sessions by creation time

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-13

"""
import sqlalchemy as sa

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # GET /sessions lists one user's sessions newest first; RLS supplies the tenant.
    op.create_index(
        "ix_sessions_user_id_created_at",
        "sessions",
        ["user_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_sessions_user_id_created_at", table_name="sessions")
```

- [ ] **Step 4: Run it and confirm it passes.** Run the same command. Expected: all tests in the file pass.

- [ ] **Step 5: Commit.**

```bash
git add alembic/versions/0007_sessions_user_index.py tests/integration/test_migration.py
git commit -m "feat: index sessions by owner and creation time"
```

---

### Task 2: `ChatSession`, its port, and the two session use cases (#185)

**Files:**
- Modify: `src/identity/domain/entities.py` (add `from typing import Any`; append `MAX_SESSION_TITLE_CHARS` and `ChatSession`)
- Modify: `src/identity/domain/ports.py` (import `ChatSession`; append `ChatSessionRepository`)
- Create: `src/identity/application/start_chat_session.py`, `src/identity/application/list_chat_sessions.py`, `tests/unit/session_fakes.py`
- Test: `tests/unit/test_chat_sessions.py`

**Interfaces:**
- Produces:
  - `ChatSession(id: UUID, tenant_id: UUID, user_id: UUID, title: str | None, context_budget: dict[str, Any] | None, created_at: datetime)`
  - `MAX_SESSION_TITLE_CHARS = 200`
  - `ChatSessionRepository.create(tenant_id, user_id, title: str | None) -> ChatSession`
  - `ChatSessionRepository.find_owned(tenant_id, user_id, session_id) -> ChatSession | None`
  - `ChatSessionRepository.list_owned(tenant_id, user_id, limit: int) -> list[ChatSession]`
  - `StartChatSession(repository).execute(tenant_id, user_id, title: str | None) -> ChatSession`
  - `ListChatSessions(repository).execute(tenant_id, user_id, limit: int) -> list[ChatSession]`
  - `MAX_SESSIONS_PER_PAGE = 100`
  - `FakeChatSessionRepository`

- [ ] **Step 1: Write the fake.** `tests/unit/session_fakes.py`:

```python
import uuid
from datetime import UTC, datetime, timedelta

from src.identity.domain.entities import ChatSession
from src.identity.domain.ports import ChatSessionRepository

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


class FakeChatSessionRepository(ChatSessionRepository):
    """In-memory sessions with the port's contract: every read is scoped to one tenant
    and one owning user, and each new session is one second newer than the last."""

    def __init__(self) -> None:
        self.sessions: list[ChatSession] = []

    async def create(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, title: str | None
    ) -> ChatSession:
        session = ChatSession(
            uuid.uuid4(), tenant_id, user_id, title, None,
            _T0 + timedelta(seconds=len(self.sessions)),
        )
        self.sessions.append(session)
        return session

    async def find_owned(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID
    ) -> ChatSession | None:
        return next(
            (
                s for s in self.sessions
                if (s.id, s.tenant_id, s.user_id) == (session_id, tenant_id, user_id)
            ),
            None,
        )

    async def list_owned(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int
    ) -> list[ChatSession]:
        owned = [s for s in self.sessions if (s.tenant_id, s.user_id) == (tenant_id, user_id)]
        return sorted(owned, key=lambda s: s.created_at, reverse=True)[:limit]
```

- [ ] **Step 2: Write the failing tests.** `tests/unit/test_chat_sessions.py`:

```python
import uuid
from datetime import UTC, datetime

import pytest

from src.identity.application.list_chat_sessions import ListChatSessions
from src.identity.application.start_chat_session import StartChatSession
from src.identity.domain.entities import ChatSession
from tests.unit.session_fakes import FakeChatSessionRepository

TENANT, USER = uuid.uuid4(), uuid.uuid4()


def _session(title: str | None) -> ChatSession:
    return ChatSession(uuid.uuid4(), TENANT, USER, title, None, datetime(2026, 1, 1, tzinfo=UTC))


def test_a_title_may_be_200_characters_but_not_201():
    assert _session("x" * 200).title == "x" * 200
    with pytest.raises(ValueError):
        _session("x" * 201)


async def test_starting_a_session_strips_its_title_and_records_the_caller_as_owner():
    repository = FakeChatSessionRepository()

    session = await StartChatSession(repository).execute(TENANT, USER, "  Returns  ")

    assert (session.title, session.tenant_id, session.user_id) == ("Returns", TENANT, USER)
    assert repository.sessions == [session]


@pytest.mark.parametrize("title", [None, "", "   "])
async def test_a_missing_or_blank_title_is_stored_as_none(title):
    session = await StartChatSession(FakeChatSessionRepository()).execute(TENANT, USER, title)
    assert session.title is None


async def test_a_title_too_long_after_stripping_is_refused_before_anything_is_stored():
    repository = FakeChatSessionRepository()

    with pytest.raises(ValueError):
        await StartChatSession(repository).execute(TENANT, USER, " " + "x" * 201 + " ")

    assert repository.sessions == []


async def test_listing_returns_only_the_callers_own_sessions_newest_first():
    repository = FakeChatSessionRepository()
    start = StartChatSession(repository)
    first = await start.execute(TENANT, USER, "first")
    await start.execute(TENANT, uuid.uuid4(), "another user's")
    await start.execute(uuid.uuid4(), USER, "another tenant's")
    second = await start.execute(TENANT, USER, "second")

    listed = await ListChatSessions(repository).execute(TENANT, USER, limit=10)

    assert listed == [second, first]


@pytest.mark.parametrize("limit", [0, 101])
async def test_listing_refuses_a_limit_outside_1_to_100(limit):
    with pytest.raises(ValueError):
        await ListChatSessions(FakeChatSessionRepository()).execute(TENANT, USER, limit)


@pytest.mark.parametrize("limit", [1, 100])
async def test_listing_accepts_the_limits_at_either_bound(limit):
    assert await ListChatSessions(FakeChatSessionRepository()).execute(TENANT, USER, limit) == []
```

- [ ] **Step 3: Run them and confirm they fail.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_chat_sessions.py -q -p no:cacheprovider`. Expected: collection fails with `ImportError` for `ChatSession`.

- [ ] **Step 4: Implement.** Append to `src/identity/domain/entities.py` (and add `from typing import Any` to its imports):

```python
MAX_SESSION_TITLE_CHARS = 200


@dataclass(frozen=True)
class ChatSession:
    """A conversation its owner answers questions in. context_budget is the latest
    turn's context-slice record, which PostgresSessionBudgetRecorder writes."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    title: str | None
    context_budget: dict[str, Any] | None
    created_at: datetime

    def __post_init__(self) -> None:
        if self.title is not None and len(self.title) > MAX_SESSION_TITLE_CHARS:
            raise ValueError(f"a session title is at most {MAX_SESSION_TITLE_CHARS} characters")
```

Append to `src/identity/domain/ports.py`, and extend its entities import to include `ChatSession`:

```python
class ChatSessionRepository(ABC):
    """Every method is scoped to one tenant and one owning user: a session another user
    owns is simply not found, exactly like a session that doesn't exist."""

    @abstractmethod
    async def create(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, title: str | None
    ) -> ChatSession: ...

    @abstractmethod
    async def find_owned(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID
    ) -> ChatSession | None: ...

    @abstractmethod
    async def list_owned(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int
    ) -> list[ChatSession]:
        """Newest first."""
        ...
```

`src/identity/application/start_chat_session.py`:

```python
import uuid

from src.identity.domain.entities import MAX_SESSION_TITLE_CHARS, ChatSession
from src.identity.domain.ports import ChatSessionRepository


class StartChatSession:
    def __init__(self, repository: ChatSessionRepository) -> None:
        self._repository = repository

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, title: str | None
    ) -> ChatSession:
        stripped = (title or "").strip()
        # Checked before the insert, so a refused title never leaves a row behind.
        if len(stripped) > MAX_SESSION_TITLE_CHARS:
            raise ValueError(f"a session title is at most {MAX_SESSION_TITLE_CHARS} characters")
        return await self._repository.create(tenant_id, user_id, stripped or None)
```

`src/identity/application/list_chat_sessions.py`:

```python
import uuid

from src.identity.domain.entities import ChatSession
from src.identity.domain.ports import ChatSessionRepository

MAX_SESSIONS_PER_PAGE = 100


class ListChatSessions:
    def __init__(self, repository: ChatSessionRepository) -> None:
        self._repository = repository

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int
    ) -> list[ChatSession]:
        if not 1 <= limit <= MAX_SESSIONS_PER_PAGE:
            raise ValueError(f"limit must be between 1 and {MAX_SESSIONS_PER_PAGE}")
        return await self._repository.list_owned(tenant_id, user_id, limit)
```

- [ ] **Step 5: Run them and confirm they pass.** Run the Step 3 command, then `mypy src` and ruff on the changed files. Expected: all pass.

- [ ] **Step 6: Commit.**

```bash
git add src/identity/domain/entities.py src/identity/domain/ports.py src/identity/application/start_chat_session.py src/identity/application/list_chat_sessions.py tests/unit/session_fakes.py tests/unit/test_chat_sessions.py
git commit -m "feat: add chat sessions to the identity domain"
```

---

### Task 3: `PostgresChatSessionRepository` (#186)

**Files:**
- Create: `src/identity/infrastructure/postgres_chat_session_repository.py`
- Test: `tests/integration/test_postgres_chat_session_repository.py`

**Interfaces:**
- Consumes: `ChatSession` and `ChatSessionRepository` (Task 2); migration 0007 (Task 1).
- Produces: `PostgresChatSessionRepository(sessionmaker: async_sessionmaker[AsyncSession])`.

- [ ] **Step 1: Write the failing tests.**

```python
import uuid
from datetime import UTC, datetime

from sqlalchemy import text

from src.identity.infrastructure.db import get_sessionmaker
from src.identity.infrastructure.postgres_chat_session_repository import (
    PostgresChatSessionRepository,
)
from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import Paradigm
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


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


def _repository(db_session) -> PostgresChatSessionRepository:
    # The same engine as db_session, through the repository's own sessions: the
    # production shape, where each call owns its transaction.
    return PostgresChatSessionRepository(get_sessionmaker(db_session.bind))


async def test_a_created_session_is_found_by_its_owner_with_server_generated_fields(db_session):
    tenant_id = uuid.uuid4()
    user_id = await _user(db_session, tenant_id)
    repository = _repository(db_session)

    created = await repository.create(tenant_id, user_id, "Returns")

    assert (created.tenant_id, created.user_id, created.title) == (tenant_id, user_id, "Returns")
    assert created.context_budget is None
    assert created.created_at.tzinfo is not None
    assert await repository.find_owned(tenant_id, user_id, created.id) == created


async def test_another_user_in_the_same_tenant_finds_nothing(db_session):
    tenant_id = uuid.uuid4()
    owner, other = await _user(db_session, tenant_id), await _user(db_session, tenant_id)
    repository = _repository(db_session)
    created = await repository.create(tenant_id, owner, None)

    assert await repository.find_owned(tenant_id, other, created.id) is None


async def test_the_owner_under_another_tenant_finds_nothing(db_session):
    tenant_id = uuid.uuid4()
    owner = await _user(db_session, tenant_id)
    repository = _repository(db_session)
    created = await repository.create(tenant_id, owner, None)

    assert await repository.find_owned(uuid.uuid4(), owner, created.id) is None


async def test_listing_returns_the_owners_sessions_newest_first_up_to_the_limit(db_session):
    tenant_id = uuid.uuid4()
    owner = await _user(db_session, tenant_id)
    repository = _repository(db_session)
    created = [await repository.create(tenant_id, owner, f"s{i}") for i in range(3)]

    assert await repository.list_owned(tenant_id, owner, 2) == [created[2], created[1]]


async def test_listing_never_includes_another_users_or_another_tenants_sessions(db_session):
    tenant_id, other_tenant = uuid.uuid4(), uuid.uuid4()
    owner, colleague = await _user(db_session, tenant_id), await _user(db_session, tenant_id)
    stranger = await _user(db_session, other_tenant)
    repository = _repository(db_session)
    mine = await repository.create(tenant_id, owner, "mine")
    await repository.create(tenant_id, colleague, "a colleague's")
    await repository.create(other_tenant, stranger, "another tenant's")

    assert await repository.list_owned(tenant_id, owner, 100) == [mine]
    assert await repository.list_owned(other_tenant, owner, 100) == []


async def test_a_recorded_budget_is_read_back_as_a_dict(db_session):
    tenant_id = uuid.uuid4()
    owner = await _user(db_session, tenant_id)
    repository = _repository(db_session)
    created = await repository.create(tenant_id, owner, None)
    contributing = frozenset({Paradigm.RAG})
    await PostgresSessionBudgetRecorder(get_sessionmaker(db_session.bind)).record(
        tenant_id, owner, created.id, allocate(128_000, contributing), contributing
    )

    found = await repository.find_owned(tenant_id, owner, created.id)

    assert found is not None and found.context_budget is not None
    assert found.context_budget["total"] == 128_000
    assert found.context_budget["contributing"] == ["rag"]
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_postgres_chat_session_repository.py -q -p no:cacheprovider`. Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement.**

```python
import json
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.domain.entities import ChatSession
from src.identity.domain.ports import ChatSessionRepository
from src.identity.infrastructure.db import set_tenant_context

_COLUMNS = "id, tenant_id, user_id, title, context_budget, created_at"


def _chat_session(row: Any) -> ChatSession:
    budget = row.context_budget
    if isinstance(budget, str):  # asyncpg returns jsonb as text unless a codec is registered
        budget = json.loads(budget)
    return ChatSession(row.id, row.tenant_id, row.user_id, row.title, budget, row.created_at)


class PostgresChatSessionRepository(ChatSessionRepository):
    """Chat sessions in the `sessions` table, each call in its own short transaction.

    Owning the transaction follows PostgresSessionBudgetRecorder. AnswerInSession
    checks ownership right before a cascade and a model call that can take seconds, and
    a connection held across them would be one nobody else could use. set_tenant_context
    runs inside each transaction, so the tenant_isolation policy always has a tenant to
    enforce, and every query also matches user_id.
    """

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def create(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, title: str | None
    ) -> ChatSession:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            row = (
                await session.execute(
                    text(
                        "INSERT INTO sessions (user_id, tenant_id, title) "
                        f"VALUES (:user_id, :tenant_id, :title) RETURNING {_COLUMNS}"
                    ),
                    {"user_id": user_id, "tenant_id": tenant_id, "title": title},
                )
            ).one()
        return _chat_session(row)

    async def find_owned(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID
    ) -> ChatSession | None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            row = (
                await session.execute(
                    text(f"SELECT {_COLUMNS} FROM sessions WHERE id = :id AND user_id = :user_id"),
                    {"id": session_id, "user_id": user_id},
                )
            ).one_or_none()
        return None if row is None else _chat_session(row)

    async def list_owned(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int
    ) -> list[ChatSession]:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            rows = (
                await session.execute(
                    text(
                        f"SELECT {_COLUMNS} FROM sessions WHERE user_id = :user_id "
                        "ORDER BY created_at DESC, id DESC LIMIT :limit"
                    ),
                    {"user_id": user_id, "limit": limit},
                )
            ).all()
        return [_chat_session(row) for row in rows]
```

- [ ] **Step 4: Run them and confirm they pass.** Run the Step 2 command, then `mypy src` and ruff on the new files. Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/identity/infrastructure/postgres_chat_session_repository.py tests/integration/test_postgres_chat_session_repository.py
git commit -m "feat: store chat sessions in Postgres under RLS"
```

---

### Task 4: `AnswerInSession` (#187)

**Files:**
- Create: `src/orchestration/application/answer_in_session.py`
- Test: `tests/unit/test_answer_in_session.py`

**Interfaces:**
- Consumes: `ChatSessionRepository` (Task 2); `SessionNotFound` (`src/orchestration/domain/errors.py`); `UnifiedAnswer` (`src/orchestration/application/unified_answer_question.py`).
- Produces: `SessionQuestionAnswerer` (Protocol: `execute(tenant_id, user_id, session_id, question) -> UnifiedAnswer`); `AnswerInSession(sessions, answerer).execute(tenant_id, user_id, session_id, question) -> UnifiedAnswer`.

- [ ] **Step 1: Write the failing tests.**

```python
import uuid

import pytest

from src.orchestration.application.answer_in_session import AnswerInSession
from src.orchestration.domain.errors import SessionNotFound
from tests.unit.session_fakes import FakeChatSessionRepository

TENANT, OWNER = uuid.uuid4(), uuid.uuid4()


class _RecordingAnswerer:
    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID, str]] = []
        self.answer = object()

    async def execute(self, tenant_id, user_id, session_id, question):
        self.calls.append((tenant_id, user_id, session_id, question))
        return self.answer


async def _owned_session(repository: FakeChatSessionRepository) -> uuid.UUID:
    return (await repository.create(TENANT, OWNER, None)).id


async def test_a_question_in_the_callers_own_session_is_answered():
    repository, answerer = FakeChatSessionRepository(), _RecordingAnswerer()
    session_id = await _owned_session(repository)

    result = await AnswerInSession(repository, answerer).execute(TENANT, OWNER, session_id, "q")

    assert result is answerer.answer
    assert answerer.calls == [(TENANT, OWNER, session_id, "q")]


@pytest.mark.parametrize("caller", ["another user", "another tenant", "unknown session"])
async def test_a_session_that_isnt_the_callers_is_refused_before_anything_is_answered(caller):
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
        await AnswerInSession(repository, answerer).execute(tenant_id, user_id, session_id, "q")

    assert answerer.calls == []
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_answer_in_session.py -q -p no:cacheprovider`. Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement.**

```python
import uuid
from typing import Protocol

from src.identity.domain.ports import ChatSessionRepository
from src.orchestration.application.unified_answer_question import UnifiedAnswer
from src.orchestration.domain.errors import SessionNotFound


class SessionQuestionAnswerer(Protocol):
    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> UnifiedAnswer: ...


class AnswerInSession:
    """Answers a question in one of the caller's own chat sessions.

    The session is looked up as the caller's before anything else happens. A session
    that doesn't exist, belongs to another user, or lives in another tenant raises the
    same SessionNotFound, so nothing is embedded, retrieved, or recorded for it, and
    the caller can't tell which of the three it was. PostgresSessionBudgetRecorder's
    own ownership check stays as a second line.
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
```

- [ ] **Step 4: Run them and confirm they pass.** Run the Step 2 command, then `mypy src` and ruff. Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/orchestration/application/answer_in_session.py tests/unit/test_answer_in_session.py
git commit -m "feat: refuse answers in sessions the caller doesn't own"
```

---

### Task 5: The shared rate-limit module (#188)

**Files:**
- Create: `src/api/rate_limit.py`
- Modify: `src/api/routers/auth.py` (remove `_RateLimitExceeded`, `RateLimitHeadersMiddleware`, `_enforce_rate_limit`, `_RATE_LIMIT`, `_RATE_LIMIT_WINDOW_SECONDS`, and the imports only they used), `src/api/exception_handlers.py`, `src/api/main.py`
- Test: `tests/unit/test_rate_limit.py`; `tests/integration/test_auth_endpoints.py` must still pass unchanged.

**Interfaces:**
- Produces:
  - `enforce_rate_limit(request, response, *, limiter: RateLimiter, key: str, limit: int, window_seconds: int = WINDOW_SECONDS) -> None`
  - `RateLimitExceeded(limit, remaining, reset_at)`
  - `RateLimitHeadersMiddleware`
  - `WINDOW_SECONDS = 60`, `AUTH_LIMIT = 5`, `SESSION_CREATE_LIMIT = 20`
  - `chat_rate_limit() -> int`

- [ ] **Step 1: Write the failing tests.**

```python
from datetime import UTC, datetime

import pytest
from fastapi import Request, Response

from src.api.rate_limit import RateLimitExceeded, chat_rate_limit, enforce_rate_limit
from src.identity.domain.ports import RateLimiter

_RESET = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)


class _FakeLimiter(RateLimiter):
    def __init__(self, allowed: bool, remaining: int) -> None:
        self._result = (allowed, remaining, _RESET)
        self.calls: list[tuple[str, int, int]] = []

    async def check(self, key: str, limit: int, window_seconds: int):
        self.calls.append((key, limit, window_seconds))
        return self._result


def _request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/", "headers": []})


async def test_an_allowed_request_carries_its_headers_on_the_response_and_request_state():
    request, response, limiter = _request(), Response(), _FakeLimiter(True, 7)

    await enforce_rate_limit(request, response, limiter=limiter, key="chat:u", limit=10)

    expected = {
        "X-RateLimit-Limit": "10",
        "X-RateLimit-Remaining": "7",
        "X-RateLimit-Reset": _RESET.isoformat(),
    }
    assert {k: response.headers[k] for k in expected} == expected
    assert request.state.rate_limit_headers == expected
    assert limiter.calls == [("chat:u", 10, 60)]


async def test_a_refused_request_raises_with_the_limit_and_the_reset_time():
    with pytest.raises(RateLimitExceeded) as refused:
        await enforce_rate_limit(
            _request(), Response(), limiter=_FakeLimiter(False, 0), key="k", limit=3
        )

    assert (refused.value.limit, refused.value.remaining, refused.value.reset_at) == (3, 0, _RESET)


def test_the_chat_limit_defaults_to_100_and_is_read_from_the_environment_per_call(monkeypatch):
    monkeypatch.delenv("CHAT_RATE_LIMIT_PER_MINUTE", raising=False)
    assert chat_rate_limit() == 100
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "2")
    assert chat_rate_limit() == 2


def test_a_chat_limit_below_1_is_refused(monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "0")
    with pytest.raises(ValueError):
        chat_rate_limit()
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_rate_limit.py -q -p no:cacheprovider`. Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/api/rate_limit.py`.** Move the code out of `auth.py` unchanged apart from the parameters. The block below marks three places with a `moved verbatim` comment. Paste the corresponding text from `auth.py` there: `RateLimitHeadersMiddleware`'s docstring, and the two comments inside `_enforce_rate_limit`. Don't keep the marker comments themselves.

```python
import os
from collections.abc import Awaitable, Callable
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.identity.domain.ports import RateLimiter

WINDOW_SECONDS = 60
AUTH_LIMIT = 5  # per client IP per window, on /auth/register and /auth/login
SESSION_CREATE_LIMIT = 20  # per user per window, on POST /sessions
_DEFAULT_CHAT_LIMIT = 100


def chat_rate_limit() -> int:
    """Requests per user per window, shared by POST /chat and answering in a session.

    Read per call, like COOKIE_SECURE, so an operator or a test can change it without
    controlling module import order.
    """
    limit = int(os.environ.get("CHAT_RATE_LIMIT_PER_MINUTE", str(_DEFAULT_CHAT_LIMIT)))
    if limit < 1:
        raise ValueError("CHAT_RATE_LIMIT_PER_MINUTE must be at least 1")
    return limit


class RateLimitExceeded(Exception):
    def __init__(self, limit: int, remaining: int, reset_at: datetime) -> None:
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    # Docstring moved verbatim from src/api/routers/auth.py.

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        headers = getattr(request.state, "rate_limit_headers", None)
        if headers is not None:
            response.headers.update(headers)
        return response


async def enforce_rate_limit(
    request: Request,
    response: Response,
    *,
    limiter: RateLimiter,
    key: str,
    limit: int,
    window_seconds: int = WINDOW_SECONDS,
) -> None:
    allowed, remaining, reset_at = await limiter.check(
        key=key, limit=limit, window_seconds=window_seconds
    )
    if not allowed:
        # Comment moved verbatim from auth.py's _enforce_rate_limit.
        raise RateLimitExceeded(limit=limit, remaining=0, reset_at=reset_at)
    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": reset_at.isoformat(),
    }
    # Comment moved verbatim from auth.py's _enforce_rate_limit.
    response.headers.update(headers)
    request.state.rate_limit_headers = headers
```

In `src/api/routers/auth.py`, replace the removed helper with a thin wrapper, and keep `register` and `login` calling `_enforce_rate_limit(request, response, "register")` and `"login"` as they do now:

```python
from src.api.rate_limit import AUTH_LIMIT, enforce_rate_limit


async def _enforce_rate_limit(request: Request, response: Response, route_name: str) -> None:
    client_ip = request.client.host if request.client else "unknown"
    await enforce_rate_limit(
        request, response, limiter=get_rate_limiter(), key=f"{route_name}:{client_ip}",
        limit=AUTH_LIMIT,
    )
```

In `src/api/exception_handlers.py`, replace the `TYPE_CHECKING` block and the deferred import with a plain `from src.api.rate_limit import RateLimitExceeded`. Type `rate_limit_exceeded_handler`'s `exc` as `RateLimitExceeded`, and register the handler for `RateLimitExceeded`. In `src/api/main.py`, import `RateLimitHeadersMiddleware` from `src.api.rate_limit`.

- [ ] **Step 4: Run the tests.** Run the Step 2 command, then `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_auth_endpoints.py -q -p no:cacheprovider`, then `mypy src` and ruff on the changed files. Expected: all pass, including `test_sixth_request_in_a_window_is_rate_limited`.

- [ ] **Step 5: Commit.**

```bash
git add src/api/rate_limit.py src/api/routers/auth.py src/api/exception_handlers.py src/api/main.py tests/unit/test_rate_limit.py
git commit -m "refactor: share the rate limiter beyond the auth routes"
```

---

### Task 6: `Caller`, the session schemas, and `answer_response` (#189)

**Files:**
- Create: `src/api/caller.py`, `src/api/schemas/sessions.py`
- Test: `tests/unit/test_caller.py`, `tests/unit/test_session_schemas.py`

**Interfaces:**
- Consumes: `ChatSession`, `MAX_SESSION_TITLE_CHARS` (Task 2); `UnifiedAnswer`; `TokenExpired`.
- Produces:
  - `Caller(tenant_id: UUID, user_id: UUID)` and `caller_from_claims(claims: dict[str, Any]) -> Caller`
  - `CreateSessionRequest`, `SessionResponse` (with `SessionResponse.of(session)`), `SessionListResponse`
  - `AnswerRequest`, `AnswerResponse`, `SourceSchema`, `RoutingSchema`, `AttemptSchema`
  - `answer_response(answer: UnifiedAnswer) -> AnswerResponse`, and `MAX_QUESTION_CHARS = 4000`

- [ ] **Step 1: Write the failing tests.** `tests/unit/test_caller.py`:

```python
import uuid

import pytest

from src.api.caller import Caller, caller_from_claims
from src.identity.domain.errors import TokenExpired


def test_the_caller_is_the_tokens_subject_in_the_tokens_tenant():
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    claims = {"sub": str(user_id), "tenant_id": str(tenant_id), "type": "access"}
    assert caller_from_claims(claims) == Caller(tenant_id=tenant_id, user_id=user_id)


@pytest.mark.parametrize(
    "claims", [{"tenant_id": str(uuid.uuid4())}, {"sub": "not-a-uuid", "tenant_id": "x"}]
)
def test_claims_without_a_usable_subject_or_tenant_are_an_invalid_token(claims):
    with pytest.raises(TokenExpired):
        caller_from_claims(claims)
```

`tests/unit/test_session_schemas.py`:

```python
import uuid

import pytest
from pydantic import ValidationError

from src.api.schemas.sessions import AnswerRequest, CreateSessionRequest, answer_response
from src.orchestration.application.unified_answer_question import StageTimings, UnifiedAnswer
from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import (
    ContextItem,
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierAttempt,
    TierOutcome,
)


def test_a_title_may_be_absent_or_up_to_200_characters():
    assert CreateSessionRequest().title is None
    assert CreateSessionRequest(title="x" * 200).title == "x" * 200
    with pytest.raises(ValidationError):
        CreateSessionRequest(title="x" * 201)


@pytest.mark.parametrize("question", ["", "   ", "x" * 4001])
def test_a_blank_or_oversized_question_is_refused(question):
    with pytest.raises(ValidationError):
        AnswerRequest(question=question)


def test_a_question_of_4000_characters_is_accepted():
    assert len(AnswerRequest(question="x" * 4000).question) == 4000


def _answer(decision: RoutingDecision | None, fallback, degraded: bool) -> UnifiedAnswer:
    source_id = uuid.UUID(int=7)
    contributing = frozenset({Paradigm.MAG, Paradigm.RAG})
    return UnifiedAnswer(
        answer="forty-five days",
        sources=[ContextItem(Paradigm.RAG, "Returns within forty-five days.", 0.8, source_id)],
        decision=decision,
        routing_fallback=fallback,
        attempts=[
            TierAttempt(Paradigm.MAG, TierOutcome.TIMEOUT, 50.4),
            TierAttempt(Paradigm.RAG, TierOutcome.HIT, 12.5),
        ],
        allocation=allocate(128_000, contributing),
        dropped={Paradigm.RAG: 1},
        degraded=degraded,
        timings=StageTimings(9.0, 1.0, 60.0, 0.5, 2.0, 900.0),
    )


def test_an_answer_maps_provenance_routing_and_degradation():
    decision = RoutingDecision(
        frozenset({Paradigm.RAG, Paradigm.MAG}), RoutingMode.PARALLEL, {Paradigm.RAG: 0.9}
    )

    response = answer_response(_answer(decision, "classifier_timeout", degraded=True))

    assert response.model_dump(mode="json") == {
        "answer": "forty-five days",
        "sources": [
            {
                "paradigm": "rag",
                "content": "Returns within forty-five days.",
                "source_id": "00000000-0000-0000-0000-000000000007",
            }
        ],
        "routing": {"paradigms": ["mag", "rag"], "mode": "parallel", "fallback": "classifier_timeout"},
        "attempts": [
            {"paradigm": "mag", "outcome": "timeout", "elapsed_ms": 50.4},
            {"paradigm": "rag", "outcome": "hit", "elapsed_ms": 12.5},
        ],
        "degraded": True,
        "dropped": {"rag": 1},
    }


def test_an_unrouted_answer_has_no_routing():
    assert answer_response(_answer(None, None, degraded=False)).routing is None
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_caller.py tests/unit/test_session_schemas.py -q -p no:cacheprovider`. Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** `src/api/caller.py`:

```python
import uuid
from dataclasses import dataclass
from typing import Any

from src.identity.domain.errors import TokenExpired


@dataclass(frozen=True)
class Caller:
    tenant_id: uuid.UUID
    user_id: uuid.UUID


def caller_from_claims(claims: dict[str, Any]) -> Caller:
    """The caller is the verified token's subject, in the token's tenant, and nothing a
    client sends besides the token. Claims this issuer would never mint surface as the
    same invalid-token error a bad signature does."""
    try:
        return Caller(tenant_id=uuid.UUID(claims["tenant_id"]), user_id=uuid.UUID(claims["sub"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise TokenExpired() from exc
```

`src/api/schemas/sessions.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.identity.domain.entities import MAX_SESSION_TITLE_CHARS, ChatSession
from src.orchestration.application.unified_answer_question import UnifiedAnswer
from src.orchestration.domain.entities import PARADIGM_ORDER

MAX_QUESTION_CHARS = 4000


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_SESSION_TITLE_CHARS)


class SessionResponse(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime
    # The latest turn's context-slice record, or None before the first answer.
    context_budget: dict[str, Any] | None

    @classmethod
    def of(cls, session: ChatSession) -> SessionResponse:
        return cls(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            context_budget=session.context_budget,
        )


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]


class AnswerRequest(BaseModel):
    # Bounded for the reason ChatRequest is: a question is embedded and forwarded to the
    # chat model at the caller's discretion.
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class SourceSchema(BaseModel):
    paradigm: str
    content: str
    source_id: uuid.UUID | None


class RoutingSchema(BaseModel):
    paradigms: list[str]
    mode: str
    fallback: str | None


class AttemptSchema(BaseModel):
    paradigm: str
    outcome: str
    elapsed_ms: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceSchema]
    routing: RoutingSchema | None
    attempts: list[AttemptSchema]
    degraded: bool
    dropped: dict[str, int]


def answer_response(answer: UnifiedAnswer) -> AnswerResponse:
    """Provenance, routing, and degradation, which a client shows a user. Router scores,
    similarity scores, and stage timings stay server-side (spec decision 8)."""
    decision = answer.decision
    routing = (
        None
        if decision is None
        else RoutingSchema(
            paradigms=[p.value for p in PARADIGM_ORDER if p in decision.paradigms],
            mode=decision.mode.value,
            fallback=answer.routing_fallback,
        )
    )
    return AnswerResponse(
        answer=answer.answer,
        sources=[
            SourceSchema(paradigm=item.paradigm.value, content=item.content, source_id=item.source_id)
            for item in answer.sources
        ],
        routing=routing,
        attempts=[
            AttemptSchema(
                paradigm=attempt.paradigm.value,
                outcome=attempt.outcome.value,
                elapsed_ms=attempt.elapsed_ms,
            )
            for attempt in answer.attempts
        ],
        degraded=answer.degraded,
        dropped={paradigm.value: count for paradigm, count in answer.dropped.items()},
    )
```

- [ ] **Step 4: Run them and confirm they pass.** Run the Step 2 command, then `mypy src` and ruff. Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/api/caller.py src/api/schemas/sessions.py tests/unit/test_caller.py tests/unit/test_session_schemas.py
git commit -m "feat: add the caller and session request and response schemas"
```

---

### Task 7: The unified pipeline composition and its dependencies (#190)

**Files:**
- Create: `src/api/unified_pipeline.py`
- Modify: `src/api/dependencies.py`
- Test: `tests/unit/test_unified_pipeline.py`

**Interfaces:**
- Consumes: `AnswerInSession` (Task 4); `PostgresChatSessionRepository` (Task 3); `Caller`, `caller_from_claims` (Task 6).
- Produces:
  - `UnifiedPipeline(answer_in_session: AnswerInSession, cascade: LatencyCascade)`
  - `build_unified_pipeline(*, sessionmaker, sessions, embedding_model, vector_store, chat_model) -> UnifiedPipeline`
  - `MAG_HIT`, `MAG_PARTIAL`, `RAG_TOP_K`
  - In `dependencies`: `get_caller(claims) -> Caller`, `get_chat_session_repository() -> PostgresChatSessionRepository`, `get_unified_pipeline() -> UnifiedPipeline` (cached), `get_answer_in_session() -> AnswerInSession`

- [ ] **Step 1: Write the failing tests.**

```python
import uuid

import pytest

from evaluation.scenarios.orchestration_meta_layer_thresholds import MAG_HIT, MAG_PARTIAL
from src.api import unified_pipeline
from src.orchestration.domain.entities import Paradigm
from src.orchestration.domain.errors import SessionNotFound
from tests.unit.rag_fakes import FakeChatModel, FakeEmbeddingModel, FakeVectorStore
from tests.unit.session_fakes import FakeChatSessionRepository


def _pipeline(sessions=None):
    return unified_pipeline.build_unified_pipeline(
        sessionmaker=object(),  # only a real query uses it; these tests never run one
        sessions=sessions or FakeChatSessionRepository(),
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
        chat_model=FakeChatModel(),
    )


def test_the_serving_thresholds_match_the_measured_ones():
    assert (unified_pipeline.MAG_HIT, unified_pipeline.MAG_PARTIAL) == (MAG_HIT, MAG_PARTIAL)


def test_the_cascade_serves_mag_and_rag_and_no_cag_tier():
    # LatencyCascade has no public accessor for its tiers; this reads the private map
    # rather than add one just for a test.
    assert set(_pipeline().cascade._tiers) == {Paradigm.MAG, Paradigm.RAG}


async def test_the_composed_path_refuses_a_session_before_answering():
    with pytest.raises(SessionNotFound):
        await _pipeline().answer_in_session.execute(
            uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), "What is the return policy?"
        )
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_unified_pipeline.py -q -p no:cacheprovider`. Expected: `ImportError`. If `MAG_HIT` or `MAG_PARTIAL` in the evaluation module differ from 0.42 and 0.37, use its values in Step 3.

- [ ] **Step 3: Implement.** `src/api/unified_pipeline.py`:

```python
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.domain.ports import ChatSessionRepository
from src.orchestration.application.answer_in_session import AnswerInSession
from src.orchestration.application.cascade_tiers import MagTier, RagTier
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)
from src.orchestration.infrastructure.prototype_query_classifier import (
    PrototypeQueryClassifier,
)
from src.orchestration.infrastructure.session_scoped_semantic_fact_search import (
    SessionScopedSemanticFactSearch,
)
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.ports import ChatModel, EmbeddingModel, VectorStore
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel

# Measured on the orchestration integration corpus, in
# evaluation/scenarios/orchestration_meta_layer_thresholds.py. src/ can't import
# evaluation/, so the serving values live here, and a unit test keeps them equal.
MAG_HIT = 0.42
MAG_PARTIAL = 0.37
RAG_TOP_K = 5


@dataclass(frozen=True)
class UnifiedPipeline:
    answer_in_session: AnswerInSession
    cascade: LatencyCascade


def build_unified_pipeline(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    sessions: ChatSessionRepository,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    chat_model: ChatModel,
) -> UnifiedPipeline:
    """The session-scoped unified answer path, as the API serves it.

    MAG and RAG tiers only. A CAG tier needs a warmed frozen cache in this process and a
    worker to warm it, and neither exists yet, so the cascade answers a CAG-only route
    from RAG (spec decision 5). One CachingEmbeddingModel serves the question's
    embedding, the router, and SearchDocuments, so the RAG tier's embedding of the
    question is a lookup.
    """
    embedder = CachingEmbeddingModel(embedding_model)
    cascade = LatencyCascade(
        [
            MagTier(
                SessionScopedSemanticFactSearch(sessionmaker),
                hit_threshold=MAG_HIT,
                partial_threshold=MAG_PARTIAL,
            ),
            RagTier(SearchDocuments(embedder, vector_store), top_k=RAG_TOP_K),
        ],
        TierTimeouts(),
    )
    answerer = UnifiedAnswerQuestion(
        embedder,
        PrototypeQueryClassifier(embedder),
        cascade,
        chat_model,
        budget_recorder=PostgresSessionBudgetRecorder(sessionmaker),
    )
    return UnifiedPipeline(AnswerInSession(sessions, answerer), cascade)
```

Append to `src/api/dependencies.py` (add `import functools` and the new imports; place it after `get_chat_model`):

```python
async def get_caller(claims: dict[str, Any] = Depends(get_current_user_claims)) -> Caller:
    return caller_from_claims(claims)


def get_chat_session_repository() -> PostgresChatSessionRepository:
    return PostgresChatSessionRepository(_sessionmaker)


@functools.cache
def get_unified_pipeline() -> UnifiedPipeline:
    # Built on first use, not at import, so importing the app doesn't embed every routing
    # exemplar or open a chat-model client.
    return build_unified_pipeline(
        sessionmaker=_sessionmaker,
        sessions=get_chat_session_repository(),
        embedding_model=_embedding_model,
        vector_store=_vector_store,
        chat_model=get_chat_model(),
    )


def get_answer_in_session() -> AnswerInSession:
    return get_unified_pipeline().answer_in_session
```

- [ ] **Step 4: Run them and confirm they pass.** Run the Step 2 command, then `mypy src` and ruff. Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/api/unified_pipeline.py src/api/dependencies.py tests/unit/test_unified_pipeline.py
git commit -m "feat: compose the unified answer path for the API"
```

---

### Task 8: The sessions router, error mapping, and the endpoint tests (#191)

**Files:**
- Create: `src/api/routers/sessions.py`
- Modify: `src/api/routers/chat.py`, `src/api/exception_handlers.py`, `src/api/main.py`
- Test: `tests/integration/test_sessions_endpoints.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `POST /sessions` (201 `SessionResponse`), `GET /sessions?limit=` (`SessionListResponse`), `POST /sessions/{session_id}/answers` (`AnswerResponse`).

- [ ] **Step 1: Write the failing tests.**

```python
import json
import os
import uuid
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.identity.infrastructure.jwt_token_issuer import JWTTokenIssuer
from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.rag.domain.entities import Chunk
from tests.integration.orchestration_env import VALID_HASH, ContextEchoChatModel

# One event loop for the module, matching the app's module-level engine (see
# test_documents_endpoints.py).
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
    # The real composition with a context-echo model: every store, the router, and the
    # budget recorder are real; only generation is replaced, so the answer shows the
    # context it was given.
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


async def _stored_budget(db_session, tenant_id: uuid.UUID, session_id: str):
    await set_tenant_context(db_session, tenant_id)
    value = (
        await db_session.execute(
            text("SELECT context_budget FROM sessions WHERE id = :id"), {"id": uuid.UUID(session_id)}
        )
    ).scalar_one()
    await db_session.commit()
    return json.loads(value) if isinstance(value, str) else value


async def _seed_policy(tenant_id: uuid.UUID, embedding_model) -> None:
    from src.api.dependencies import get_vector_store

    chunk = Chunk(uuid.uuid4(), uuid.uuid4(), _POLICY, embedding_model.embed(_POLICY))
    await get_vector_store().upsert(chunk, tenant_id)


async def test_every_sessions_route_refuses_a_request_without_a_token(
    app_database_url, redis_url, qdrant_url, embedding_model
):
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        responses = [
            await client.post("/sessions", json={}),
            await client.get("/sessions"),
            await client.post(f"/sessions/{uuid.uuid4()}/answers", json={"question": "q"}),
        ]
    assert [r.status_code for r in responses] == [401, 401, 401]


async def test_created_sessions_are_listed_for_their_owner_newest_first(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        first = await client.post("/sessions", json={"title": "first"}, headers=headers)
        second = await client.post("/sessions", json={"title": "  second  "}, headers=headers)
        listed = await client.get("/sessions", headers=headers)

    assert (first.status_code, second.status_code) == (201, 201)
    assert first.headers["X-RateLimit-Limit"] == "20"
    assert [s["id"] for s in listed.json()["sessions"]] == [second.json()["id"], first.json()["id"]]
    assert [s["title"] for s in listed.json()["sessions"]] == ["second", "first"]
    assert all(s["context_budget"] is None for s in listed.json()["sessions"])


async def test_answering_in_an_owned_session_routes_cascades_and_records_the_budget(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    user_id = await _user(db_session, tenant_id)
    await set_tenant_context(db_session, tenant_id)
    await PostgresSemanticMemoryRepository(db_session).save(
        SemanticMemory(
            id=uuid.uuid4(), user_id=user_id, fact_key="preferred_shipping_speed",
            fact_value="The user prefers expedited two-day shipping.",
            embedding=embedding_model.embed("The user prefers expedited two-day shipping."),
        ),
        tenant_id,
    )
    await db_session.commit()
    await _seed_policy(tenant_id, embedding_model)
    headers = _auth(user_id, tenant_id)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        response = await client.post(
            f"/sessions/{session_id}/answers",
            json={"question": "What is the return policy for unopened items?"},
            headers=headers,
        )
        listed = await client.get("/sessions", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert "forty-five days" in body["answer"]
    assert any(s["paradigm"] == "rag" and "forty-five" in s["content"] for s in body["sources"])
    assert body["routing"]["mode"] in {"cascade", "parallel"}
    assert "rag" in {a["paradigm"] for a in body["attempts"]}
    assert "cag" not in {a["paradigm"] for a in body["attempts"]}
    assert response.headers["X-RateLimit-Limit"] == "100"
    assert (await _stored_budget(db_session, tenant_id, session_id))["total"] == 128_000
    assert listed.json()["sessions"][0]["context_budget"]["total"] == 128_000


async def test_another_users_session_in_the_same_tenant_is_not_found_and_left_untouched(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    owner, intruder = await _user(db_session, tenant_id), await _user(db_session, tenant_id)
    await _seed_policy(tenant_id, embedding_model)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (
            await client.post("/sessions", json={}, headers=_auth(owner, tenant_id))
        ).json()["id"]
        response = await client.post(
            f"/sessions/{session_id}/answers",
            json={"question": "What is the return policy?"},
            headers=_auth(intruder, tenant_id),
        )

    assert (response.status_code, response.json()) == (404, _NOT_FOUND)
    assert await _stored_budget(db_session, tenant_id, session_id) is None


async def test_another_tenant_and_an_unknown_id_get_the_identical_404(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id, other_tenant = uuid.uuid4(), uuid.uuid4()
    owner = await _user(db_session, tenant_id)
    stranger = await _user(db_session, other_tenant)
    question = {"question": "What is the return policy?"}
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (
            await client.post("/sessions", json={}, headers=_auth(owner, tenant_id))
        ).json()["id"]
        foreign = await client.post(
            f"/sessions/{session_id}/answers", json=question, headers=_auth(stranger, other_tenant)
        )
        unknown = await client.post(
            f"/sessions/{uuid.uuid4()}/answers", json=question, headers=_auth(owner, tenant_id)
        )

    assert (foreign.status_code, foreign.content) == (unknown.status_code, unknown.content)
    assert (foreign.status_code, foreign.json()) == (404, _NOT_FOUND)
    assert await _stored_budget(db_session, tenant_id, session_id) is None


async def test_listing_never_shows_another_users_or_another_tenants_sessions(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id, other_tenant = uuid.uuid4(), uuid.uuid4()
    owner, colleague = await _user(db_session, tenant_id), await _user(db_session, tenant_id)
    stranger = await _user(db_session, other_tenant)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        mine = await client.post("/sessions", json={"title": "mine"}, headers=_auth(owner, tenant_id))
        await client.post("/sessions", json={}, headers=_auth(colleague, tenant_id))
        await client.post("/sessions", json={}, headers=_auth(stranger, other_tenant))
        listed = await client.get("/sessions", headers=_auth(owner, tenant_id))

    assert [s["id"] for s in listed.json()["sessions"]] == [mine.json()["id"]]


async def test_the_chat_limit_is_shared_between_answering_and_post_chat(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    os.environ["CHAT_RATE_LIMIT_PER_MINUTE"] = "2"
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    await _seed_policy(tenant_id, embedding_model)
    question = {"question": "What is the return policy?"}
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        answers = [
            await client.post(f"/sessions/{session_id}/answers", json=question, headers=headers)
            for _ in range(3)
        ]
        # Refused before its RAG-only pipeline runs, so no chat model is needed.
        chat = await client.post("/chat", json=question, headers=headers)

    assert [a.status_code for a in answers] == [200, 200, 429]
    assert answers[2].headers["X-RateLimit-Remaining"] == "0"
    assert chat.status_code == 429


async def test_creating_sessions_is_limited_to_20_a_minute_per_user(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        statuses = [
            (await client.post("/sessions", json={}, headers=headers)).status_code
            for _ in range(21)
        ]
    assert statuses == [201] * 20 + [429]


async def test_invalid_requests_are_refused_with_422(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        responses = [
            await client.post(f"/sessions/{session_id}/answers", json={"question": "   "}, headers=headers),
            await client.post(f"/sessions/{session_id}/answers", json={"question": "x" * 4001}, headers=headers),
            await client.post("/sessions", json={"title": "x" * 201}, headers=headers),
            await client.get("/sessions?limit=0", headers=headers),
            await client.get("/sessions?limit=101", headers=headers),
            await client.post("/sessions/not-a-uuid/answers", json={"question": "q"}, headers=headers),
        ]
    assert [r.status_code for r in responses] == [422] * 6
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_sessions_endpoints.py -q -p no:cacheprovider`. Expected: the route tests fail with 404 for `/sessions`, which isn't routed yet.

- [ ] **Step 3: Implement the router.** `src/api/routers/sessions.py`:

```python
import uuid

from fastapi import APIRouter, Depends, Query, Request, Response

from src.api.caller import Caller
from src.api.dependencies import (
    get_answer_in_session,
    get_caller,
    get_chat_session_repository,
    get_rate_limiter,
)
from src.api.rate_limit import SESSION_CREATE_LIMIT, chat_rate_limit, enforce_rate_limit
from src.api.schemas.sessions import (
    AnswerRequest,
    AnswerResponse,
    CreateSessionRequest,
    SessionListResponse,
    SessionResponse,
    answer_response,
)
from src.identity.application.list_chat_sessions import MAX_SESSIONS_PER_PAGE, ListChatSessions
from src.identity.application.start_chat_session import StartChatSession
from src.identity.domain.ports import ChatSessionRepository
from src.orchestration.application.answer_in_session import AnswerInSession

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    payload: CreateSessionRequest,
    request: Request,
    response: Response,
    caller: Caller = Depends(get_caller),
    sessions: ChatSessionRepository = Depends(get_chat_session_repository),
) -> SessionResponse:
    await enforce_rate_limit(
        request, response, limiter=get_rate_limiter(), key=f"sessions:{caller.user_id}",
        limit=SESSION_CREATE_LIMIT,
    )
    session = await StartChatSession(sessions).execute(
        caller.tenant_id, caller.user_id, payload.title
    )
    return SessionResponse.of(session)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    caller: Caller = Depends(get_caller),
    sessions: ChatSessionRepository = Depends(get_chat_session_repository),
    limit: int = Query(default=20, ge=1, le=MAX_SESSIONS_PER_PAGE),
) -> SessionListResponse:
    found = await ListChatSessions(sessions).execute(caller.tenant_id, caller.user_id, limit)
    return SessionListResponse(sessions=[SessionResponse.of(s) for s in found])


@router.post("/{session_id}/answers", response_model=AnswerResponse)
async def answer(
    session_id: uuid.UUID,
    payload: AnswerRequest,
    request: Request,
    response: Response,
    # Declared before the pipeline, so a request without a valid token is refused
    # before anything builds the pipeline.
    caller: Caller = Depends(get_caller),
    answer_in_session: AnswerInSession = Depends(get_answer_in_session),
) -> AnswerResponse:
    await enforce_rate_limit(
        request, response, limiter=get_rate_limiter(), key=f"chat:{caller.user_id}",
        limit=chat_rate_limit(),
    )
    result = await answer_in_session.execute(
        caller.tenant_id, caller.user_id, session_id, payload.question
    )
    return answer_response(result)
```

In `src/api/routers/chat.py`, replace the `claims` dependency with the caller and the shared limit:

```python
@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    caller: Caller = Depends(get_caller),
) -> ChatResponse:
    await enforce_rate_limit(
        request, response, limiter=get_rate_limiter(), key=f"chat:{caller.user_id}",
        limit=chat_rate_limit(),
    )
    search = SearchDocuments(embedding_model=get_embedding_model(), vector_store=get_vector_store())
    use_case = AnswerQuestion(search_documents=search, chat_model=get_chat_model(), top_k=_TOP_K)
    result = await use_case.execute(tenant_id=caller.tenant_id, question=payload.question)
```

The rest of `chat()` is unchanged. Update its imports: add `Request`, `Response`, `Caller`, `get_caller`, `get_rate_limiter`, `chat_rate_limit`, and `enforce_rate_limit`; remove `Any` and `get_current_user_claims`.

In `src/api/exception_handlers.py`, add the handlers, and import `SessionNotFound` and `QueryExceedsBudget` from `src.orchestration.domain.errors`:

```python
async def session_not_found_handler(request: Request, exc: SessionNotFound) -> JSONResponse:
    # One status and body whether the session is missing, another user's, or another
    # tenant's, so a caller can't learn which session ids exist.
    return JSONResponse(status_code=404, content={"detail": "Session not found"})


async def query_exceeds_budget_handler(request: Request, exc: QueryExceedsBudget) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
```

Register both in `register_exception_handlers` with the same `# type: ignore[arg-type]` the others carry.

In `src/api/main.py`, include the router, and drain the cascade on shutdown:

```python
from src.api.routers.sessions import router as sessions_router

app.include_router(sessions_router)


@app.on_event("shutdown")
async def drain_unified_cascade() -> None:
    from src.api.dependencies import get_unified_pipeline

    # Only a pipeline some request built has anything to drain; don't build one now.
    if get_unified_pipeline.cache_info().currsize:
        await get_unified_pipeline().cascade.drain()
```

- [ ] **Step 4: Run the tests.** Run the Step 2 command. Then run `tests/integration/test_chat_endpoint.py`, `test_auth_endpoints.py`, and `test_documents_endpoints.py`, then the full unit suite, `mypy src`, and ruff on every changed file. Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/api/routers/sessions.py src/api/routers/chat.py src/api/exception_handlers.py src/api/main.py tests/integration/test_sessions_endpoints.py
git commit -m "feat: answer through the unified path in the caller's own sessions"
```

---

### Task 9: Live check, security review, documentation, and integration (#192)

**Files:**
- Modify: `docs/architecture/OVERVIEW.md`, `docs/security/SECURITY.md`, `docs/database/DATABASE.md`, `docs/testing/TESTING.md`, `docs/architecture/CONTEXT_GRAPH.md`, `CLAUDE.md`, `README.md`, and this plan (execution notes)

- [ ] **Step 1: Live check.** Start the API against local Postgres (migrated to 0007), Redis, and Qdrant, with `CHAT_PROVIDER=ollama` and `CHAT_MODEL=qwen3.5`. Then, with `httpx` from a scratch script:
  1. register and log in two users;
  2. upload a text document as the first;
  3. create a session and answer "What is the return policy?" in it;
  4. list sessions;
  5. repeat the answer as the second user.

  Expected statuses: 201, 200, 201, 201, 200, 200, 404. Record the statuses, the routing, and the attempts in the execution notes.

- [ ] **Step 2: Security review.** Invoke the `fullstack-e2e-security-engineer` skill on `develop..HEAD`, against `docs/security/SECURITY.md` and OWASP's API Security Top 10. Fix each finding test-first in its own commit, or record why it stays, in the execution notes.

- [ ] **Step 3: Documentation.**
  - **`OVERVIEW.md`:** describe the three endpoints, the MAG-and-RAG composition, and the ownership check, and remove the unified endpoint from what remains unbuilt. Ingestion, streaming, workers, the frontend, and Kubernetes stay listed.
  - **`SECURITY.md`:** describe object-level authorization on sessions, and the per-user chat limit, as enforced controls. Name the integration tests that verify each.
  - **`DATABASE.md`:** `PostgresChatSessionRepository` as the sessions table's reader and creator, and index 0007.
  - **`TESTING.md`, `CLAUDE.md`, and `README.md`:** the new unit and integration test counts and file counts, and the built scope.
  - **`CONTEXT_GRAPH.md`:** the new identity and API classes, counted with `ast` the way the sixth generation's note describes.

- [ ] **Step 4: Verify.** Run the full unit suite and the full integration suite (about 11 minutes), `mypy src`, and ruff on every changed file. Expected: all pass, and the 8 vLLM skips unchanged.

- [ ] **Step 5: Commit, merge, and close.**
  - Commit the documentation.
  - Merge to `develop` locally with a `merge:` commit, re-run the unit suite and `mypy src` on the merged result, and remove the worktree and branch.
  - Close the task issues and #183, each with a comment naming the merge. Leave Epic #182 open for its later stories, with a comment.

## Execution notes

Where executing this plan changed what it says, the change is kept and recorded here rather than bent back to match the text above.
