import json
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import Paradigm
from src.orchestration.domain.errors import SessionNotFound
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
_CONTRIBUTING = frozenset({Paradigm.CAG, Paradigm.RAG})


async def _create_user_and_session(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    await set_tenant_context(db_session, tenant_id)
    now = datetime.now(UTC)
    user_id = uuid.uuid4()
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
    session_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "t"},
    )
    await db_session.commit()
    return session_id


async def _stored_budget(db_session, tenant_id: uuid.UUID, session_id: uuid.UUID):
    await set_tenant_context(db_session, tenant_id)
    value = (
        await db_session.execute(
            text("SELECT context_budget FROM sessions WHERE id = :id"), {"id": session_id}
        )
    ).scalar_one()
    return json.loads(value) if isinstance(value, str) else value


def _recorder(db_session, **kwargs) -> PostgresSessionBudgetRecorder:
    # Same engine as db_session, but the recorder's own sessions and
    # connections -- the production shape, where it owns its unit of work.
    return PostgresSessionBudgetRecorder(get_sessionmaker(db_session.bind), **kwargs)


async def test_record_commits_the_allocation_into_the_real_sessions_row(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _create_user_and_session(db_session, tenant_id)
    recorder = _recorder(db_session, clock=lambda: datetime(2026, 9, 13, 12, 0, tzinfo=UTC))

    await recorder.record(tenant_id, session_id, allocate(128_000, _CONTRIBUTING), _CONTRIBUTING)

    # Read through db_session's own connection: only a committed write is visible here.
    assert await _stored_budget(db_session, tenant_id, session_id) == {
        "total": 128_000,
        "slices": {"cag": 72_533, "mag": 0, "rag": 36_266, "query": 12_800, "reserve": 6_401},
        "contributing": ["cag", "rag"],
        "recorded_at": "2026-09-13T12:00:00+00:00",
    }


async def test_recording_under_another_tenant_is_refused_by_rls_and_leaves_the_row_untouched(
    db_session,
):
    owner = uuid.uuid4()
    session_id = await _create_user_and_session(db_session, owner)

    with pytest.raises(SessionNotFound):
        await _recorder(db_session).record(
            uuid.uuid4(), session_id, allocate(128_000, _CONTRIBUTING), _CONTRIBUTING
        )

    assert await _stored_budget(db_session, owner, session_id) is None


async def test_recording_for_a_session_that_does_not_exist_raises(db_session):
    with pytest.raises(SessionNotFound):
        await _recorder(db_session).record(
            uuid.uuid4(), uuid.uuid4(), allocate(128_000, _CONTRIBUTING), _CONTRIBUTING
        )
