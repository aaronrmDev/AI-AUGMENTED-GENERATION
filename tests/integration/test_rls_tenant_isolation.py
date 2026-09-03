import uuid
from datetime import UTC, datetime

from sqlalchemy import text

from src.identity.domain.entities import PasswordHash, User
from src.identity.infrastructure.db import set_tenant_context
from src.identity.infrastructure.postgres_user_repository import PostgresUserRepository

VALID_HASH = PasswordHash("$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl")


async def test_rls_returns_zero_cross_tenant_sessions_even_without_an_app_level_filter(db_session):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    now = datetime.now(UTC)

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
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {
            "id": uuid.uuid4(),
            "user_id": user_a.id,
            "tenant_id": tenant_a,
            "title": "tenant a's session",
        },
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_b)
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {
            "id": uuid.uuid4(),
            "user_id": user_b.id,
            "tenant_id": tenant_b,
            "title": "tenant b's session",
        },
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    # Deliberately no WHERE tenant_id = ... — RLS alone must do the filtering.
    result = await db_session.execute(text("SELECT title FROM sessions"))
    titles = {row.title for row in result}

    assert titles == {"tenant a's session"}
    assert "tenant b's session" not in titles
