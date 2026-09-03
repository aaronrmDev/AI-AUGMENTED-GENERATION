import uuid
from datetime import UTC, datetime

import pytest

from src.identity.domain.entities import PasswordHash, User
from src.identity.domain.errors import EmailAlreadyRegistered
from src.identity.infrastructure.postgres_user_repository import PostgresUserRepository

VALID_HASH = PasswordHash("$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl")


def _new_user(email: str) -> User:
    now = datetime.now(UTC)
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
