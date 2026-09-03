import uuid
from datetime import UTC, datetime

import pytest

from src.identity.domain.entities import PasswordHash, User
from src.identity.domain.errors import InvalidCredentials


def test_user_equality_is_by_all_fields():
    now = datetime.now(UTC)
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


def test_password_hash_does_not_expose_its_value_in_a_repr():
    secret = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
    ph = PasswordHash(secret)

    assert secret not in repr(ph)
    # str() must keep returning the real value — the infrastructure layer uses
    # it to persist and verify the hash. Only repr is redacted.
    assert str(ph) == secret


def test_user_repr_does_not_leak_the_stored_password_hash():
    """A User in a log line or traceback must not print the hash it carries.

    Python's default dataclass repr prints every field, and a repr is produced
    implicitly in far more places than a deliberate log call — an f-string, a
    `%r`, an unhandled exception's frame locals. Nothing has to be written
    wrong for the hash to escape that way, so the field is excluded at the
    dataclass level rather than trusting every future call site.
    """
    now = datetime.now(UTC)
    secret = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
    user = User(
        id=uuid.uuid4(),
        email="a@example.com",
        hashed_password=PasswordHash(secret),
        tenant_id=uuid.uuid4(),
        created_at=now,
        updated_at=now,
    )

    assert secret not in repr(user)
    assert secret not in f"{user}"
    # The non-secret fields are still there — this redacts one field, it does
    # not make the object undebuggable.
    assert "a@example.com" in repr(user)


def test_invalid_credentials_error_carries_no_identifying_detail():
    err = InvalidCredentials()
    assert str(err) == "Invalid credentials"
