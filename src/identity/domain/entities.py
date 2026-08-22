from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
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

    def __repr__(self) -> str:
        # A hand-written redacted repr rather than field(repr=False): this class
        # already replaces the generated __init__ with a validating one via
        # object.__setattr__, and `value` is its only field — a generated repr
        # with that field suppressed would read "PasswordHash()", which says
        # less than this does. The reason to redact at all is that a default
        # dataclass repr prints every field, so an Argon2id hash reaches any
        # log line, f-string, or unhandled-exception traceback that happens to
        # include a User or a PasswordHash — a silent violation of
        # docs/security/SECRETS_MANAGEMENT.md's "never logged" rule that
        # nothing in the code has to do wrong on purpose to trigger.
        # __str__ deliberately still returns the real value: the infrastructure
        # layer uses str(hash) to persist and verify it.
        return "PasswordHash(***)"


@dataclass(frozen=True)
class User:
    id: uuid.UUID
    email: str
    # repr=False for the same reason PasswordHash redacts its own: User is the
    # object most likely to end up in a log line or a traceback frame, and
    # without this its default repr would print the stored Argon2id hash in
    # full. Excluded from repr only — the field is still compared by __eq__ and
    # still part of the constructor signature.
    hashed_password: PasswordHash = field(repr=False)
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
