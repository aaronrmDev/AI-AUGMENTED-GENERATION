from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.identity.domain.entities import PasswordHash, RefreshToken, TokenPair, User


class PasswordHasher(ABC):
    @abstractmethod
    def hash(self, plain_password: str) -> PasswordHash: ...

    @abstractmethod
    def verify(self, plain_password: str, hashed: PasswordHash) -> bool: ...


class TokenIssuer(ABC):
    @abstractmethod
    def issue_pair(self, user_id: uuid.UUID, tenant_id: uuid.UUID) -> TokenPair: ...

    @abstractmethod
    def verify_access_token(self, token: str) -> dict[str, Any]:
        """Returns the decoded claims, or raises TokenExpired / TokenAlreadyUsed-equivalent.

        dict[str, Any] rather than a TypedDict: these are JWT claims decoded
        from an attacker-supplied token, so the value types are whatever
        survived JSON decoding (str for `sub`/`tenant_id`, int for `exp`/`iat`)
        and the key set depends on which issuer minted the token. Any is the
        honest annotation for that; narrowing happens at each use site, which
        has to validate the value anyway.
        """
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
