import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from src.identity.domain.entities import AccessToken, PasswordHash, RefreshToken, TokenPair, User
from src.identity.domain.errors import EmailAlreadyRegistered
from src.identity.domain.ports import PasswordHasher, RefreshTokenStore, TokenIssuer, UserRepository


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

    def __init__(self) -> None:
        # Counts calls to verify(). AuthenticateUser is required to do the same
        # amount of hashing work whether or not the email exists, so that an
        # unregistered address can't be distinguished from a registered one by
        # response time alone. Asserting on this counter is how a unit test can
        # check "equal work was done" without measuring a wall clock, which
        # would be flaky.
        self.verify_call_count = 0

    def hash(self, plain_password: str) -> PasswordHash:
        return PasswordHash(f"$argon2id$fake${plain_password}")

    def verify(self, plain_password: str, hashed: PasswordHash) -> bool:
        self.verify_call_count += 1
        return str(hashed) == f"$argon2id$fake${plain_password}"


class FakeTokenIssuer(TokenIssuer):
    def issue_pair(self, user_id: uuid.UUID, tenant_id: uuid.UUID) -> TokenPair:
        now = datetime.now(UTC)
        # tenant_id is encoded into the access token's value, not ignored: a
        # fake that drops the argument would let a use case pass the wrong
        # tenant through (or none at all) with every test still green, which is
        # exactly the bug a tenant-isolation system can least afford to ship.
        return TokenPair(
            access_token=AccessToken(
                value=f"access-{user_id}-{tenant_id}", expires_at=now + timedelta(minutes=15)
            ),
            refresh_token=RefreshToken(
                token_id=uuid.uuid4(),
                value=f"refresh-{user_id}",
                expires_at=now + timedelta(days=7),
            ),
        )

    def verify_access_token(self, token: str) -> dict[str, Any]:
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
