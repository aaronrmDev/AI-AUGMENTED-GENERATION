import uuid
from datetime import UTC, datetime

from src.identity.domain.entities import User
from src.identity.domain.ports import PasswordHasher, UserRepository


class RegisterUser:
    def __init__(self, user_repository: UserRepository, password_hasher: PasswordHasher) -> None:
        self._users = user_repository
        self._hasher = password_hasher

    async def execute(self, email: str, plain_password: str) -> User:
        now = datetime.now(UTC)
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
