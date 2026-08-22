import uuid

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.identity.domain.entities import PasswordHash, User
from src.identity.domain.errors import EmailAlreadyRegistered
from src.identity.domain.ports import UserRepository


class PostgresUserRepository(UserRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, user: User) -> None:
        try:
            await self._session.execute(
                text(
                    """
                    INSERT INTO users
                        (id, email, hashed_password, tenant_id, created_at, updated_at)
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
    def _row_to_user(row: RowMapping) -> User:
        # RowMapping, not Row or a plain Mapping: both call sites pass
        # result.mappings().first(), and SQLAlchemy types RowMapping as
        # Mapping[str, Any] only at runtime — its declared key type doesn't
        # unify with Mapping[str, Any] under strict checking, so naming the
        # concrete type is what actually matches. Values stay untyped by
        # necessity (the driver returns whatever each column decoded to); this
        # function is the boundary that pins them down by building a real User.
        return User(
            id=row["id"],
            email=row["email"],
            hashed_password=PasswordHash(row["hashed_password"]),
            tenant_id=row["tenant_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
