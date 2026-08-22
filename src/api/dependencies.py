import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from src.identity.domain.errors import TokenExpired
from src.identity.infrastructure.db import get_engine, get_sessionmaker, set_tenant_context
from src.identity.infrastructure.jwt_token_issuer import JWTTokenIssuer
from src.identity.infrastructure.postgres_user_repository import PostgresUserRepository
from src.identity.infrastructure.redis_rate_limiter import RedisRateLimiter
from src.identity.infrastructure.redis_refresh_token_store import RedisRefreshTokenStore

_engine = get_engine(os.environ["APP_DATABASE_URL"])
_sessionmaker = get_sessionmaker(_engine)


def get_token_issuer() -> JWTTokenIssuer:
    return JWTTokenIssuer(secret_key=os.environ["JWT_SECRET_KEY"])


def get_refresh_token_store() -> RedisRefreshTokenStore:
    return RedisRefreshTokenStore(os.environ["REDIS_URL"])


def get_rate_limiter() -> RedisRateLimiter:
    return RedisRateLimiter(os.environ["REDIS_URL"])


async def get_raw_db_session() -> AsyncGenerator[AsyncSession, None]:
    """A session with no tenant context set — only for pre-auth flows like register/login."""
    async with _sessionmaker() as session:
        yield session


async def get_current_user_claims(
    authorization: str | None = Header(default=None),
    token_issuer: JWTTokenIssuer = Depends(get_token_issuer),
) -> dict[str, Any]:
    # Header(default=None), not Header(...) (a required field): the required
    # form makes FastAPI raise its own RequestValidationError and return a
    # generic 422 the instant the header is absent, before this function body
    # ever runs — bypassing every handler in exception_handlers.py entirely.
    # Accepting None and checking it explicitly keeps the missing-header case
    # on the same path as the malformed-header case below, so both surface
    # as the domain's TokenExpired through the registered handler.
    if authorization is None or not authorization.startswith("Bearer "):
        raise TokenExpired()
    token = authorization.removeprefix("Bearer ")
    return token_issuer.verify_access_token(token)


async def get_db_session(
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> AsyncGenerator[AsyncSession, None]:
    """A tenant-scoped session for any endpoint behind auth.

    The tenant context is set before the caller ever sees the session, so no
    endpoint can forget to do it. Nothing consumes this yet — this sub-project
    ships no protected routes — and the first one that does is where it gets
    exercised end-to-end against real RLS.
    """
    async with _sessionmaker() as session:
        await set_tenant_context(session, uuid.UUID(claims["tenant_id"]))
        yield session


def get_user_repository_unscoped(
    session: AsyncSession = Depends(get_raw_db_session),
) -> PostgresUserRepository:
    return PostgresUserRepository(session)


def get_user_repository_scoped(
    session: AsyncSession = Depends(get_db_session),
) -> PostgresUserRepository:
    return PostgresUserRepository(session)
