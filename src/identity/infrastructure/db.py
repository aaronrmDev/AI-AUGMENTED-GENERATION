import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def get_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def get_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def set_tenant_context(session: AsyncSession, tenant_id: uuid.UUID) -> None:
    # Postgres's SET/SET LOCAL grammar only accepts a literal, never a bind
    # parameter, so "SET LOCAL app.current_tenant_id = :tenant_id" fails
    # under asyncpg specifically: asyncpg always binds parameters server-side
    # via the extended query protocol (unlike psycopg2, which mogrifies
    # placeholders into the SQL text client-side before sending), so Postgres
    # sees a literal "$1" where it requires a constant and raises a syntax
    # error. set_config() is a regular function, so it accepts a normal bound
    # parameter; its third argument (true) scopes the setting to SET LOCAL's
    # transaction-local semantics, i.e. it's reset at the next commit/rollback
    # rather than persisting for the life of the connection.
    await session.execute(
        text("SELECT set_config('app.current_tenant_id', :tenant_id, true)"),
        {"tenant_id": str(tenant_id)},
    )
