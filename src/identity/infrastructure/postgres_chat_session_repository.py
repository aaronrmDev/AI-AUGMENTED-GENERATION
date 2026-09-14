import json
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.domain.entities import ChatSession
from src.identity.domain.ports import ChatSessionRepository
from src.identity.infrastructure.db import set_tenant_context

_COLUMNS = "id, tenant_id, user_id, title, context_budget, created_at"


def _chat_session(row: Any) -> ChatSession:
    budget = row.context_budget
    if isinstance(budget, str):  # asyncpg returns jsonb as text unless a codec is registered
        budget = json.loads(budget)
    return ChatSession(row.id, row.tenant_id, row.user_id, row.title, budget, row.created_at)


class PostgresChatSessionRepository(ChatSessionRepository):
    """Chat sessions in the `sessions` table, each call in its own short transaction.

    Owning the transaction follows PostgresSessionBudgetRecorder. AnswerInSession
    checks ownership right before a cascade and a model call that can take seconds, and
    a connection held across them would be one nobody else could use. set_tenant_context
    runs inside each transaction, so the tenant_isolation policy always has a tenant to
    enforce, and every query also matches user_id.
    """

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def create(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, title: str | None
    ) -> ChatSession:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            row = (
                await session.execute(
                    text(
                        "INSERT INTO sessions (user_id, tenant_id, title) "
                        f"VALUES (:user_id, :tenant_id, :title) RETURNING {_COLUMNS}"
                    ),
                    {"user_id": user_id, "tenant_id": tenant_id, "title": title},
                )
            ).one()
        return _chat_session(row)

    async def find_owned(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID
    ) -> ChatSession | None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            row = (
                await session.execute(
                    text(f"SELECT {_COLUMNS} FROM sessions WHERE id = :id AND user_id = :user_id"),
                    {"id": session_id, "user_id": user_id},
                )
            ).one_or_none()
        return None if row is None else _chat_session(row)

    async def list_owned(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int
    ) -> list[ChatSession]:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            rows = (
                await session.execute(
                    text(
                        f"SELECT {_COLUMNS} FROM sessions WHERE user_id = :user_id "
                        "ORDER BY created_at DESC, id DESC LIMIT :limit"
                    ),
                    {"user_id": user_id, "limit": limit},
                )
            ).all()
        return [_chat_session(row) for row in rows]
