import uuid
from datetime import UTC, datetime, timedelta

from src.identity.domain.entities import ChatSession
from src.identity.domain.ports import ChatSessionRepository

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


class FakeChatSessionRepository(ChatSessionRepository):
    """In-memory sessions with the port's contract: every read is scoped to one tenant
    and one owning user, and each new session is one second newer than the last."""

    def __init__(self) -> None:
        self.sessions: list[ChatSession] = []

    async def create(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, title: str | None
    ) -> ChatSession:
        session = ChatSession(
            uuid.uuid4(), tenant_id, user_id, title, None,
            _T0 + timedelta(seconds=len(self.sessions)),
        )
        self.sessions.append(session)
        return session

    async def find_owned(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID
    ) -> ChatSession | None:
        return next(
            (
                s for s in self.sessions
                if (s.id, s.tenant_id, s.user_id) == (session_id, tenant_id, user_id)
            ),
            None,
        )

    async def list_owned(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int
    ) -> list[ChatSession]:
        owned = [s for s in self.sessions if (s.tenant_id, s.user_id) == (tenant_id, user_id)]
        return sorted(owned, key=lambda s: s.created_at, reverse=True)[:limit]
