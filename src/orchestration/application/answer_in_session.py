import uuid
from typing import Protocol

from src.identity.domain.ports import ChatSessionRepository
from src.orchestration.application.unified_answer_question import UnifiedAnswer
from src.orchestration.domain.errors import SessionNotFound


class SessionQuestionAnswerer(Protocol):
    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> UnifiedAnswer: ...


class AnswerInSession:
    """Answers a question in one of the caller's own chat sessions.

    The session is looked up as the caller's before anything else happens. A session
    that doesn't exist, belongs to another user, or lives in another tenant raises the
    same SessionNotFound, so nothing is embedded, retrieved, or recorded for it, and
    the caller can't tell which of the three it was. PostgresSessionBudgetRecorder's
    own ownership check stays as a second line.
    """

    def __init__(
        self, sessions: ChatSessionRepository, answerer: SessionQuestionAnswerer
    ) -> None:
        self._sessions = sessions
        self._answerer = answerer

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> UnifiedAnswer:
        if await self._sessions.find_owned(tenant_id, user_id, session_id) is None:
            raise SessionNotFound(session_id)
        return await self._answerer.execute(tenant_id, user_id, session_id, question)
