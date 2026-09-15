import uuid

from src.identity.domain.entities import ChatSession
from src.identity.domain.ports import ChatSessionRepository

MAX_SESSIONS_PER_PAGE = 100


class ListChatSessions:
    def __init__(self, repository: ChatSessionRepository) -> None:
        self._repository = repository

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, limit: int
    ) -> list[ChatSession]:
        if not 1 <= limit <= MAX_SESSIONS_PER_PAGE:
            raise ValueError(f"limit must be between 1 and {MAX_SESSIONS_PER_PAGE}")
        return await self._repository.list_owned(tenant_id, user_id, limit)
