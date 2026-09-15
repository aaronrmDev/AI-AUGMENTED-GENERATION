import uuid

from src.identity.domain.entities import MAX_SESSION_TITLE_CHARS, ChatSession
from src.identity.domain.ports import ChatSessionRepository


class StartChatSession:
    def __init__(self, repository: ChatSessionRepository) -> None:
        self._repository = repository

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, title: str | None
    ) -> ChatSession:
        stripped = (title or "").strip()
        # Checked before the insert, so a refused title never leaves a row behind.
        if len(stripped) > MAX_SESSION_TITLE_CHARS:
            raise ValueError(f"a session title is at most {MAX_SESSION_TITLE_CHARS} characters")
        return await self._repository.create(tenant_id, user_id, stripped or None)
