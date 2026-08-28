import uuid

from src.mag.domain.entities import EpisodicMemory
from src.mag.domain.ports import EpisodicMemoryRepository


class RetrieveEpisodes:
    def __init__(self, episodic_memory_repository: EpisodicMemoryRepository) -> None:
        self._episodes = episodic_memory_repository

    async def execute(self, tenant_id: uuid.UUID, session_id: uuid.UUID) -> list[EpisodicMemory]:
        return await self._episodes.get_by_session(session_id, tenant_id)
