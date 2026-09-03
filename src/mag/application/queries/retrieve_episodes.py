import uuid

from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryRepository


class RetrieveEpisodes:
    def __init__(self, episodic_memory_repository: EpisodicMemoryRepository) -> None:
        self._episodes = episodic_memory_repository

    async def by_session(
        self, tenant_id: uuid.UUID, session_id: uuid.UUID
    ) -> list[EpisodicMemory]:
        return await self._episodes.get_by_session(session_id, tenant_id)

    async def by_similarity(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredEpisode]:
        return await self._episodes.search_by_similarity(query_embedding, tenant_id, top_k)
