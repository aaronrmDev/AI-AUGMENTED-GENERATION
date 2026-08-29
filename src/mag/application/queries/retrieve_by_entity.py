import uuid

from src.mag.domain.entities import ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryRepository


class EntityRetrieval:
    def __init__(self, episodic_memory_repository: EpisodicMemoryRepository) -> None:
        self._episodes = episodic_memory_repository

    async def execute(
        self, tenant_id: uuid.UUID, session_id: uuid.UUID, entity: str, top_k: int
    ) -> list[ScoredEpisode]:
        episodes = await self._episodes.get_by_session_matching_entity(
            session_id, tenant_id, entity, top_k
        )
        # Binary relevance -- see get_by_session_matching_entity's docstring:
        # a structured content["entities"] hit and a substring-only fallback
        # hit both come back from the repository with no way to tell which
        # kind matched, so every result scores the same 1.0 rather than
        # inventing a graded confidence this system has no signal to back.
        return [ScoredEpisode(episode=e, score=1.0) for e in episodes]
