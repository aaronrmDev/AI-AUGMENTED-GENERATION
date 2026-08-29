import uuid

from src.mag.domain.entities import ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryRepository


class SalienceRetrieval:
    def __init__(self, episodic_memory_repository: EpisodicMemoryRepository) -> None:
        self._episodes = episodic_memory_repository

    async def execute(
        self, tenant_id: uuid.UUID, session_id: uuid.UUID, top_k: int
    ) -> list[ScoredEpisode]:
        episodes = await self._episodes.get_by_session_ranked_by_salience(
            session_id, tenant_id, top_k
        )
        # salience_score IS the relevance signal here -- nothing graded to
        # compute on top of it, unlike temporal retrieval's rank-decay
        # fallback (see TemporalRetrieval.execute).
        return [ScoredEpisode(episode=e, score=e.salience_score) for e in episodes]
