import uuid

from src.mag.domain.entities import ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryIndex


class SemanticSimilarityRetrieval:
    # Formalizes #67 as a named strategy alongside the other five: the
    # underlying search was already real and correctly-ordered as of Batch
    # A/B (see EpisodicMemoryIndex.search's docstring), just unscored until
    # this batch's score-carrying change. Wraps the Qdrant index, not the
    # Postgres repository -- Qdrant is this system's embedding-bearing real-
    # ANN read path (see EpisodicMemoryRepository.search_by_similarity's
    # docstring for why the Postgres side is deliberately not it).
    def __init__(self, episodic_memory_index: EpisodicMemoryIndex) -> None:
        self._index = episodic_memory_index

    async def execute(
        self,
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
    ) -> list[ScoredEpisode]:
        return await self._index.search(query_embedding, tenant_id, session_id, top_k)
