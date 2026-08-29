import uuid

from src.mag.domain.entities import ScoredFact, SemanticMemory
from src.mag.domain.ports import SemanticMemoryRepository


class FindSemanticFacts:
    def __init__(self, semantic_memory_repository: SemanticMemoryRepository) -> None:
        self._repository = semantic_memory_repository

    async def by_key(
        self, user_id: uuid.UUID, fact_key: str, tenant_id: uuid.UUID
    ) -> SemanticMemory | None:
        return await self._repository.find_by_key(user_id, fact_key, tenant_id)

    async def by_similarity(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredFact]:
        return await self._repository.search_by_similarity(
            query_embedding, user_id, tenant_id, top_k
        )
