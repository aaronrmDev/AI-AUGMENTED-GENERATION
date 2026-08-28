import uuid

from src.mag.domain.entities import ProceduralMemory
from src.mag.domain.ports import ProceduralMemoryRepository


class FindProcedure:
    def __init__(self, procedural_memory_repository: ProceduralMemoryRepository) -> None:
        self._repository = procedural_memory_repository

    async def by_task_pattern(
        self, user_id: uuid.UUID, task_pattern: str, tenant_id: uuid.UUID
    ) -> ProceduralMemory | None:
        return await self._repository.find_by_task_pattern(user_id, task_pattern, tenant_id)
