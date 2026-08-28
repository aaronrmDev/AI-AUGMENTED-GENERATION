import uuid

from src.mag.domain.entities import WorkingMemoryTurn
from src.mag.domain.ports import WorkingMemoryStore


class RetrieveWorkingMemory:
    def __init__(self, working_memory_store: WorkingMemoryStore) -> None:
        self._working_memory_store = working_memory_store

    async def execute(self, session_id: uuid.UUID, limit: int = 20) -> list[WorkingMemoryTurn]:
        return await self._working_memory_store.get_recent_turns(session_id, limit)
