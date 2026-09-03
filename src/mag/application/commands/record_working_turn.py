import uuid
from datetime import UTC, datetime
from typing import Any

from src.mag.domain.entities import WorkingMemoryTurn
from src.mag.domain.ports import WorkingMemoryStore


class RecordWorkingTurn:
    def __init__(self, working_memory_store: WorkingMemoryStore) -> None:
        self._working_memory_store = working_memory_store

    async def execute(
        self,
        session_id: uuid.UUID,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemoryTurn:
        turn = WorkingMemoryTurn(
            role=role,
            content=content,
            recorded_at=datetime.now(UTC),
            metadata=metadata or {},
        )
        await self._working_memory_store.push_turn(session_id, turn)
        return turn
