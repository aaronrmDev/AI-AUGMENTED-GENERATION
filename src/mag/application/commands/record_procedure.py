import uuid
from datetime import datetime
from typing import Any

from src.mag.domain.entities import ProceduralMemory
from src.mag.domain.ports import ProceduralMemoryRepository


class RecordProcedure:
    def __init__(self, procedural_memory_repository: ProceduralMemoryRepository) -> None:
        self._repository = procedural_memory_repository

    async def execute(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        task_pattern: str,
        workflow: dict[str, Any],
        success_rate: float = 0.0,
        last_used: datetime | None = None,
    ) -> ProceduralMemory:
        procedure = ProceduralMemory(
            # Deterministic (uuid5), not uuid4 -- matches RecordSemanticFact's
            # corrected shape. Postgres upserts by (user_id, task_pattern)
            # (migration 0004's unique constraint); a fresh uuid4 per call
            # would still upsert the row itself (the constraint upserts by
            # its own key columns, not id) but a deterministic id keeps the
            # id stable across re-records the same way EXCLUDED.id already
            # relies on for semantic_memory -- see record_semantic_fact.py.
            id=uuid.uuid5(uuid.NAMESPACE_OID, f"procedural_memory:{user_id}:{task_pattern}"),
            user_id=user_id,
            task_pattern=task_pattern,
            workflow=workflow,
            success_rate=success_rate,
            last_used=last_used,
        )
        await self._repository.save(procedure, tenant_id)
        return procedure
