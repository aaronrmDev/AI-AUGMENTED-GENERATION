import json
import uuid

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from src.mag.domain.entities import ProceduralMemory
from src.mag.domain.ports import ProceduralMemoryRepository


class PostgresProceduralMemoryRepository(ProceduralMemoryRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, procedure: ProceduralMemory, tenant_id: uuid.UUID) -> None:
        # ON CONFLICT DO UPDATE, not a bare INSERT: a task_pattern is a slot a
        # later call overwrites (uq_procedural_memory_user_id_task_pattern
        # from migration 0004 makes (user_id, task_pattern) unique), not an
        # append-only log -- matches PostgresSemanticMemoryRepository.save's
        # corrected shape exactly.
        await self._session.execute(
            text(
                """
                INSERT INTO procedural_memory (
                    id, user_id, tenant_id, task_pattern, success_rate, last_used, workflow
                )
                VALUES (
                    :id, :user_id, :tenant_id, :task_pattern, :success_rate, :last_used,
                    CAST(:workflow AS jsonb)
                )
                ON CONFLICT (user_id, task_pattern) DO UPDATE SET
                    id = EXCLUDED.id,
                    success_rate = EXCLUDED.success_rate,
                    last_used = EXCLUDED.last_used,
                    workflow = EXCLUDED.workflow
                """
            ),
            {
                "id": procedure.id,
                "user_id": procedure.user_id,
                "tenant_id": tenant_id,
                "task_pattern": procedure.task_pattern,
                "success_rate": procedure.success_rate,
                "last_used": procedure.last_used,
                "workflow": json.dumps(procedure.workflow),
            },
        )
        await self._session.flush()

    async def find_by_task_pattern(
        self, user_id: uuid.UUID, task_pattern: str, tenant_id: uuid.UUID
    ) -> ProceduralMemory | None:
        result = await self._session.execute(
            text(
                """
                SELECT id, user_id, task_pattern, success_rate, last_used, workflow
                FROM procedural_memory
                WHERE user_id = :user_id AND task_pattern = :task_pattern
                    AND tenant_id = :tenant_id
                """
            ),
            {"user_id": user_id, "task_pattern": task_pattern, "tenant_id": tenant_id},
        )
        row = result.mappings().first()
        return self._row_to_procedure(row) if row else None

    @staticmethod
    def _row_to_procedure(row: RowMapping) -> ProceduralMemory:
        workflow = row["workflow"]
        if isinstance(workflow, str):
            workflow = json.loads(workflow)
        return ProceduralMemory(
            id=row["id"],
            user_id=row["user_id"],
            task_pattern=row["task_pattern"],
            workflow=workflow,
            success_rate=row["success_rate"],
            last_used=row["last_used"],
        )
