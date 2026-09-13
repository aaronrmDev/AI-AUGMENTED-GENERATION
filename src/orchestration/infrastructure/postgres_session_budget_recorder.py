import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src.identity.infrastructure.db import set_tenant_context
from src.orchestration.domain.entities import PARADIGM_ORDER, BudgetAllocation, Paradigm
from src.orchestration.domain.errors import SessionNotFound
from src.orchestration.domain.ports import SessionBudgetRecorder


def _utc_now() -> datetime:
    return datetime.now(UTC)


def budget_record(
    allocation: BudgetAllocation, contributing: frozenset[Paradigm], recorded_at: datetime
) -> dict[str, object]:
    return {
        "total": allocation.total,
        "slices": {
            "cag": allocation.cag,
            "mag": allocation.mag,
            "rag": allocation.rag,
            "query": allocation.query,
            "reserve": allocation.reserve,
        },
        "contributing": [p.value for p in PARADIGM_ORDER if p in contributing],
        "recorded_at": recorded_at.isoformat(),
    }


class PostgresSessionBudgetRecorder(SessionBudgetRecorder):
    """Writes the latest turn's allocation into sessions.context_budget.

    The column has existed since migration 0001 and DATABASE.md already
    describes it; this is its first writer. set_tenant_context is called
    here rather than trusted to the caller, so the tenant_isolation RLS
    policy always has a tenant to enforce -- and an UPDATE that RLS hides
    (another tenant's session) looks exactly like a missing session, which
    is why both raise SessionNotFound instead of passing silently.
    Flushes but does not commit, matching the MAG repositories.
    """

    def __init__(self, session: AsyncSession, clock: Callable[[], datetime] = _utc_now) -> None:
        self._session = session
        self._clock = clock

    async def record(
        self,
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        allocation: BudgetAllocation,
        contributing: frozenset[Paradigm],
    ) -> None:
        await set_tenant_context(self._session, tenant_id)
        result = await self._session.execute(
            text("UPDATE sessions SET context_budget = CAST(:budget AS jsonb) WHERE id = :id"),
            {
                "budget": json.dumps(budget_record(allocation, contributing, self._clock())),
                "id": session_id,
            },
        )
        if cast(CursorResult[Any], result).rowcount != 1:
            raise SessionNotFound(session_id)
        await self._session.flush()
