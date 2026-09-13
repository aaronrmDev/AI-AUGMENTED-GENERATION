import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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
    describes it; this is its first writer.

    Unlike the MAG repositories, which flush into their caller's session,
    this recorder opens and commits its own short transaction. That is a
    measured requirement, not a style choice: the budget is recorded after
    the cascade, and when the cascade's timeout cancels a query mid-flight,
    SQLAlchemy terminates that query's connection -- a recorder writing
    through a shared session would then fail the whole request. A turn's
    budget record is its own unit of work, so it gets its own connection.

    set_tenant_context runs inside that transaction, so the tenant_isolation
    RLS policy always has a tenant to enforce, and the UPDATE also matches
    on user_id, so one user cannot overwrite another user's record within a
    tenant. An UPDATE that matches no row -- a missing session, one RLS hides
    because it belongs to another tenant, or another user's -- raises
    SessionNotFound and rolls back instead of passing silently.
    """

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._clock = clock

    async def record(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        allocation: BudgetAllocation,
        contributing: frozenset[Paradigm],
    ) -> None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            result = await session.execute(
                text(
                    "UPDATE sessions SET context_budget = CAST(:budget AS jsonb) "
                    "WHERE id = :id AND user_id = :user_id"
                ),
                {
                    "budget": json.dumps(budget_record(allocation, contributing, self._clock())),
                    "id": session_id,
                    "user_id": user_id,
                },
            )
            if cast(CursorResult[Any], result).rowcount != 1:
                raise SessionNotFound(session_id)
