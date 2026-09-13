from datetime import UTC, datetime

from src.orchestration.domain.entities import BudgetAllocation, Paradigm
from src.orchestration.infrastructure.postgres_session_budget_recorder import budget_record


def test_the_record_carries_every_slice_the_contributing_set_and_a_timestamp():
    allocation = BudgetAllocation(cag=5, mag=4, rag=3, query=2, reserve=1)
    record = budget_record(
        allocation,
        frozenset({Paradigm.RAG, Paradigm.CAG}),
        datetime(2026, 9, 13, 12, 0, tzinfo=UTC),
    )
    assert record == {
        "total": 15,
        "slices": {"cag": 5, "mag": 4, "rag": 3, "query": 2, "reserve": 1},
        "contributing": ["cag", "rag"],
        "recorded_at": "2026-09-13T12:00:00+00:00",
    }
