"""Every cleanup path of SessionScopedSemanticFactSearch, deterministically, with a
fake session: the real-Postgres cancellation test can't tell a correct cleanup
from one SQLAlchemy's own disconnect handling happened to cover."""
import asyncio
import uuid

import pytest

from src.mag.domain.entities import ScoredFact, SemanticMemory
from src.orchestration.infrastructure.session_scoped_semantic_fact_search import (
    SessionScopedSemanticFactSearch,
)

_TENANT = uuid.uuid4()
_USER = uuid.uuid4()
_FACTS = [
    ScoredFact(
        SemanticMemory(uuid.uuid4(), _USER, "k", "v", [1.0, 0.0]),
        0.9,
    )
]


class _FakeSession:
    def __init__(self, close_error: Exception | None = None) -> None:
        self.executed: list[object] = []
        self.closed = False
        self.invalidated = False
        self._close_error = close_error

    async def execute(self, statement, params=None):
        self.executed.append(params)

    async def close(self) -> None:
        if self._close_error is not None:
            raise self._close_error
        self.closed = True

    async def invalidate(self) -> None:
        self.invalidated = True


class _FakeRepository:
    def __init__(self, error: BaseException | None = None, delay_seconds: float = 0.0) -> None:
        self._error = error
        self._delay = delay_seconds
        self.calls: list[tuple[uuid.UUID, uuid.UUID, int]] = []

    async def search_by_similarity(self, query_embedding, user_id, tenant_id, top_k):
        self.calls.append((user_id, tenant_id, top_k))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return list(_FACTS)


def _search(session: _FakeSession, repository: _FakeRepository) -> SessionScopedSemanticFactSearch:
    return SessionScopedSemanticFactSearch(
        lambda: session, repository_factory=lambda _session: repository  # type: ignore[arg-type,return-value]
    )


async def test_a_successful_search_sets_the_tenant_closes_its_session_and_returns_the_facts():
    session, repository = _FakeSession(), _FakeRepository()

    facts = await _search(session, repository).search(_TENANT, _USER, [1.0, 0.0], 3)

    assert facts == _FACTS
    assert session.executed == [{"tenant_id": str(_TENANT)}]
    assert repository.calls == [(_USER, _TENANT, 3)]
    assert (session.closed, session.invalidated) == (True, False)


async def test_a_failing_query_invalidates_the_session_and_propagates():
    session = _FakeSession()

    with pytest.raises(RuntimeError):
        await _search(session, _FakeRepository(error=RuntimeError("db down"))).search(
            _TENANT, _USER, [1.0, 0.0], 3
        )

    assert (session.closed, session.invalidated) == (False, True)


async def test_a_cancelled_search_invalidates_the_session_and_stays_cancelled():
    session = _FakeSession()
    task = asyncio.create_task(
        _search(session, _FakeRepository(delay_seconds=5.0)).search(_TENANT, _USER, [1.0, 0.0], 3)
    )
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert session.invalidated is True


async def test_a_close_that_fails_after_a_successful_search_keeps_the_results():
    # The facts are already in hand; losing them to a cleanup error would turn
    # a good MAG answer into an ERROR outcome.
    session = _FakeSession(close_error=RuntimeError("connection reset"))

    facts = await _search(session, _FakeRepository()).search(_TENANT, _USER, [1.0, 0.0], 3)

    assert facts == _FACTS
    assert session.invalidated is True
