import uuid
from datetime import UTC, datetime

from src.mag.application.queries.retrieve_working_memory import RetrieveWorkingMemory
from src.mag.domain.entities import WorkingMemoryTurn
from tests.unit.mag_fakes import FakeWorkingMemoryStore


def _turn(content: str) -> WorkingMemoryTurn:
    return WorkingMemoryTurn(role="user", content=content, recorded_at=datetime.now(UTC))


async def test_execute_delegates_to_the_store_get_recent_turns():
    store = FakeWorkingMemoryStore()
    session_id = uuid.uuid4()
    await store.push_turn(session_id, _turn("hello"))

    query = RetrieveWorkingMemory(store)
    turns = await query.execute(session_id)

    assert [t.content for t in turns] == ["hello"]


async def test_execute_defaults_limit_to_20():
    store = FakeWorkingMemoryStore()
    session_id = uuid.uuid4()
    for i in range(25):
        await store.push_turn(session_id, _turn(f"turn {i}"))

    query = RetrieveWorkingMemory(store)
    turns = await query.execute(session_id)

    assert len(turns) == 20


async def test_execute_limit_bounds_the_returned_count_to_the_most_recent_n():
    store = FakeWorkingMemoryStore()
    session_id = uuid.uuid4()
    for i in range(10):
        await store.push_turn(session_id, _turn(f"turn {i}"))

    query = RetrieveWorkingMemory(store)
    turns = await query.execute(session_id, limit=3)

    assert [t.content for t in turns] == ["turn 7", "turn 8", "turn 9"]


async def test_execute_returns_empty_list_for_a_session_with_no_turns():
    store = FakeWorkingMemoryStore()
    query = RetrieveWorkingMemory(store)

    turns = await query.execute(uuid.uuid4())

    assert turns == []
