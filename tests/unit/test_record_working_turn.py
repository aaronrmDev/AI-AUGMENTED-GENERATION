import uuid
from datetime import UTC, datetime

from src.mag.application.commands.record_working_turn import RecordWorkingTurn
from tests.unit.mag_fakes import FakeWorkingMemoryStore


async def test_execute_pushes_a_turn_and_returns_it():
    store = FakeWorkingMemoryStore()
    command = RecordWorkingTurn(store)
    session_id = uuid.uuid4()

    before = datetime.now(UTC)
    turn = await command.execute(session_id=session_id, role="user", content="hello there")
    after = datetime.now(UTC)

    assert turn.role == "user"
    assert turn.content == "hello there"
    assert turn.metadata == {}
    assert before <= turn.recorded_at <= after

    pushed = await store.get_recent_turns(session_id, limit=10)
    assert pushed == [turn]


async def test_execute_passes_through_metadata():
    store = FakeWorkingMemoryStore()
    command = RecordWorkingTurn(store)
    session_id = uuid.uuid4()

    turn = await command.execute(
        session_id=session_id,
        role="assistant",
        content="hi",
        metadata={"tool_calls": ["search"]},
    )

    assert turn.metadata == {"tool_calls": ["search"]}


async def test_execute_scopes_turns_to_their_own_session():
    store = FakeWorkingMemoryStore()
    command = RecordWorkingTurn(store)
    session_a = uuid.uuid4()
    session_b = uuid.uuid4()

    await command.execute(session_id=session_a, role="user", content="in session a")
    await command.execute(session_id=session_b, role="user", content="in session b")

    turns_a = await store.get_recent_turns(session_a, limit=10)
    turns_b = await store.get_recent_turns(session_b, limit=10)
    assert [t.content for t in turns_a] == ["in session a"]
    assert [t.content for t in turns_b] == ["in session b"]
