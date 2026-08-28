import uuid

import pytest_asyncio

from src.mag.domain.entities import WorkingMemoryTurn
from src.mag.infrastructure.redis_working_memory_store import RedisWorkingMemoryStore


@pytest_asyncio.fixture
async def store(redis_url):
    s = RedisWorkingMemoryStore(redis_url)
    yield s
    await s._client.aclose()


def _turn(content: str) -> WorkingMemoryTurn:
    from datetime import UTC, datetime

    return WorkingMemoryTurn(role="user", content=content, recorded_at=datetime.now(UTC))


async def test_push_then_get_recent_turns_returns_them_oldest_to_newest(store):
    session_id = uuid.uuid4()

    await store.push_turn(session_id, _turn("first"))
    await store.push_turn(session_id, _turn("second"))
    await store.push_turn(session_id, _turn("third"))

    turns = await store.get_recent_turns(session_id, limit=10)

    assert [t.content for t in turns] == ["first", "second", "third"]
    assert all(t.role == "user" for t in turns)


async def test_limit_truncates_against_a_real_redis_list_to_the_most_recent_n(store):
    session_id = uuid.uuid4()
    for i in range(5):
        await store.push_turn(session_id, _turn(f"turn {i}"))

    turns = await store.get_recent_turns(session_id, limit=2)

    assert [t.content for t in turns] == ["turn 3", "turn 4"]


async def test_push_turn_sets_a_ttl_on_the_key(redis_url):
    import redis.asyncio as redis

    store = RedisWorkingMemoryStore(redis_url)
    session_id = uuid.uuid4()
    await store.push_turn(session_id, _turn("hello"))

    client = redis.from_url(redis_url)
    ttl = await client.ttl(f"session:{session_id}:working_memory")
    await client.aclose()
    await store._client.aclose()

    assert 0 < ttl <= 86400


async def test_two_sessions_do_not_leak_into_each_others_working_memory(store):
    session_a = uuid.uuid4()
    session_b = uuid.uuid4()

    await store.push_turn(session_a, _turn("in session a"))
    await store.push_turn(session_b, _turn("in session b"))

    turns_a = await store.get_recent_turns(session_a, limit=10)
    turns_b = await store.get_recent_turns(session_b, limit=10)

    assert [t.content for t in turns_a] == ["in session a"]
    assert [t.content for t in turns_b] == ["in session b"]


async def test_get_recent_turns_returns_empty_list_for_an_unknown_session(store):
    turns = await store.get_recent_turns(uuid.uuid4(), limit=10)
    assert turns == []


async def test_recorded_at_round_trips_through_json_as_a_real_datetime(store):
    session_id = uuid.uuid4()
    turn = _turn("timestamped")

    await store.push_turn(session_id, turn)
    (retrieved,) = await store.get_recent_turns(session_id, limit=10)

    assert retrieved.recorded_at == turn.recorded_at


async def test_metadata_round_trips(store):
    from datetime import UTC, datetime

    session_id = uuid.uuid4()
    turn = WorkingMemoryTurn(
        role="assistant",
        content="hi",
        recorded_at=datetime.now(UTC),
        metadata={"tool_calls": ["search"]},
    )

    await store.push_turn(session_id, turn)
    (retrieved,) = await store.get_recent_turns(session_id, limit=10)

    assert retrieved.metadata == {"tool_calls": ["search"]}
