import uuid
from datetime import UTC, datetime

import pytest

from src.identity.application.list_chat_sessions import ListChatSessions
from src.identity.application.start_chat_session import StartChatSession
from src.identity.domain.entities import ChatSession
from tests.unit.session_fakes import FakeChatSessionRepository

TENANT, USER = uuid.uuid4(), uuid.uuid4()


def _session(title: str | None) -> ChatSession:
    return ChatSession(uuid.uuid4(), TENANT, USER, title, None, datetime(2026, 1, 1, tzinfo=UTC))


def test_a_title_may_be_200_characters_but_not_201():
    assert _session("x" * 200).title == "x" * 200
    with pytest.raises(ValueError):
        _session("x" * 201)


async def test_starting_a_session_strips_its_title_and_records_the_caller_as_owner():
    repository = FakeChatSessionRepository()

    session = await StartChatSession(repository).execute(TENANT, USER, "  Returns  ")

    assert (session.title, session.tenant_id, session.user_id) == ("Returns", TENANT, USER)
    assert repository.sessions == [session]


@pytest.mark.parametrize("title", [None, "", "   "])
async def test_a_missing_or_blank_title_is_stored_as_none(title):
    session = await StartChatSession(FakeChatSessionRepository()).execute(TENANT, USER, title)
    assert session.title is None


async def test_a_title_too_long_after_stripping_is_refused_before_anything_is_stored():
    repository = FakeChatSessionRepository()

    with pytest.raises(ValueError):
        await StartChatSession(repository).execute(TENANT, USER, " " + "x" * 201 + " ")

    assert repository.sessions == []


async def test_listing_returns_only_the_callers_own_sessions_newest_first():
    repository = FakeChatSessionRepository()
    start = StartChatSession(repository)
    first = await start.execute(TENANT, USER, "first")
    await start.execute(TENANT, uuid.uuid4(), "another user's")
    await start.execute(uuid.uuid4(), USER, "another tenant's")
    second = await start.execute(TENANT, USER, "second")

    listed = await ListChatSessions(repository).execute(TENANT, USER, limit=10)

    assert listed == [second, first]


@pytest.mark.parametrize("limit", [0, 101])
async def test_listing_refuses_a_limit_outside_1_to_100(limit):
    with pytest.raises(ValueError):
        await ListChatSessions(FakeChatSessionRepository()).execute(TENANT, USER, limit)


@pytest.mark.parametrize("limit", [1, 100])
async def test_listing_accepts_the_limits_at_either_bound(limit):
    assert await ListChatSessions(FakeChatSessionRepository()).execute(TENANT, USER, limit) == []
