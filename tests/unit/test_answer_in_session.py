import uuid

import pytest

from src.orchestration.application.answer_in_session import AnswerInSession
from src.orchestration.domain.errors import SessionNotFound
from tests.unit.session_fakes import FakeChatSessionRepository

TENANT, OWNER = uuid.uuid4(), uuid.uuid4()


class _RecordingAnswerer:
    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID, str]] = []
        self.answer = object()

    async def execute(self, tenant_id, user_id, session_id, question):
        self.calls.append((tenant_id, user_id, session_id, question))
        return self.answer


async def _owned_session(repository: FakeChatSessionRepository) -> uuid.UUID:
    return (await repository.create(TENANT, OWNER, None)).id


async def test_a_question_in_the_callers_own_session_is_answered():
    repository, answerer = FakeChatSessionRepository(), _RecordingAnswerer()
    session_id = await _owned_session(repository)

    result = await AnswerInSession(repository, answerer).execute(TENANT, OWNER, session_id, "q")

    assert result is answerer.answer
    assert answerer.calls == [(TENANT, OWNER, session_id, "q")]


@pytest.mark.parametrize("caller", ["another user", "another tenant", "unknown session"])
async def test_a_session_that_isnt_the_callers_is_refused_before_anything_is_answered(caller):
    repository, answerer = FakeChatSessionRepository(), _RecordingAnswerer()
    session_id = await _owned_session(repository)
    tenant_id, user_id = TENANT, OWNER
    if caller == "another user":
        user_id = uuid.uuid4()
    elif caller == "another tenant":
        tenant_id = uuid.uuid4()
    else:
        session_id = uuid.uuid4()

    with pytest.raises(SessionNotFound):
        await AnswerInSession(repository, answerer).execute(tenant_id, user_id, session_id, "q")

    assert answerer.calls == []
