import uuid
from datetime import UTC, datetime

from src.mag.application.queries.retrieve_episodes import RetrieveEpisodes
from src.mag.domain.entities import EpisodicMemory
from tests.unit.mag_fakes import FakeEpisodicMemoryRepository


def _episode(session_id: uuid.UUID) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id,
        content={"input": "hi"},
        embedding=[0.0] * 384,
        timestamp=datetime.now(UTC),
    )


async def test_execute_delegates_to_get_by_session():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    episode = _episode(session_id)
    await repo.save(episode, tenant_id)

    result = await RetrieveEpisodes(repo).execute(tenant_id=tenant_id, session_id=session_id)

    assert result == [episode]


async def test_execute_returns_an_empty_list_for_a_session_with_no_episodes():
    repo = FakeEpisodicMemoryRepository()

    result = await RetrieveEpisodes(repo).execute(
        tenant_id=uuid.uuid4(), session_id=uuid.uuid4()
    )

    assert result == []


async def test_execute_does_not_return_another_sessions_episodes():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_a = uuid.uuid4()
    session_b = uuid.uuid4()
    episode_a = _episode(session_a)
    episode_b = _episode(session_b)
    await repo.save(episode_a, tenant_id)
    await repo.save(episode_b, tenant_id)

    result = await RetrieveEpisodes(repo).execute(tenant_id=tenant_id, session_id=session_a)

    assert result == [episode_a]
