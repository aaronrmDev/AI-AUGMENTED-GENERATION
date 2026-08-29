import uuid
from datetime import UTC, datetime

from src.mag.application.queries.retrieve_by_salience import SalienceRetrieval
from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from tests.unit.mag_fakes import FakeEpisodicMemoryRepository


def _episode(session_id: uuid.UUID, salience_score: float = 0.0) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id,
        content={"input": "hi"},
        embedding=[0.0] * 384,
        timestamp=datetime.now(UTC),
        salience_score=salience_score,
    )


async def test_execute_orders_episodes_highest_salience_first():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    low = _episode(session_id, salience_score=0.2)
    high = _episode(session_id, salience_score=0.9)
    mid = _episode(session_id, salience_score=0.5)
    await repo.save(low, tenant_id)
    await repo.save(high, tenant_id)
    await repo.save(mid, tenant_id)

    result = await SalienceRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, top_k=10
    )

    assert [scored.episode.id for scored in result] == [high.id, mid.id, low.id]


async def test_execute_score_is_the_episodes_real_salience_score():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    episode = _episode(session_id, salience_score=0.6789)
    await repo.save(episode, tenant_id)

    result = await SalienceRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, top_k=10
    )

    # Not rounded, not derived -- exactly the entity's own salience_score.
    assert result == [ScoredEpisode(episode=episode, score=0.6789)]
    assert result[0].score == episode.salience_score


async def test_execute_truncates_to_top_k():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    episodes = [_episode(session_id, salience_score=float(i)) for i in range(5)]
    for episode in episodes:
        await repo.save(episode, tenant_id)

    result = await SalienceRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, top_k=2
    )

    assert len(result) == 2
    assert [scored.episode.salience_score for scored in result] == [4.0, 3.0]


async def test_execute_returns_an_empty_list_for_a_session_with_no_episodes():
    repo = FakeEpisodicMemoryRepository()

    result = await SalienceRetrieval(repo).execute(
        tenant_id=uuid.uuid4(), session_id=uuid.uuid4(), top_k=10
    )

    assert result == []


async def test_execute_does_not_return_another_sessions_episodes():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_a = uuid.uuid4()
    session_b = uuid.uuid4()
    episode_a = _episode(session_a, salience_score=0.1)
    episode_b = _episode(session_b, salience_score=0.9)
    await repo.save(episode_a, tenant_id)
    await repo.save(episode_b, tenant_id)

    result = await SalienceRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_a, top_k=10
    )

    assert [scored.episode.id for scored in result] == [episode_a.id]
