import uuid
from datetime import UTC, datetime, timedelta

from src.mag.application.queries.retrieve_by_temporal_window import TemporalRetrieval
from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from tests.unit.mag_fakes import FakeEpisodicMemoryRepository


def _episode(session_id: uuid.UUID, timestamp: datetime) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id,
        content={"input": "hi"},
        embedding=[0.0] * 384,
        timestamp=timestamp,
    )


async def test_within_given_returns_only_in_window_episodes_all_scored_uniformly():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    base = datetime.now(UTC)
    inside_early = _episode(session_id, base)
    inside_late = _episode(session_id, base + timedelta(minutes=5))
    outside = _episode(session_id, base + timedelta(days=1))
    await repo.save(inside_early, tenant_id)
    await repo.save(inside_late, tenant_id)
    await repo.save(outside, tenant_id)

    result = await TemporalRetrieval(repo).execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=10,
        within=(base, base + timedelta(minutes=30)),
    )

    # get_by_session_in_window returns newest first.
    assert result == [
        ScoredEpisode(episode=inside_late, score=1.0),
        ScoredEpisode(episode=inside_early, score=1.0),
    ]


async def test_within_given_truncates_to_top_k():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    base = datetime.now(UTC)
    episodes = [_episode(session_id, base + timedelta(minutes=i)) for i in range(5)]
    for e in episodes:
        await repo.save(e, tenant_id)

    result = await TemporalRetrieval(repo).execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=2,
        within=(base, base + timedelta(hours=1)),
    )

    assert len(result) == 2
    # Newest first, so the last two episodes saved come back first.
    assert [s.episode.id for s in result] == [episodes[4].id, episodes[3].id]
    assert all(s.score == 1.0 for s in result)


async def test_within_given_on_empty_session_returns_empty_list():
    repo = FakeEpisodicMemoryRepository()

    result = await TemporalRetrieval(repo).execute(
        tenant_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        top_k=10,
        within=(datetime.now(UTC), datetime.now(UTC)),
    )

    assert result == []


async def test_within_given_does_not_return_another_sessions_episodes():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_a = uuid.uuid4()
    session_b = uuid.uuid4()
    base = datetime.now(UTC)
    episode_a = _episode(session_a, base)
    episode_b = _episode(session_b, base)
    await repo.save(episode_a, tenant_id)
    await repo.save(episode_b, tenant_id)

    result = await TemporalRetrieval(repo).execute(
        tenant_id=tenant_id,
        session_id=session_a,
        top_k=10,
        within=(base - timedelta(minutes=1), base + timedelta(minutes=1)),
    )

    assert [s.episode.id for s in result] == [episode_a.id]


async def test_within_omitted_returns_recent_episodes_with_descending_graded_scores():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    base = datetime.now(UTC)
    episodes = [_episode(session_id, base + timedelta(minutes=i)) for i in range(4)]
    for e in episodes:
        await repo.save(e, tenant_id)

    result = await TemporalRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, top_k=10
    )

    # Newest first: episodes[3], episodes[2], episodes[1], episodes[0].
    assert [s.episode.id for s in result] == [
        episodes[3].id,
        episodes[2].id,
        episodes[1].id,
        episodes[0].id,
    ]
    scores = [s.score for s in result]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]
    assert all(0.0 < score <= 1.0 for score in scores)
    assert scores[0] == 1.0


async def test_within_omitted_truncates_to_top_k():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    base = datetime.now(UTC)
    episodes = [_episode(session_id, base + timedelta(minutes=i)) for i in range(5)]
    for e in episodes:
        await repo.save(e, tenant_id)

    result = await TemporalRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, top_k=2
    )

    assert len(result) == 2
    assert [s.episode.id for s in result] == [episodes[4].id, episodes[3].id]
    assert result[0].score == 1.0
    assert 0.0 < result[1].score < 1.0


async def test_within_omitted_on_empty_session_returns_empty_list():
    repo = FakeEpisodicMemoryRepository()

    result = await TemporalRetrieval(repo).execute(
        tenant_id=uuid.uuid4(), session_id=uuid.uuid4(), top_k=10
    )

    assert result == []


async def test_within_omitted_does_not_return_another_sessions_episodes():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_a = uuid.uuid4()
    session_b = uuid.uuid4()
    base = datetime.now(UTC)
    episode_a = _episode(session_a, base)
    episode_b = _episode(session_b, base)
    await repo.save(episode_a, tenant_id)
    await repo.save(episode_b, tenant_id)

    result = await TemporalRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_a, top_k=10
    )

    assert [s.episode.id for s in result] == [episode_a.id]
