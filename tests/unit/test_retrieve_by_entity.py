import uuid
from datetime import UTC, datetime, timedelta

from src.mag.application.queries.retrieve_by_entity import EntityRetrieval
from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from tests.unit.mag_fakes import FakeEpisodicMemoryRepository


def _episode(
    session_id: uuid.UUID, content: dict, timestamp: datetime | None = None
) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id,
        content=content,
        embedding=[0.0] * 384,
        timestamp=timestamp or datetime.now(UTC),
    )


async def test_matches_via_structured_entities_list():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    matching = _episode(session_id, {"entities": ["Acme Corp", "Bob"], "output": "unrelated"})
    await repo.save(matching, tenant_id)

    result = await EntityRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, entity="Acme Corp", top_k=10
    )

    assert result == [ScoredEpisode(episode=matching, score=1.0)]


async def test_matches_via_substring_fallback_when_entities_not_populated():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    matching = _episode(session_id, {"output": "We discussed Acme Corp's quarterly earnings"})
    await repo.save(matching, tenant_id)

    result = await EntityRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, entity="Acme Corp", top_k=10
    )

    assert result == [ScoredEpisode(episode=matching, score=1.0)]


async def test_excludes_a_non_matching_episode():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    non_matching = _episode(session_id, {"entities": ["Bob"], "output": "unrelated content"})
    await repo.save(non_matching, tenant_id)

    result = await EntityRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, entity="Acme Corp", top_k=10
    )

    assert result == []


async def test_truncates_to_top_k():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    base = datetime.now(UTC)
    episodes = [
        _episode(session_id, {"entities": ["Acme Corp"]}, timestamp=base + timedelta(minutes=i))
        for i in range(5)
    ]
    for e in episodes:
        await repo.save(e, tenant_id)

    result = await EntityRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, entity="Acme Corp", top_k=2
    )

    assert len(result) == 2
    # Newest first, so the last two episodes saved come back first.
    assert [s.episode.id for s in result] == [episodes[4].id, episodes[3].id]


async def test_every_result_scores_exactly_one():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    episodes = [_episode(session_id, {"entities": ["Acme Corp"]}) for _ in range(3)]
    for e in episodes:
        await repo.save(e, tenant_id)

    result = await EntityRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, entity="Acme Corp", top_k=10
    )

    assert len(result) == 3
    assert all(s.score == 1.0 for s in result)


async def test_entity_with_zero_matches_returns_empty_list():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    await repo.save(_episode(session_id, {"entities": ["Bob"]}), tenant_id)

    result = await EntityRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_id, entity="Nonexistent Entity", top_k=10
    )

    assert result == []


async def test_does_not_return_another_sessions_episodes():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_a = uuid.uuid4()
    session_b = uuid.uuid4()
    episode_a = _episode(session_a, {"entities": ["Acme Corp"]})
    episode_b = _episode(session_b, {"entities": ["Acme Corp"]})
    await repo.save(episode_a, tenant_id)
    await repo.save(episode_b, tenant_id)

    result = await EntityRetrieval(repo).execute(
        tenant_id=tenant_id, session_id=session_a, entity="Acme Corp", top_k=10
    )

    assert [s.episode.id for s in result] == [episode_a.id]
