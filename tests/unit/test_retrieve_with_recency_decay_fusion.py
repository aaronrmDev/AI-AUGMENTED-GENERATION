import uuid
from datetime import UTC, datetime, timedelta

from src.mag.application.queries.retrieve_with_recency_decay_fusion import (
    RecencyDecayFusionRetrieval,
)
from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryIndex
from tests.unit.mag_fakes import FakeEpisodicMemoryRepository
from tests.unit.rag_fakes import FakeChatModel

# Large enough that exp(-ln2 * age_hours / half_life) rounds to 1.0 for any
# age this test suite ever constructs -- isolates the pure cross-strategy
# fusion math from decay, in tests that aren't specifically about decay.
_NO_EFFECTIVE_DECAY_HALF_LIFE = 1e9


class _FakeEpisodicMemoryIndex(EpisodicMemoryIndex):
    # Deliberately local, not added to tests/unit/mag_fakes.py -- matches
    # test_capture_episode.py's identical convention for the same port.
    def __init__(self, results: list[ScoredEpisode] | None = None) -> None:
        self._results = results or []
        self.search_called = False

    async def ensure_collection(self) -> None:
        pass

    async def upsert(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None:
        raise NotImplementedError

    async def search(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredEpisode]:
        self.search_called = True
        return self._results[:top_k]


def _episode(
    session_id: uuid.UUID,
    timestamp: datetime,
    salience_score: float = 0.0,
    content: dict | None = None,
) -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid.uuid4(),
        session_id=session_id,
        content=content or {"input": "hi"},
        embedding=[0.1] * 384,
        timestamp=timestamp,
        salience_score=salience_score,
    )


def _fusion(
    repo: FakeEpisodicMemoryRepository,
    index: _FakeEpisodicMemoryIndex | None = None,
    chat_model: FakeChatModel | None = None,
) -> RecencyDecayFusionRetrieval:
    return RecencyDecayFusionRetrieval(
        episodic_memory_repository=repo,
        episodic_memory_index=index or _FakeEpisodicMemoryIndex(),
        chat_model=chat_model or FakeChatModel(response='{"scores": []}'),
    )


async def test_only_temporal_and_salience_run_when_no_optional_params_are_given():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    now = datetime.now(UTC)
    await repo.save(_episode(session_id, now, salience_score=0.5), tenant_id)
    index = _FakeEpisodicMemoryIndex()
    chat_model = FakeChatModel(response='{"scores": []}')

    await _fusion(repo, index, chat_model).execute(
        tenant_id=tenant_id, session_id=session_id, top_k=5, now=now
    )

    assert index.search_called is False
    assert chat_model.last_prompt is None


async def test_all_five_strategies_run_when_every_optional_param_is_given():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    now = datetime.now(UTC)
    episode = _episode(session_id, now, salience_score=0.5, content={"entities": ["alice"]})
    await repo.save(episode, tenant_id)
    index = _FakeEpisodicMemoryIndex(
        [ScoredEpisode(episode=_episode(session_id, now), score=0.8)]
    )
    chat_model = FakeChatModel(response='{"scores": [{"episode_index": 1, "score": 0.7}]}')

    await _fusion(repo, index, chat_model).execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=5,
        query_embedding=[0.1] * 384,
        causal_query="why did it fail",
        entity="alice",
        now=now,
    )

    assert index.search_called is True
    assert chat_model.last_prompt is not None


async def test_top_k_truncates_the_fused_result():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    now = datetime.now(UTC)
    for i in range(5):
        episode = _episode(session_id, now - timedelta(minutes=i), salience_score=i)
        await repo.save(episode, tenant_id)

    result = await _fusion(repo).execute(
        tenant_id=tenant_id, session_id=session_id, top_k=2, now=now
    )

    assert len(result) == 2


async def test_an_empty_session_returns_an_empty_list():
    repo = FakeEpisodicMemoryRepository()

    result = await _fusion(repo).execute(
        tenant_id=uuid.uuid4(), session_id=uuid.uuid4(), top_k=5
    )

    assert result == []


async def test_moderate_agreement_across_both_strategies_beats_being_extreme_in_only_one():
    # Y is temporal's #1 (most recent) but salience's worst; Z is salience's
    # #1 (highest salience_score) but temporal's oldest/worst; X is 2nd
    # place in BOTH. Min-max normalization within each strategy plus equal
    # weighting means X's balanced, cross-strategy-agreed relevance beats
    # either single-strategy extreme -- this is #12's "fuse ... with
    # heuristic weights" actually rewarding agreement, not just echoing
    # whichever strategy scored highest.
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    now = datetime.now(UTC)
    episode_z = _episode(session_id, now - timedelta(hours=2), salience_score=1.0)
    episode_x = _episode(session_id, now - timedelta(hours=1), salience_score=0.6)
    episode_y = _episode(session_id, now, salience_score=0.0)
    await repo.save(episode_z, tenant_id)
    await repo.save(episode_x, tenant_id)
    await repo.save(episode_y, tenant_id)

    fusion = RecencyDecayFusionRetrieval(
        episodic_memory_repository=repo,
        episodic_memory_index=_FakeEpisodicMemoryIndex(),
        chat_model=FakeChatModel(response='{"scores": []}'),
    )
    result = await fusion.execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=3,
        now=now,
        decay_half_life_hours=_NO_EFFECTIVE_DECAY_HALF_LIFE,
    )

    scores = {s.episode.id: s.score for s in result}
    assert scores[episode_x.id] > scores[episode_y.id]
    assert scores[episode_x.id] > scores[episode_z.id]


async def test_recency_decay_favors_a_newer_episode_over_an_otherwise_identical_older_one():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    now = datetime.now(UTC)
    # Identical salience so the only difference fusion can react to is age.
    newer = _episode(session_id, now, salience_score=0.5)
    older = _episode(session_id, now - timedelta(hours=48), salience_score=0.5)
    await repo.save(newer, tenant_id)
    await repo.save(older, tenant_id)

    result = await _fusion(repo).execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=2,
        now=now,
        decay_half_life_hours=24.0,
    )

    scores = {s.episode.id: s.score for s in result}
    assert scores[newer.id] > scores[older.id]


async def test_custom_weights_override_the_default_equal_weighting():
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    now = datetime.now(UTC)
    # high_salience is temporal's worst (oldest) but salience's best;
    # recent_only is temporal's best (newest) but salience's worst.
    high_salience = _episode(session_id, now - timedelta(hours=1), salience_score=1.0)
    recent_only = _episode(session_id, now, salience_score=0.0)
    await repo.save(high_salience, tenant_id)
    await repo.save(recent_only, tenant_id)

    result = await _fusion(repo).execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=2,
        now=now,
        decay_half_life_hours=_NO_EFFECTIVE_DECAY_HALF_LIFE,
        weights={"temporal": 0.0, "salience": 1.0},
    )

    scores = {s.episode.id: s.score for s in result}
    assert scores[high_salience.id] > scores[recent_only.id]


async def test_a_flat_score_distribution_within_one_strategy_normalizes_to_full_weight():
    # Two episodes tied on salience_score must not cancel each other out to
    # 0.0 under min-max normalization (hi == lo) -- both should carry that
    # strategy's full normalized weight, same reasoning CausalRetrieval's
    # all-0.0 exhausted-retry floor depends on downstream.
    repo = FakeEpisodicMemoryRepository()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    now = datetime.now(UTC)
    tied_a = _episode(session_id, now, salience_score=0.3)
    tied_b = _episode(session_id, now, salience_score=0.3)
    await repo.save(tied_a, tenant_id)
    await repo.save(tied_b, tenant_id)

    result = await _fusion(repo).execute(
        tenant_id=tenant_id,
        session_id=session_id,
        top_k=2,
        now=now,
        decay_half_life_hours=_NO_EFFECTIVE_DECAY_HALF_LIFE,
    )

    assert len(result) == 2
    assert all(s.score > 0.0 for s in result)
