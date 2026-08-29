import json
import uuid

from src.mag.application.commands.capture_episode import CaptureEpisode
from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryIndex
from tests.unit.mag_fakes import FakeEpisodicMemoryRepository, FakeMemoryGraphRepository
from tests.unit.rag_fakes import FakeChatModel, FakeEmbeddingModel


class FakeQdrantEpisodicMemoryIndex(EpisodicMemoryIndex):
    # Deliberately local, not added to tests/unit/mag_fakes.py: that file is
    # shared with two other MAG verticals running in parallel in this same
    # worktree, and this fake only stands in for this vertical's own
    # infrastructure class.
    def __init__(self) -> None:
        self.upserted: list[tuple[EpisodicMemory, uuid.UUID]] = []

    async def ensure_collection(self) -> None:
        pass

    async def upsert(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None:
        self.upserted.append((episode, tenant_id))

    async def search(
        self,
        query_embedding: list[float],
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        top_k: int,
    ) -> list[ScoredEpisode]:
        return []


class _SpyEmbeddingModel(FakeEmbeddingModel):
    # FakeEmbeddingModel hashes purely by len(text) % 7 -- two strings of
    # equal length embed identically, which is exactly the case a
    # sort_keys=True regression test needs to tell apart (json.dumps with
    # and without sort_keys produces same-length strings for the same dict,
    # just reordered). This spy records the literal text it was asked to
    # embed so a test can assert on that directly, instead of trying to
    # infer it from an embedding whose own hash can't distinguish the cases
    # under test.
    def __init__(self) -> None:
        super().__init__()
        self.embedded_texts: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        return super().embed(text)


def _use_case(
    repo: FakeEpisodicMemoryRepository,
    index: FakeQdrantEpisodicMemoryIndex,
    embedder: FakeEmbeddingModel | None = None,
    chat_model: FakeChatModel | None = None,
    graph: FakeMemoryGraphRepository | None = None,
) -> CaptureEpisode:
    return CaptureEpisode(
        episodic_memory_repository=repo,
        episodic_memory_index=index,
        embedding_model=embedder or FakeEmbeddingModel(),
        # Defaults to a fixed low score so tests that don't care about
        # salience get a stable, predictable value rather than 0.0 reading
        # as "the feature doesn't exist" -- see the dedicated salience tests
        # below for the retry/validation/fallback behavior itself.
        chat_model=chat_model or FakeChatModel(response='{"salience_score": 0.2}'),
        memory_graph_repository=graph or FakeMemoryGraphRepository(),
    )


async def test_execute_saves_the_episode_to_the_postgres_repository():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()

    episode = await _use_case(repo, index).execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id, content={"input": "hello"}
    )

    assert repo.saved == [(episode, tenant_id)]


async def test_execute_upserts_the_episode_into_the_qdrant_index():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()

    episode = await _use_case(repo, index).execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id, content={"input": "hello"}
    )

    assert index.upserted == [(episode, tenant_id)]


async def test_execute_generates_a_fresh_uuid_for_every_episode():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    use_case = _use_case(repo, index)

    first = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), session_id=uuid.uuid4(), content={"n": 1}
    )
    second = await use_case.execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), session_id=uuid.uuid4(), content={"n": 2}
    )

    assert first.id != second.id
    assert isinstance(first.id, uuid.UUID)


async def test_execute_embeds_the_json_serialized_content_with_sorted_keys():
    # Regression-proof against a same-length-string collision: the
    # FakeEmbeddingModel this test used to rely on can't tell
    # '{"b": ..., "a": ...}' apart from '{"a": ..., "b": ...}' by output
    # alone (both hash to the same length-derived vector), so a bug that
    # dropped sort_keys=True entirely would not have failed this test. The
    # spy asserts on the literal text handed to embed() instead.
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    embedder = _SpyEmbeddingModel()
    content = {"b": "second", "a": "first"}

    await _use_case(repo, index, embedder).execute(
        tenant_id=uuid.uuid4(), user_id=uuid.uuid4(), session_id=uuid.uuid4(), content=content
    )

    assert embedder.embedded_texts == [json.dumps(content, sort_keys=True)]
    assert embedder.embedded_texts != [json.dumps(content)]


async def test_execute_returns_an_episodic_memory_carrying_the_given_session_and_content():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()
    content = {"input": "what's the weather", "output": "sunny"}

    episode = await _use_case(repo, index).execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=session_id, content=content
    )

    assert isinstance(episode, EpisodicMemory)
    assert episode.session_id == session_id
    assert episode.content == content
    assert episode.salience_score == 0.2


async def test_execute_sets_salience_score_from_the_chat_models_json_response():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    chat_model = FakeChatModel(response='{"salience_score": 0.85}')

    episode = await _use_case(repo, index, chat_model=chat_model).execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"outcome": "failure"},
    )

    assert episode.salience_score == 0.85


async def test_execute_strips_a_markdown_fence_around_the_salience_response():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    chat_model = FakeChatModel(response='```json\n{"salience_score": 0.6}\n```')

    episode = await _use_case(repo, index, chat_model=chat_model).execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"input": "hi"},
    )

    assert episode.salience_score == 0.6


async def test_execute_defaults_salience_score_to_zero_after_exhausting_retries_on_bad_json():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    chat_model = FakeChatModel(response="not json at all")

    episode = await _use_case(repo, index, chat_model=chat_model).execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"input": "hi"},
    )

    assert episode.salience_score == 0.0


async def test_execute_defaults_salience_score_to_zero_when_out_of_range():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    chat_model = FakeChatModel(response='{"salience_score": 1.5}')

    episode = await _use_case(repo, index, chat_model=chat_model).execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"input": "hi"},
    )

    assert episode.salience_score == 0.0


async def test_execute_defaults_salience_score_to_zero_when_the_model_returns_a_boolean():
    # Regression test: bool is a subclass of int in Python, so
    # float(True) == 1.0 -- without an explicit isinstance(x, bool) guard, a
    # model returning {"salience_score": true} would silently pass the
    # 0.0-1.0 range check as a maximal score on the first attempt, with no
    # retry. A review caught this gap (present in retrieve_by_causal_
    # relevance.py's validator but missing here) before it shipped.
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    chat_model = FakeChatModel(response='{"salience_score": true}')

    episode = await _use_case(repo, index, chat_model=chat_model).execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"input": "hi"},
    )

    assert episode.salience_score == 0.0


async def test_execute_saves_the_episode_with_its_computed_salience_score():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    chat_model = FakeChatModel(response='{"salience_score": 0.7}')

    episode = await _use_case(repo, index, chat_model=chat_model).execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"input": "hi"},
    )

    assert repo.saved == [(episode, repo.saved[0][1])]
    assert repo.saved[0][0].salience_score == 0.7


async def test_execute_upserts_an_episode_node_into_the_memory_graph():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    graph = FakeMemoryGraphRepository()
    tenant_id = uuid.uuid4()

    episode = await _use_case(repo, index, graph=graph).execute(
        tenant_id=tenant_id, user_id=uuid.uuid4(), session_id=uuid.uuid4(), content={"input": "hi"}
    )

    assert graph.upserted_episodes == [(episode, tenant_id)]


async def test_execute_links_participated_in_from_the_given_user_and_session():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    graph = FakeMemoryGraphRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()

    await _use_case(repo, index, graph=graph).execute(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id, content={"input": "hi"}
    )

    assert graph.participated_in_links == [(user_id, session_id, tenant_id)]


async def test_execute_does_not_link_temporally_follows_for_the_first_episode_in_a_session():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    graph = FakeMemoryGraphRepository()

    await _use_case(repo, index, graph=graph).execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"input": "hi"},
    )

    assert graph.temporally_follows_links == []


async def test_execute_links_temporally_follows_from_the_sessions_previous_episode():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    graph = FakeMemoryGraphRepository()
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    use_case = _use_case(repo, index, graph=graph)

    first = await use_case.execute(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id, content={"n": 1}
    )
    second = await use_case.execute(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id, content={"n": 2}
    )

    assert graph.temporally_follows_links == [(first.id, second.id, tenant_id)]


async def test_execute_links_mentions_for_every_entity_in_content():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    graph = FakeMemoryGraphRepository()
    tenant_id = uuid.uuid4()

    episode = await _use_case(repo, index, graph=graph).execute(
        tenant_id=tenant_id,
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"input": "hi", "entities": ["Paris", "Bob"]},
    )

    assert graph.mentions_links == [
        (episode.id, "Paris", tenant_id),
        (episode.id, "Bob", tenant_id),
    ]


async def test_execute_links_no_mentions_when_content_has_no_entities():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    graph = FakeMemoryGraphRepository()

    await _use_case(repo, index, graph=graph).execute(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"input": "hi"},
    )

    assert graph.mentions_links == []
