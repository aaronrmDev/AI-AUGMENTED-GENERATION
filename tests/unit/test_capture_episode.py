import json
import uuid

from src.mag.application.commands.capture_episode import CaptureEpisode
from src.mag.domain.entities import EpisodicMemory
from src.mag.domain.ports import EpisodicMemoryIndex
from tests.unit.mag_fakes import FakeEpisodicMemoryRepository
from tests.unit.rag_fakes import FakeEmbeddingModel


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
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[EpisodicMemory]:
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
) -> CaptureEpisode:
    return CaptureEpisode(
        episodic_memory_repository=repo,
        episodic_memory_index=index,
        embedding_model=embedder or FakeEmbeddingModel(),
    )


async def test_execute_saves_the_episode_to_the_postgres_repository():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()

    episode = await _use_case(repo, index).execute(
        tenant_id=tenant_id, session_id=session_id, content={"input": "hello"}
    )

    assert repo.saved == [(episode, tenant_id)]


async def test_execute_upserts_the_episode_into_the_qdrant_index():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    tenant_id = uuid.uuid4()
    session_id = uuid.uuid4()

    episode = await _use_case(repo, index).execute(
        tenant_id=tenant_id, session_id=session_id, content={"input": "hello"}
    )

    assert index.upserted == [(episode, tenant_id)]


async def test_execute_generates_a_fresh_uuid_for_every_episode():
    repo = FakeEpisodicMemoryRepository()
    index = FakeQdrantEpisodicMemoryIndex()
    use_case = _use_case(repo, index)

    first = await use_case.execute(
        tenant_id=uuid.uuid4(), session_id=uuid.uuid4(), content={"n": 1}
    )
    second = await use_case.execute(
        tenant_id=uuid.uuid4(), session_id=uuid.uuid4(), content={"n": 2}
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
        tenant_id=uuid.uuid4(), session_id=uuid.uuid4(), content=content
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
        tenant_id=tenant_id, session_id=session_id, content=content
    )

    assert isinstance(episode, EpisodicMemory)
    assert episode.session_id == session_id
    assert episode.content == content
    assert episode.salience_score == 0.0
