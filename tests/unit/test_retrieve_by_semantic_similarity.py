import uuid
from datetime import UTC, datetime

from src.mag.application.queries.retrieve_by_semantic_similarity import (
    SemanticSimilarityRetrieval,
)
from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryIndex


class _FakeEpisodicMemoryIndex(EpisodicMemoryIndex):
    # Deliberately local, not added to tests/unit/mag_fakes.py -- matches
    # test_capture_episode.py's identical convention for the same port.
    def __init__(self, results: list[ScoredEpisode]) -> None:
        self._results = results
        self.last_call: tuple[list[float], uuid.UUID, int] | None = None

    async def ensure_collection(self) -> None:
        pass

    async def upsert(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None:
        raise NotImplementedError

    async def search(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredEpisode]:
        self.last_call = (query_embedding, tenant_id, top_k)
        return self._results[:top_k]


def _scored_episode(score: float) -> ScoredEpisode:
    episode = EpisodicMemory(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        content={"input": "hi"},
        embedding=[0.1] * 384,
        timestamp=datetime.now(UTC),
    )
    return ScoredEpisode(episode=episode, score=score)


async def test_execute_delegates_to_the_index_search_and_returns_its_scored_results():
    results = [_scored_episode(0.9), _scored_episode(0.5)]
    index = _FakeEpisodicMemoryIndex(results)
    tenant_id = uuid.uuid4()
    query_embedding = [0.2] * 384

    result = await SemanticSimilarityRetrieval(index).execute(
        tenant_id=tenant_id, query_embedding=query_embedding, top_k=2
    )

    assert result == results
    assert index.last_call == (query_embedding, tenant_id, 2)


async def test_execute_returns_an_empty_list_when_the_index_has_no_matches():
    index = _FakeEpisodicMemoryIndex([])

    result = await SemanticSimilarityRetrieval(index).execute(
        tenant_id=uuid.uuid4(), query_embedding=[0.1] * 384, top_k=5
    )

    assert result == []
