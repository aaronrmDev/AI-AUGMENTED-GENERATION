import uuid

from src.mag.application.commands.capture_episode import CaptureEpisode
from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
from src.mag.application.queries.retrieve_working_memory import RetrieveWorkingMemory
from src.mag.domain.entities import EpisodicMemory, ScoredEpisode, SemanticMemory
from src.mag.domain.ports import EpisodicMemoryIndex
from src.orchestration.application.state_aware_retrieve import StateAwareRetrieve
from src.rag.domain.entities import SearchResult
from tests.unit.mag_fakes import (
    FakeEpisodicMemoryRepository,
    FakeMemoryGraphRepository,
    FakeSemanticMemoryRepository,
    FakeWorkingMemoryStore,
)
from tests.unit.orchestration_fakes import FakeBagOfWordsEmbeddingModel
from tests.unit.rag_fakes import FakeChatModel, FakeRetriever

_TENANT = uuid.uuid4()
_USER = uuid.uuid4()
_SESSION = uuid.uuid4()


class _SpySemanticMemoryRepository(FakeSemanticMemoryRepository):
    # A review finding caught that no test here asserts the actual
    # user_id/tenant_id values StateAwareRetrieve passes into
    # find_semantic_facts.by_similarity -- FakeSemanticMemoryRepository
    # ignores both for filtering, so a positional argument swap at the
    # call site would pass every existing test unnoticed. This spy
    # records the real call args so a test can pin them directly.
    def __init__(self) -> None:
        super().__init__()
        # Each entry is (user_id, tenant_id, top_k).
        self.search_calls: list[tuple[uuid.UUID, uuid.UUID, int]] = []

    async def search_by_similarity(self, query_embedding, user_id, tenant_id, top_k):
        self.search_calls.append((user_id, tenant_id, top_k))
        return await super().search_by_similarity(query_embedding, user_id, tenant_id, top_k)


class _FakeEpisodicMemoryIndex(EpisodicMemoryIndex):
    # Deliberately local, matching test_capture_episode.py's own established
    # precedent for why this fake doesn't live in tests/unit/mag_fakes.py.
    def __init__(self) -> None:
        self.upserted: list[tuple[EpisodicMemory, uuid.UUID]] = []

    async def upsert(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None:
        self.upserted.append((episode, tenant_id))

    async def search(
        self, query_embedding: list[float], tenant_id: uuid.UUID, session_id: uuid.UUID, top_k: int
    ) -> list[ScoredEpisode]:
        return []


def _build(
    fallback_results: list[SearchResult],
    rewrite_response: str = "rewritten query",
    known_fact: SemanticMemory | None = None,
    semantic_repo: FakeSemanticMemoryRepository | None = None,
) -> tuple[
    StateAwareRetrieve, FakeRetriever, FakeEpisodicMemoryRepository, _FakeEpisodicMemoryIndex
]:
    embedder = FakeBagOfWordsEmbeddingModel()

    semantic_repo = semantic_repo if semantic_repo is not None else FakeSemanticMemoryRepository()
    if known_fact is not None:
        semantic_repo.set_search_results([known_fact])
    find_semantic_facts = FindSemanticFacts(semantic_repo)

    working_memory_store = FakeWorkingMemoryStore()
    retrieve_working_memory = RetrieveWorkingMemory(working_memory_store)

    rewrite_chat_model = FakeChatModel(response=rewrite_response)
    fallback = FakeRetriever(fallback_results)

    episodic_repo = FakeEpisodicMemoryRepository()
    episodic_index = _FakeEpisodicMemoryIndex()
    capture_episode = CaptureEpisode(
        episodic_memory_repository=episodic_repo,
        episodic_memory_index=episodic_index,
        embedding_model=embedder,
        chat_model=FakeChatModel(response='{"salience_score": 0.5}'),
        memory_graph_repository=FakeMemoryGraphRepository(),
    )

    retriever = StateAwareRetrieve(
        embedding_model=embedder,
        find_semantic_facts=find_semantic_facts,
        retrieve_working_memory=retrieve_working_memory,
        chat_model=rewrite_chat_model,
        fallback_retriever=fallback,
        capture_episode=capture_episode,
    )
    return retriever, fallback, episodic_repo, episodic_index


async def test_facts_are_looked_up_with_the_real_user_id_and_tenant_id_not_swapped():
    spy_repo = _SpySemanticMemoryRepository()
    retriever, _, _, _ = _build(fallback_results=[], semantic_repo=spy_repo)

    await retriever.execute(_TENANT, _USER, _SESSION, "how do I visualize this", top_k=5)

    assert len(spy_repo.search_calls) == 1
    user_id, tenant_id, top_k = spy_repo.search_calls[0]
    assert user_id == _USER
    assert tenant_id == _TENANT


async def test_retrieves_using_the_rewritten_query_not_the_raw_one():
    retriever, fallback, _, _ = _build(
        fallback_results=[], rewrite_response="visualize data with matplotlib and pandas"
    )

    await retriever.execute(_TENANT, _USER, _SESSION, "how do I visualize this", top_k=5)

    assert len(fallback.calls) == 1
    assert fallback.calls[0][0] == _TENANT
    assert fallback.calls[0][1] == "visualize data with matplotlib and pandas"


async def test_falls_back_to_the_raw_query_when_the_rewrite_is_blank():
    retriever, fallback, _, _ = _build(fallback_results=[], rewrite_response="   ")

    await retriever.execute(_TENANT, _USER, _SESSION, "how do I visualize this", top_k=5)

    assert len(fallback.calls) == 1
    assert fallback.calls[0][1] == "how do I visualize this"


async def test_only_the_first_line_of_a_multiline_rewrite_is_used():
    retriever, fallback, _, _ = _build(
        fallback_results=[], rewrite_response="the real rewrite\nsome trailing explanation"
    )

    await retriever.execute(_TENANT, _USER, _SESSION, "raw query", top_k=5)

    assert len(fallback.calls) == 1
    assert fallback.calls[0][1] == "the real rewrite"


async def test_fetches_more_candidates_than_top_k_so_the_boost_can_promote_a_lower_ranked_result():
    # The exact bug a real integration run caught: asking the fallback for
    # only top_k results BEFORE boosting would let embedding-only ranking
    # permanently exclude a document the boost step could otherwise have
    # promoted -- this pins the over-fetch-then-rerank shape directly.
    retriever, fallback, _, _ = _build(fallback_results=[], rewrite_response="query")

    await retriever.execute(_TENANT, _USER, _SESSION, "query", top_k=1)

    assert fallback.calls[0][2] > 1


async def test_ranking_boost_reorders_results_toward_fact_matching_content():
    # Constructed so the naive fallback order (by original score) would
    # NOT already put the fact-matching result first -- the boost has to
    # do real work to flip this, not just agree with what was already true.
    generic = SearchResult(
        document_id=uuid.uuid4(), chunk_id=uuid.uuid4(),
        content="a generic overview of visualization tools", score=0.80,
    )
    matplotlib_specific = SearchResult(
        document_id=uuid.uuid4(), chunk_id=uuid.uuid4(),
        content="matplotlib pyplot tutorial for data visualization", score=0.75,
    )
    # FakeBagOfWordsEmbeddingModel is stateless/deterministic -- a fresh
    # instance embeds identically to the one _build constructs internally.
    fact_value = "matplotlib pyplot"
    fact = SemanticMemory(
        id=uuid.uuid4(), user_id=_USER, fact_key="preferred_library",
        fact_value=fact_value, embedding=FakeBagOfWordsEmbeddingModel().embed(fact_value),
    )

    retriever, _, _, _ = _build(
        fallback_results=[generic, matplotlib_specific], known_fact=fact
    )

    # top_k=1 on purpose: without the over-fetch-then-rerank fix, the
    # fallback would only ever have been asked for 1 result (whichever the
    # naive embedding order preferred -- generic, per its higher raw
    # score), and the boost would never see matplotlib_specific at all.
    results = await retriever.execute(_TENANT, _USER, _SESSION, "how do I visualize this", top_k=1)

    assert len(results) == 1
    assert results[0].document_id == matplotlib_specific.document_id
    assert results[0].score > matplotlib_specific.score  # boost actually raised it


async def test_writes_back_a_real_episode_recording_the_interaction():
    result = SearchResult(
        document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="matplotlib docs", score=0.9
    )
    retriever, _, episodic_repo, _ = _build(fallback_results=[result])

    await retriever.execute(_TENANT, _USER, _SESSION, "how do I visualize this", top_k=5)

    episodes = await episodic_repo.get_by_session(_SESSION, _TENANT)
    assert len(episodes) == 1
    assert episodes[0].content["type"] == "state_aware_retrieval"
    assert episodes[0].content["query"] == "how do I visualize this"
    assert episodes[0].content["top_result_content"] == "matplotlib docs"


async def test_write_back_handles_no_results_without_crashing():
    retriever, _, episodic_repo, _ = _build(fallback_results=[])

    await retriever.execute(_TENANT, _USER, _SESSION, "an unanswerable query", top_k=5)

    episodes = await episodic_repo.get_by_session(_SESSION, _TENANT)
    assert episodes[0].content["top_result_content"] is None
