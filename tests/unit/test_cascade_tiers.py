import uuid

import pytest

from src.mag.domain.entities import ScoredFact, SemanticMemory
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.cascade_tiers import CagTier, MagTier, RagTier
from src.orchestration.domain.entities import Paradigm, TierOutcome, TierRequest
from src.orchestration.domain.ports import SemanticFactSearch
from src.orchestration.domain.similarity import cosine_similarity
from src.rag.domain.entities import SearchResult
from tests.unit.mag_fakes import FakeSemanticMemoryRepository
from tests.unit.orchestration_fakes import FakeBagOfWordsEmbeddingModel, FakeFrozenCache
from tests.unit.rag_fakes import FakeRetriever

_TENANT = uuid.uuid4()
_USER = uuid.uuid4()
_WARMED = "return policy allows customers to return unopened items within thirty days"
_PARTIAL_QUERY = "what is the return policy for unopened items"
_EMBEDDER = FakeBagOfWordsEmbeddingModel()


def _request(query="q", embedding=None, tenant_id=_TENANT, user_id=_USER) -> TierRequest:
    return TierRequest(
        tenant_id, user_id, uuid.uuid4(), query, embedding if embedding is not None else [1.0, 0.0]
    )


def _warmed_retriever():
    cache = FakeFrozenCache()
    retriever = CacheWarmedRetrieve(_EMBEDDER, cache, FakeRetriever(), similarity_threshold=0.99)
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED)
    retriever.note_warmed(_TENANT, document_id, _WARMED)
    return retriever, cache, document_id


def _partial_score() -> float:
    score = cosine_similarity(_EMBEDDER.embed(_PARTIAL_QUERY), _EMBEDDER.embed(_WARMED))
    assert 0.1 < score < 0.95, "fake-embedder precondition"
    return score


async def test_cag_tier_reports_a_hit_at_or_above_the_hit_threshold():
    retriever, _, document_id = _warmed_retriever()
    tier = CagTier(retriever, hit_threshold=0.9, partial_threshold=0.5)
    result = await tier.attempt(_request(_WARMED, _EMBEDDER.embed(_WARMED)))
    assert result.outcome is TierOutcome.HIT
    [item] = result.items
    assert (item.paradigm, item.content, item.source_id) == (Paradigm.CAG, _WARMED, document_id)
    assert item.score == pytest.approx(1.0)


async def test_cag_tier_reports_a_partial_between_the_two_thresholds():
    retriever, _, _ = _warmed_retriever()
    score = _partial_score()
    tier = CagTier(retriever, hit_threshold=score + 0.01, partial_threshold=score - 0.01)
    result = await tier.attempt(_request(_PARTIAL_QUERY, _EMBEDDER.embed(_PARTIAL_QUERY)))
    assert result.outcome is TierOutcome.PARTIAL
    assert [item.content for item in result.items] == [_WARMED]


async def test_cag_tier_reports_a_miss_below_the_partial_threshold():
    retriever, _, _ = _warmed_retriever()
    score = _partial_score()
    tier = CagTier(retriever, hit_threshold=2.0, partial_threshold=score + 0.05)
    result = await tier.attempt(_request(_PARTIAL_QUERY, _EMBEDDER.embed(_PARTIAL_QUERY)))
    assert result.outcome is TierOutcome.MISS
    assert result.items == []


async def test_cag_tier_misses_for_another_tenant():
    retriever, _, _ = _warmed_retriever()
    tier = CagTier(retriever, hit_threshold=0.9, partial_threshold=0.5)
    request = _request(_WARMED, _EMBEDDER.embed(_WARMED), tenant_id=uuid.uuid4())
    assert (await tier.attempt(request)).outcome is TierOutcome.MISS


async def test_cag_tier_misses_once_the_document_is_evicted():
    retriever, cache, document_id = _warmed_retriever()
    cache.evict(_TENANT, document_id)
    tier = CagTier(retriever, hit_threshold=0.9, partial_threshold=0.5)
    result = await tier.attempt(_request(_WARMED, _EMBEDDER.embed(_WARMED)))
    assert result.outcome is TierOutcome.MISS


class _FakeFactSearch(SemanticFactSearch):
    # Scores facts with the fake repository's real cosine similarity and records
    # the scope each search ran under.
    def __init__(self, facts: list[SemanticMemory] | None = None) -> None:
        self._repository = FakeSemanticMemoryRepository()
        self._repository.set_search_results(facts or [])
        self.calls: list[tuple[uuid.UUID, uuid.UUID, int]] = []

    async def search(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
    ) -> list[ScoredFact]:
        self.calls.append((tenant_id, user_id, top_k))
        return await self._repository.search_by_similarity(
            query_embedding, user_id, tenant_id, top_k
        )


def _fact(key: str, embedding: list[float]) -> SemanticMemory:
    return SemanticMemory(
        id=uuid.uuid4(), user_id=_USER, fact_key=key, fact_value=f"{key} value", embedding=embedding
    )


def _mag_tier(search: _FakeFactSearch, top_k: int = 5) -> MagTier:
    return MagTier(search, hit_threshold=0.9, partial_threshold=0.5, top_k=top_k)


async def test_mag_tier_hits_when_the_best_fact_clears_the_hit_threshold_and_drops_weak_facts():
    strong, weak = _fact("strong", [1.0, 0.0]), _fact("weak", [0.0, 1.0])
    result = await _mag_tier(_FakeFactSearch([strong, weak])).attempt(
        _request(embedding=[1.0, 0.0])
    )
    assert result.outcome is TierOutcome.HIT
    assert [(i.paradigm, i.content, i.source_id) for i in result.items] == [
        (Paradigm.MAG, "strong: strong value", strong.id)
    ]


async def test_mag_tier_reports_partial_when_relevant_facts_fall_short_of_a_hit():
    a, b = _fact("a", [1.0, 0.0]), _fact("b", [0.0, 1.0])
    result = await _mag_tier(_FakeFactSearch([a, b])).attempt(_request(embedding=[0.8, 0.6]))
    assert result.outcome is TierOutcome.PARTIAL
    assert [item.content for item in result.items] == ["a: a value", "b: b value"]


async def test_mag_tier_misses_when_no_fact_clears_the_partial_threshold():
    search = _FakeFactSearch([_fact("a", [1.0, 0.0])])
    result = await _mag_tier(search).attempt(_request(embedding=[-1.0, 0.0]))
    assert result.outcome is TierOutcome.MISS


async def test_mag_tier_scopes_its_search_to_the_requesting_tenant_and_user():
    search = _FakeFactSearch()
    other_tenant, other_user = uuid.uuid4(), uuid.uuid4()
    await _mag_tier(search, top_k=3).attempt(_request(tenant_id=other_tenant, user_id=other_user))
    assert search.calls == [(other_tenant, other_user, 3)]


async def test_rag_tier_hits_with_every_retrieved_result():
    document_id = uuid.uuid4()
    results = [
        SearchResult(document_id=document_id, chunk_id=uuid.uuid4(), content="fresh", score=0.7)
    ]
    retriever = FakeRetriever(results)
    result = await RagTier(retriever, top_k=4).attempt(_request("what changed today"))
    assert result.outcome is TierOutcome.HIT
    assert [(i.paradigm, i.content, i.score, i.source_id) for i in result.items] == [
        (Paradigm.RAG, "fresh", 0.7, document_id)
    ]
    assert retriever.calls == [(_TENANT, "what changed today", 4)]


async def test_rag_tier_misses_when_nothing_is_retrieved():
    assert (await RagTier(FakeRetriever()).attempt(_request())).outcome is TierOutcome.MISS


def test_a_partial_threshold_above_the_hit_threshold_is_rejected():
    retriever, _, _ = _warmed_retriever()
    with pytest.raises(ValueError):
        CagTier(retriever, hit_threshold=0.5, partial_threshold=0.6)
    with pytest.raises(ValueError):
        MagTier(_FakeFactSearch(), hit_threshold=0.5, partial_threshold=0.6)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_top_k_is_rejected(bad):
    with pytest.raises(ValueError):
        RagTier(FakeRetriever(), top_k=bad)
    with pytest.raises(ValueError):
        MagTier(_FakeFactSearch(), hit_threshold=0.9, partial_threshold=0.5, top_k=bad)
