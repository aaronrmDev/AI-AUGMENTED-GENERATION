import uuid

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.rag.domain.entities import SearchResult
from tests.unit.orchestration_fakes import FakeBagOfWordsEmbeddingModel, FakeFrozenCache
from tests.unit.rag_fakes import FakeRetriever

_THRESHOLD = 0.3
_TENANT = uuid.uuid4()
_WARMED_CONTENT = "return policy allows customers to return unopened items within thirty days"
_MATCHING_QUERY = "what is the return policy for unopened items"
_UNRELATED_QUERY = "quarterly financial earnings report for shareholders"


def _build(cache: FakeFrozenCache | None = None, fallback: FakeRetriever | None = None):
    embedder = FakeBagOfWordsEmbeddingModel()
    cache = cache or FakeFrozenCache()
    fallback = fallback or FakeRetriever()
    retriever = CacheWarmedRetrieve(embedder, cache, fallback, _THRESHOLD)
    return retriever, cache, fallback


async def test_a_close_match_against_a_warmed_document_is_a_hit_with_no_fallback_call():
    retriever, cache, fallback = _build()
    document_id = uuid.uuid4()
    cache.preload(document_id, _WARMED_CONTENT)
    retriever.note_warmed(document_id, _WARMED_CONTENT)

    results = await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)

    assert len(results) == 1
    assert results[0].document_id == document_id
    assert results[0].content == _WARMED_CONTENT
    assert fallback.calls == []
    assert retriever.stats() == (1, 0)


async def test_nothing_warmed_yet_falls_through_to_the_fallback():
    fallback_results = [
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="x", score=0.5)
    ]
    retriever, cache, fallback = _build(fallback=FakeRetriever(fallback_results))

    results = await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)

    assert results == fallback_results
    assert fallback.calls == [(_TENANT, _MATCHING_QUERY, 5)]
    assert retriever.stats() == (0, 1)


async def test_an_unrelated_query_against_a_warmed_document_falls_through_to_the_fallback():
    retriever, cache, fallback = _build()
    document_id = uuid.uuid4()
    cache.preload(document_id, _WARMED_CONTENT)
    retriever.note_warmed(document_id, _WARMED_CONTENT)

    await retriever.execute(_TENANT, _UNRELATED_QUERY, top_k=5)

    assert fallback.calls == [(_TENANT, _UNRELATED_QUERY, 5)]
    assert retriever.stats() == (0, 1)


async def test_a_document_demoted_since_indexing_falls_through_rather_than_serving_stale_content():
    retriever, cache, fallback = _build()
    document_id = uuid.uuid4()
    cache.preload(document_id, _WARMED_CONTENT)
    retriever.note_warmed(document_id, _WARMED_CONTENT)
    cache.evict(document_id)  # simulates TieringPolicy/SyncCycle demoting it since indexing

    await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)

    assert fallback.calls == [(_TENANT, _MATCHING_QUERY, 5)]
    assert retriever.stats() == (0, 1)


async def test_stats_accumulate_correctly_across_a_mixed_sequence():
    retriever, cache, fallback = _build()
    document_id = uuid.uuid4()
    cache.preload(document_id, _WARMED_CONTENT)
    retriever.note_warmed(document_id, _WARMED_CONTENT)

    await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)  # hit
    await retriever.execute(_TENANT, _UNRELATED_QUERY, top_k=5)  # miss
    await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)  # hit

    assert retriever.stats() == (2, 1)
