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
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)

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
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)

    await retriever.execute(_TENANT, _UNRELATED_QUERY, top_k=5)

    assert fallback.calls == [(_TENANT, _UNRELATED_QUERY, 5)]
    assert retriever.stats() == (0, 1)


async def test_a_document_demoted_since_indexing_falls_through_rather_than_serving_stale_content():
    retriever, cache, fallback = _build()
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)
    cache.evict(_TENANT, document_id)  # simulates TieringPolicy/SyncCycle demoting it

    await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)

    assert fallback.calls == [(_TENANT, _MATCHING_QUERY, 5)]
    assert retriever.stats() == (0, 1)


async def test_a_document_repromoted_with_different_content_does_not_serve_stale_content():
    # A review finding caught the original implementation trusting its own
    # locally memoized content on ANY non-None FrozenCache.lookup, so a
    # document evicted and later re-preloaded with DIFFERENT content --
    # exactly how TieringPolicy/WarmCache really preload, bypassing
    # note_warmed entirely -- would be served from stale local text on
    # what the code itself recorded as a confirmed hit.
    retriever, cache, fallback = _build()
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)
    cache.evict(_TENANT, document_id)
    new_content = _WARMED_CONTENT.replace("thirty days", "forty-five days")
    cache.preload(_TENANT, document_id, new_content)  # bypasses note_warmed, like TieringPolicy

    await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)

    assert fallback.calls == [(_TENANT, _MATCHING_QUERY, 5)]
    assert retriever.stats() == (0, 1)


async def test_a_warmed_document_for_one_tenant_is_never_served_to_another():
    # A review finding caught the original implementation ignoring
    # tenant_id entirely on the hit path -- a shared instance across
    # tenants (this project's own established singleton-service DI shape)
    # could match one tenant's query against another tenant's warmed
    # content, a cross-tenant data leak.
    retriever, cache, fallback = _build()
    document_id = uuid.uuid4()
    other_tenant = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)

    await retriever.execute(other_tenant, _MATCHING_QUERY, top_k=5)

    assert fallback.calls == [(other_tenant, _MATCHING_QUERY, 5)]
    assert retriever.stats() == (0, 1)


async def test_stats_accumulate_correctly_across_a_mixed_sequence():
    retriever, cache, fallback = _build()
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)

    await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)  # hit
    await retriever.execute(_TENANT, _UNRELATED_QUERY, top_k=5)  # miss
    await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)  # hit

    assert retriever.stats() == (2, 1)
