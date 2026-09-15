import threading
import uuid

import pytest

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.domain.similarity import cosine_similarity
from src.rag.domain.entities import SearchResult
from tests.unit.orchestration_fakes import FakeBagOfWordsEmbeddingModel, FakeFrozenCache
from tests.unit.rag_fakes import FakeRetriever, ThreadRecordingEmbeddingModel

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


class _CountingEmbedder(FakeBagOfWordsEmbeddingModel):
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return super().embed(text)


def _warmed() -> tuple[CacheWarmedRetrieve, FakeFrozenCache, uuid.UUID]:
    retriever, cache, _ = _build()
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)
    return retriever, cache, document_id


def test_best_warmed_match_reports_the_closest_document_with_no_threshold_applied():
    retriever, _, document_id = _warmed()
    embedder = FakeBagOfWordsEmbeddingModel()
    query_embedding = embedder.embed(_UNRELATED_QUERY)

    result = retriever.best_warmed_match(_TENANT, query_embedding)

    assert result is not None
    assert result.document_id == document_id
    assert result.content == _WARMED_CONTENT
    assert result.score == pytest.approx(
        cosine_similarity(query_embedding, embedder.embed(_WARMED_CONTENT))
    )


def test_best_warmed_match_is_scoped_to_the_tenant():
    retriever, _, _ = _warmed()
    embedding = FakeBagOfWordsEmbeddingModel().embed(_WARMED_CONTENT)
    assert retriever.best_warmed_match(uuid.uuid4(), embedding) is None


def test_best_warmed_match_ignores_a_document_evicted_from_the_frozen_cache():
    retriever, cache, document_id = _warmed()
    cache.evict(_TENANT, document_id)
    embedding = FakeBagOfWordsEmbeddingModel().embed(_WARMED_CONTENT)
    assert retriever.best_warmed_match(_TENANT, embedding) is None


class _WarmsAnotherDocumentMidScan(FakeBagOfWordsEmbeddingModel):
    """For the warmed text, returns a vector that warms a second document the
    first time it is read -- what note_warmed on the event loop does to a scan
    running in CagTier's worker thread."""

    def __init__(self) -> None:
        self.retriever: CacheWarmedRetrieve | None = None
        self.fired = False

    def embed(self, text: str) -> list[float]:
        vector = super().embed(text)
        if text != _WARMED_CONTENT:
            return vector
        embedder = self

        class _Vector(list):
            def __iter__(inner):
                if embedder.retriever is not None and not embedder.fired:
                    embedder.fired = True
                    embedder.retriever.note_warmed(_TENANT, uuid.uuid4(), "another warmed document")
                return super().__iter__()

        return _Vector(vector)


def test_best_warmed_match_survives_a_document_being_warmed_during_its_scan():
    embedder = _WarmsAnotherDocumentMidScan()
    cache = FakeFrozenCache()
    retriever = CacheWarmedRetrieve(embedder, cache, FakeRetriever(), _THRESHOLD)
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)
    embedder.retriever = retriever

    result = retriever.best_warmed_match(_TENANT, embedder.embed(_MATCHING_QUERY))

    assert embedder.fired is True
    assert result is not None
    assert result.document_id == document_id


async def test_execute_does_not_embed_the_query_when_nothing_is_warmed_for_the_tenant():
    embedder = _CountingEmbedder()
    retriever = CacheWarmedRetrieve(embedder, FakeFrozenCache(), FakeRetriever(), _THRESHOLD)
    await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)
    assert embedder.calls == 0


def test_an_evicted_best_candidate_does_not_shadow_a_confirmed_runner_up():
    retriever, cache, _ = _build()
    embedder = FakeBagOfWordsEmbeddingModel()
    query = embedder.embed(_MATCHING_QUERY)
    evicted, valid = uuid.uuid4(), uuid.uuid4()
    other = "the policy desk opens at nine"
    # Precondition, checked rather than assumed: the evicted document is the best candidate.
    assert cosine_similarity(query, embedder.embed(_WARMED_CONTENT)) > cosine_similarity(
        query, embedder.embed(other)
    )
    for document_id, content in ((evicted, _WARMED_CONTENT), (valid, other)):
        cache.preload(_TENANT, document_id, content)
        retriever.note_warmed(_TENANT, document_id, content)
    cache.evict(_TENANT, evicted)  # evicted, expired, or re-preloaded: nobody called forget

    result = retriever.best_warmed_match(_TENANT, query)

    assert result is not None
    assert result.document_id == valid
    assert result.score == cosine_similarity(query, embedder.embed(other))


def test_a_forgotten_document_is_not_matched_even_when_its_text_is_cached_again():
    retriever, cache, _ = _build()
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)

    retriever.forget(_TENANT, document_id)
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)

    embedding = FakeBagOfWordsEmbeddingModel().embed(_MATCHING_QUERY)
    assert retriever.best_warmed_match(_TENANT, embedding) is None


def test_forgetting_something_never_warmed_is_a_no_op():
    retriever, _, _ = _build()
    retriever.forget(_TENANT, uuid.uuid4())
    retriever.forget(uuid.uuid4(), uuid.uuid4())


async def test_embedding_the_query_and_matching_it_run_off_the_event_loop():
    # Both are CPU work. Behind RagTier they would otherwise stall the other cascade
    # tiers sharing the loop in a PARALLEL route.
    embedder = ThreadRecordingEmbeddingModel(FakeBagOfWordsEmbeddingModel())
    cache = FakeFrozenCache()
    retriever = CacheWarmedRetrieve(embedder, cache, FakeRetriever(), _THRESHOLD)
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)
    embedded_before_query = len(embedder.thread_ids)
    matched_on: list[int] = []
    best_warmed_match = retriever.best_warmed_match

    def recording_match(tenant_id, query_embedding):
        matched_on.append(threading.get_ident())
        return best_warmed_match(tenant_id, query_embedding)

    retriever.best_warmed_match = recording_match

    results = await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)

    assert results[0].document_id == document_id
    query_embeds = embedder.thread_ids[embedded_before_query:]
    assert len(query_embeds) == 1
    assert len(matched_on) == 1
    assert threading.get_ident() not in query_embeds + matched_on
