import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.ingest_data_source import IngestDataSource
from src.orchestration.domain.entities import (
    DataSourceProfile,
    FreshnessPolicy,
    IngestionRoute,
    SourceScope,
)
from src.orchestration.domain.errors import ScopeMismatch
from src.orchestration.domain.freshness_router import source_id_for
from tests.unit.freshness_fakes import (
    FakeDataSourceRepository,
    FakeExpiringCache,
    FakeRagIndex,
    FakeSessionFactWriter,
)
from tests.unit.orchestration_fakes import FakeBagOfWordsEmbeddingModel
from tests.unit.rag_fakes import FakeRetriever

T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)
TENANT, USER = uuid.uuid4(), uuid.uuid4()
PRICES = DataSourceProfile("prices", SourceScope.TENANT, HOUR)
POLICY_DOC = DataSourceProfile("return-policy", SourceScope.TENANT, 7 * DAY)
SIZE = DataSourceProfile("size", SourceScope.USER, 30 * DAY)


def _build(policy: FreshnessPolicy | None = None, route_policy=None):
    repository, rag, cache = FakeDataSourceRepository(), FakeRagIndex(), FakeExpiringCache()
    writer = FakeSessionFactWriter()
    warmed = CacheWarmedRetrieve(FakeBagOfWordsEmbeddingModel(), cache, FakeRetriever(), 0.3)
    kwargs = {"policy": policy}
    if route_policy is not None:
        kwargs["route_policy"] = route_policy
    use_case = IngestDataSource(repository, rag, cache, warmed, writer, **kwargs)
    return use_case, repository, rag, cache, warmed, writer


async def test_a_new_volatile_source_goes_to_rag_only_under_a_deterministic_id():
    use_case, repository, rag, cache, _, writer = _build()

    result = await use_case.execute(TENANT, PRICES, "costs 41 dollars", T0)

    source_id = source_id_for(TENANT, "prices", None)
    assert (result.source_id, result.route, result.changed) == (
        source_id, IngestionRoute.RAG_ONLY, True,
    )
    assert rag.replaced == [(TENANT, source_id, "prices", "costs 41 dollars")]
    assert cache.preload_calls == []
    assert writer.records == []
    assert await repository.version_times(TENANT, source_id) == [T0]


async def test_a_new_stable_source_is_indexed_in_rag_and_left_for_the_batch_to_cache():
    use_case, repository, rag, cache, _, _ = _build()

    result = await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0)

    assert result.route is IngestionRoute.CAG_WITH_RAG_BACKUP
    assert rag.replaced == [(TENANT, result.source_id, "return-policy", "forty-five days")]
    assert cache.preload_calls == []
    stored = await repository.get(TENANT, "return-policy", None)
    assert stored is not None
    assert stored.cached_until is None


async def test_a_user_scoped_source_is_written_to_mag_and_nowhere_else():
    use_case, _, rag, cache, _, writer = _build()

    result = await use_case.execute(TENANT, SIZE, "wears size 10", T0, user_id=USER)

    assert result.route is IngestionRoute.MAG
    assert writer.records == [(TENANT, USER, "size", "wears size 10")]
    assert rag.replaced == []
    assert cache.preload_calls == []


async def test_identical_content_is_a_confirmation_not_a_change():
    use_case, repository, rag, _, _, _ = _build()
    await use_case.execute(TENANT, PRICES, "costs 41 dollars", T0)

    result = await use_case.execute(TENANT, PRICES, "costs 41 dollars", T0 + HOUR)

    assert result.changed is False
    assert len(rag.replaced) == 1
    stored = await repository.get(TENANT, "prices", None)
    assert stored is not None
    assert (stored.last_changed_at, stored.last_ingested_at) == (T0, T0 + HOUR)
    assert await repository.version_times(TENANT, result.source_id) == [T0]


async def test_a_change_replaces_the_rag_text_and_records_a_version():
    use_case, repository, rag, _, _, _ = _build()
    await use_case.execute(TENANT, PRICES, "costs 41 dollars", T0)

    result = await use_case.execute(TENANT, PRICES, "costs 43 dollars", T0 + HOUR)

    assert result.changed is True
    assert rag.replaced[-1][3] == "costs 43 dollars"
    assert await repository.version_times(TENANT, result.source_id) == [T0, T0 + HOUR]
    assert await repository.current_content(TENANT, result.source_id) == "costs 43 dollars"


async def test_a_change_to_a_cached_source_evicts_and_forgets_it_at_once():
    use_case, repository, _, cache, warmed, _ = _build()
    first = await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0)
    cache.preload_until(TENANT, first.source_id, "forty-five days", T0 + DAY)
    warmed.note_warmed(TENANT, first.source_id, "forty-five days")

    await use_case.execute(TENANT, POLICY_DOC, "sixty days", T0 + HOUR)

    assert (TENANT, first.source_id) in cache.evict_calls
    # Re-preloading the old text proves the memo is gone: nothing matches it now.
    cache.preload(TENANT, first.source_id, "forty-five days")
    embedding = FakeBagOfWordsEmbeddingModel().embed("forty-five days")
    assert warmed.best_warmed_match(TENANT, embedding) is None
    stored = await repository.get(TENANT, "return-policy", None)
    assert stored is not None
    assert stored.cached_until is None


async def test_a_confirmation_renews_a_cached_entry_for_one_ttl_from_now():
    use_case, repository, _, cache, _, _ = _build()
    first = await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0)
    cache.preload_until(TENANT, first.source_id, "forty-five days", T0 + DAY)

    await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0 + 2 * DAY)

    expected = T0 + 2 * DAY + timedelta(days=3.5)  # 7-day interval x 0.5
    assert cache.expiry(TENANT, first.source_id) == expected
    stored = await repository.get(TENANT, "return-policy", None)
    assert stored is not None
    assert stored.cached_until == expected


async def test_a_confirmation_of_an_uncached_source_caches_nothing():
    use_case, repository, _, cache, _, _ = _build()
    await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0)

    await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0 + DAY)

    assert cache.preload_calls == []
    assert cache.renew_calls == []
    stored = await repository.get(TENANT, "return-policy", None)
    assert stored is not None
    assert stored.cached_until is None


async def test_with_ttl_disabled_a_confirmation_renews_with_no_expiry():
    use_case, _, _, cache, _, _ = _build(policy=FreshnessPolicy(ttl_factor=None))
    first = await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0)
    cache.preload_until(TENANT, first.source_id, "forty-five days", T0 + DAY)

    await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0 + HOUR)

    assert cache.renew_calls == [(TENANT, first.source_id, None)]


@pytest.mark.parametrize(("profile", "user_id"), [(SIZE, None), (PRICES, USER)])
async def test_scope_and_user_id_must_agree(profile, user_id):
    use_case, repository, rag, _, _, writer = _build()
    with pytest.raises(ScopeMismatch):
        await use_case.execute(TENANT, profile, "text", T0, user_id=user_id)
    assert await repository.list_sources(TENANT) == []
    assert (rag.replaced, writer.records) == ([], [])


async def test_an_injected_route_policy_can_cache_a_user_scoped_source():
    # The measurement's "cache everything" baseline: the same code path, one route.
    use_case, _, rag, _, _, writer = _build(
        route_policy=lambda profile, policy: IngestionRoute.CAG_WITH_RAG_BACKUP
    )
    result = await use_case.execute(TENANT, SIZE, "wears size 10", T0, user_id=USER)
    assert result.route is IngestionRoute.CAG_WITH_RAG_BACKUP
    assert rag.replaced == [(TENANT, result.source_id, "size", "wears size 10")]
    assert writer.records == []


async def test_a_route_policy_cannot_send_tenant_data_to_mag():
    use_case, _, _, _, _, _ = _build(route_policy=lambda profile, policy: IngestionRoute.MAG)
    with pytest.raises(ValueError):
        await use_case.execute(TENANT, PRICES, "text", T0)


async def test_later_ingestions_keep_the_stored_interval():
    use_case, repository, _, _, _, _ = _build()
    await use_case.execute(TENANT, PRICES, "v1", T0)
    redeclared = DataSourceProfile("prices", SourceScope.TENANT, 30 * DAY)

    result = await use_case.execute(TENANT, redeclared, "v2", T0 + HOUR)

    assert result.route is IngestionRoute.RAG_ONLY
    stored = await repository.get(TENANT, "prices", None)
    assert stored is not None
    assert stored.expected_change_interval == HOUR


async def test_the_same_key_in_two_tenants_is_two_sources():
    use_case, repository, _, _, _, _ = _build()
    other = uuid.uuid4()
    a = await use_case.execute(TENANT, PRICES, "v1", T0)
    b = await use_case.execute(other, PRICES, "v1", T0)
    assert a.source_id != b.source_id
    assert len(await repository.list_sources(TENANT)) == 1
