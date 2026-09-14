import uuid
from datetime import UTC, datetime, timedelta

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.refresh_cached_sources import RefreshCachedSources
from src.orchestration.domain.entities import (
    DataSource,
    FreshnessPolicy,
    IngestionRoute,
    SourceScope,
)
from tests.unit.freshness_fakes import FakeDataSourceRepository, FakeExpiringCache
from tests.unit.orchestration_fakes import FakeBagOfWordsEmbeddingModel
from tests.unit.rag_fakes import FakeRetriever

T0 = datetime(2026, 1, 20, tzinfo=UTC)
DAY = timedelta(days=1)
TENANT = uuid.uuid4()


def _fixtures():
    repository, cache = FakeDataSourceRepository(), FakeExpiringCache()
    warmed = CacheWarmedRetrieve(FakeBagOfWordsEmbeddingModel(), cache, FakeRetriever(), 0.3)
    return repository, cache, warmed


async def _seed(
    repository: FakeDataSourceRepository,
    key: str,
    route: IngestionRoute,
    versions: list[tuple[datetime, str]],
    *,
    interval: timedelta = 7 * DAY,
    last_ingested_at: datetime | None = None,
    scope: SourceScope = SourceScope.TENANT,
    tenant_id: uuid.UUID = TENANT,
) -> DataSource:
    user_id = uuid.uuid4() if scope is SourceScope.USER else None
    source = None
    for at, text in versions:
        source = DataSource(
            id=uuid.uuid5(uuid.NAMESPACE_OID, f"{tenant_id}:{key}"), tenant_id=tenant_id,
            user_id=user_id, source_key=key, scope=scope, expected_change_interval=interval,
            route=route, content_hash=text, last_changed_at=at,
            last_ingested_at=last_ingested_at or at,
        )
        await repository.save(source, changed_content=text)
    assert source is not None
    return source


async def test_an_uncached_confirmed_source_is_preloaded_until_one_ttl_after_confirmation():
    repository, cache, warmed = _fixtures()
    source = await _seed(
        repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP,
        [(T0 - 10 * DAY, "forty-five days")], last_ingested_at=T0 - DAY,
    )

    preloaded = await RefreshCachedSources(repository, cache, warmed).run(TENANT, T0)

    expires_at = T0 - DAY + timedelta(days=3.5)
    assert preloaded == ["policy"]
    assert cache.preload_calls == [(TENANT, source.id, "forty-five days", expires_at)]
    embedding = FakeBagOfWordsEmbeddingModel().embed("forty-five days")
    match = warmed.best_warmed_match(TENANT, embedding)
    assert match is not None
    assert match.document_id == source.id
    stored = await repository.get(TENANT, "policy", None)
    assert stored is not None
    assert stored.cached_until == expires_at


async def test_an_already_cached_source_is_not_preloaded_again():
    repository, cache, warmed = _fixtures()
    source = await _seed(repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP, [(T0, "v1")])
    cache.preload_until(TENANT, source.id, "v1", None)
    cache.preload_calls.clear()

    assert await RefreshCachedSources(repository, cache, warmed).run(TENANT, T0) == []
    assert cache.preload_calls == []


async def test_a_source_unconfirmed_for_longer_than_its_ttl_stays_on_rag():
    repository, cache, warmed = _fixtures()
    await _seed(
        repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP, [(T0 - 10 * DAY, "v1")],
        last_ingested_at=T0 - timedelta(days=3.5),  # exactly one TTL ago: expired
    )
    assert await RefreshCachedSources(repository, cache, warmed).run(TENANT, T0) == []


async def test_rag_only_and_mag_sources_are_never_preloaded():
    repository, cache, warmed = _fixtures()
    await _seed(repository, "prices", IngestionRoute.RAG_ONLY, [(T0, "v1")])
    await _seed(repository, "size", IngestionRoute.MAG, [(T0, "v1")], scope=SourceScope.USER)
    assert await RefreshCachedSources(repository, cache, warmed).run(TENANT, T0) == []


async def test_with_ttl_disabled_age_never_blocks_a_preload():
    repository, cache, warmed = _fixtures()
    source = await _seed(
        repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP, [(T0 - 400 * DAY, "v1")]
    )
    refresh = RefreshCachedSources(
        repository, cache, warmed, policy=FreshnessPolicy(ttl_factor=None)
    )
    assert await refresh.run(TENANT, T0) == ["policy"]
    assert cache.preload_calls == [(TENANT, source.id, "v1", None)]


async def test_another_tenants_sources_are_untouched():
    repository, cache, warmed = _fixtures()
    await _seed(
        repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP, [(T0, "v1")],
        tenant_id=uuid.uuid4(),
    )
    assert await RefreshCachedSources(repository, cache, warmed).run(TENANT, T0) == []
