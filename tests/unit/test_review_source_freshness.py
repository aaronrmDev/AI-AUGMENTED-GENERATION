import uuid
from datetime import UTC, datetime, timedelta

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.review_source_freshness import ReviewSourceFreshness
from src.orchestration.domain.entities import (
    DataSource,
    IngestionRoute,
    SourceMigration,
    SourceScope,
)
from tests.unit.freshness_fakes import FakeDataSourceRepository, FakeExpiringCache
from tests.unit.orchestration_fakes import FakeBagOfWordsEmbeddingModel
from tests.unit.rag_fakes import FakeRetriever

T0 = datetime(2026, 1, 20, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)
TENANT = uuid.uuid4()
CAG, RAG_ONLY = IngestionRoute.CAG_WITH_RAG_BACKUP, IngestionRoute.RAG_ONLY


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
    scope: SourceScope = SourceScope.TENANT,
) -> DataSource:
    user_id = uuid.uuid4() if scope is SourceScope.USER else None
    source = None
    for at, text in versions:
        source = DataSource(
            id=uuid.uuid5(uuid.NAMESPACE_OID, f"{TENANT}:{key}"), tenant_id=TENANT,
            user_id=user_id, source_key=key, scope=scope, expected_change_interval=interval,
            route=route, content_hash=text, last_changed_at=at, last_ingested_at=at,
        )
        await repository.save(source, changed_content=text)
    assert source is not None
    return source


async def test_a_cached_source_changing_fast_is_demoted_evicted_and_forgotten():
    repository, cache, warmed = _fixtures()
    start = T0 - 4 * HOUR
    source = await _seed(
        repository, "catalog", CAG, [(start + i * HOUR, f"v{i}") for i in range(4)]
    )
    cache.preload_until(TENANT, source.id, "v3", None)
    warmed.note_warmed(TENANT, source.id, "v3")

    migrations = await ReviewSourceFreshness(repository, cache, warmed).run(TENANT, T0)

    assert migrations == [SourceMigration("catalog", CAG, RAG_ONLY, HOUR)]
    assert (TENANT, source.id) in cache.evict_calls
    assert warmed.best_warmed_match(TENANT, FakeBagOfWordsEmbeddingModel().embed("v3")) is None
    stored = await repository.get(TENANT, "catalog", None)
    assert stored is not None
    assert (stored.route, stored.expected_change_interval, stored.cached_until) == (
        RAG_ONLY, HOUR, None,
    )


async def test_a_quiet_rag_only_source_is_promoted_without_a_preload():
    repository, cache, warmed = _fixtures()
    await _seed(repository, "sale", RAG_ONLY, [(T0 - 8 * DAY, "v1")], interval=HOUR)

    migrations = await ReviewSourceFreshness(repository, cache, warmed).run(TENANT, T0)

    assert migrations == [SourceMigration("sale", RAG_ONLY, CAG, 8 * DAY)]
    assert cache.preload_calls == []
    stored = await repository.get(TENANT, "sale", None)
    assert stored is not None
    assert (stored.route, stored.expected_change_interval) == (CAG, 8 * DAY)


async def test_user_scoped_sources_are_never_reviewed():
    repository, cache, warmed = _fixtures()
    start = T0 - 4 * HOUR
    await _seed(
        repository, "size", CAG, [(start + i * HOUR, f"v{i}") for i in range(4)],
        scope=SourceScope.USER,
    )
    assert await ReviewSourceFreshness(repository, cache, warmed).run(TENANT, T0) == []


async def test_a_source_with_no_migration_is_left_exactly_as_stored():
    repository, cache, warmed = _fixtures()
    source = await _seed(repository, "policy", CAG, [(T0 - DAY, "v1")])
    assert await ReviewSourceFreshness(repository, cache, warmed).run(TENANT, T0) == []
    assert await repository.get(TENANT, "policy", None) == source
    assert cache.evict_calls == []
