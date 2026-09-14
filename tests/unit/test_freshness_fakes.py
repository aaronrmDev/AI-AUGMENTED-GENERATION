import uuid
from datetime import UTC, datetime, timedelta

from src.orchestration.domain.entities import DataSource, IngestionRoute, SourceScope
from tests.unit.freshness_fakes import FakeDataSourceRepository, FakeExpiringCache

T0 = datetime(2026, 1, 1, tzinfo=UTC)
TENANT = uuid.uuid4()


def _source(**changes) -> DataSource:
    base = DataSource(
        id=uuid.uuid4(), tenant_id=TENANT, user_id=None, source_key="k",
        scope=SourceScope.TENANT, expected_change_interval=timedelta(days=1),
        route=IngestionRoute.RAG_ONLY, content_hash="h1", last_changed_at=T0, last_ingested_at=T0,
    )
    return DataSource(**{**base.__dict__, **changes})


async def test_the_fake_repository_records_versions_only_for_changed_content():
    repository = FakeDataSourceRepository()
    source = _source()
    await repository.save(source, changed_content="v1")
    await repository.save(source)  # a confirmation: no new version
    later = _source(id=source.id, content_hash="h2", last_changed_at=T0 + timedelta(hours=1))
    await repository.save(later, changed_content="v2")

    assert await repository.version_times(TENANT, source.id) == [T0, T0 + timedelta(hours=1)]
    assert await repository.current_content(TENANT, source.id) == "v2"
    assert await repository.get(TENANT, "k", None) == later
    assert await repository.version_times(uuid.uuid4(), source.id) == []
    assert await repository.list_sources(uuid.uuid4()) == []


def test_the_fake_cache_renews_only_live_entries():
    cache = FakeExpiringCache()
    document = uuid.uuid4()
    assert cache.renew(TENANT, document, T0) is False
    cache.preload_until(TENANT, document, "text", T0)
    assert cache.renew(TENANT, document, T0 + timedelta(days=1)) is True
    assert cache.expiry(TENANT, document) == T0 + timedelta(days=1)
