import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.orchestration.domain.entities import DataSource, IngestionRoute, SourceScope
from src.orchestration.domain.freshness_router import source_id_for
from src.orchestration.infrastructure.postgres_data_source_repository import (
    PostgresDataSourceRepository,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)


async def _create_user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id) "
            "VALUES (:id, :email, :hashed_password, :tenant_id)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id,
        },
    )
    await db_session.commit()
    return user_id


def _source(
    tenant_id: uuid.UUID, key: str = "policy", user_id: uuid.UUID | None = None, **changes
) -> DataSource:
    base = dict(
        id=source_id_for(tenant_id, key, user_id), tenant_id=tenant_id, user_id=user_id,
        source_key=key, scope=SourceScope.USER if user_id else SourceScope.TENANT,
        # A seconds component checks the interval round-trips exactly.
        expected_change_interval=7 * DAY + timedelta(seconds=90),
        route=IngestionRoute.MAG if user_id else IngestionRoute.CAG_WITH_RAG_BACKUP,
        content_hash="h1", last_changed_at=T0, last_ingested_at=T0, cached_until=None,
    )
    return DataSource(**{**base, **changes})


def _repository(db_session) -> PostgresDataSourceRepository:
    return PostgresDataSourceRepository(get_sessionmaker(db_session.bind))


async def test_a_source_round_trips_with_its_interval_and_nullable_fields(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    source = _source(tenant_id)
    await repository.save(source, changed_content="forty-five days")
    assert await repository.get(tenant_id, "policy", None) == source


async def test_changed_content_becomes_versions_and_confirmations_do_not(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    await repository.save(_source(tenant_id), changed_content="v1")
    await repository.save(_source(tenant_id, last_ingested_at=T0 + HOUR))  # confirmation
    changed = _source(
        tenant_id, content_hash="h2", last_changed_at=T0 + DAY, last_ingested_at=T0 + DAY
    )
    await repository.save(changed, changed_content="v2")

    assert await repository.version_times(tenant_id, changed.id) == [T0, T0 + DAY]
    assert await repository.current_content(tenant_id, changed.id) == "v2"


async def test_saving_again_updates_route_interval_and_cache_expiry(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    await repository.save(_source(tenant_id), changed_content="v1")
    migrated = _source(
        tenant_id, route=IngestionRoute.RAG_ONLY, expected_change_interval=HOUR,
        cached_until=T0 + 3 * DAY,
    )
    await repository.save(migrated)
    assert await repository.get(tenant_id, "policy", None) == migrated
    assert await repository.list_sources(tenant_id) == [migrated]


async def test_a_user_source_and_a_tenant_source_with_one_key_are_distinct(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    tenant_source, user_source = _source(tenant_id, "size"), _source(tenant_id, "size", user_id)
    await repository.save(tenant_source, changed_content="t")
    await repository.save(user_source, changed_content="u")

    assert await repository.get(tenant_id, "size", None) == tenant_source
    assert await repository.get(tenant_id, "size", user_id) == user_source
    assert await repository.current_content(tenant_id, user_source.id) == "u"


async def test_another_tenant_can_neither_see_list_nor_read_versions(db_session):
    repository, tenant_id, other = _repository(db_session), uuid.uuid4(), uuid.uuid4()
    source = _source(tenant_id)
    await repository.save(source, changed_content="v1")
    assert await repository.get(other, "policy", None) is None
    assert await repository.list_sources(other) == []
    assert await repository.version_times(other, source.id) == []
    assert await repository.current_content(other, source.id) is None


@pytest.mark.parametrize(
    ("scope", "with_user", "interval"),
    [("user", False, HOUR), ("tenant", True, HOUR), ("tenant", False, timedelta(0))],
)
async def test_the_schema_refuses_inconsistent_rows(db_session, scope, with_user, interval):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id) if with_user else None
    await set_tenant_context(db_session, tenant_id)
    with pytest.raises(IntegrityError):
        # asyncpg types each parameter from its column: an interval takes a timedelta.
        await db_session.execute(
            text(
                "INSERT INTO data_sources (id, tenant_id, user_id, source_key, scope, "
                "expected_change_interval, route, content_hash, last_changed_at, "
                "last_ingested_at) VALUES (:id, :tenant_id, :user_id, 'k', :scope, "
                ":interval, 'rag_only', 'h', now(), now())"
            ),
            {
                "id": uuid.uuid4(), "tenant_id": tenant_id, "user_id": user_id, "scope": scope,
                "interval": interval,
            },
        )
    await db_session.rollback()


async def test_the_schema_refuses_a_second_tenant_source_with_the_same_key(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    await repository.save(_source(tenant_id), changed_content="v1")
    duplicate = _source(tenant_id, id=uuid.uuid4())
    with pytest.raises(IntegrityError):
        await repository.save(duplicate)
