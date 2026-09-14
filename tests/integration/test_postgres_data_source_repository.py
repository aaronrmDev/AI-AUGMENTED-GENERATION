import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.orchestration.domain.entities import (
    DataSource,
    IngestionRoute,
    SourceScope,
    SourceVersion,
)
from src.orchestration.domain.freshness_router import source_id_for
from src.orchestration.infrastructure.postgres_data_source_repository import (
    PostgresDataSourceRepository,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)
CACHED, RAG_ONLY = IngestionRoute.CAG_WITH_RAG_BACKUP, IngestionRoute.RAG_ONLY


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
        route=IngestionRoute.MAG if user_id else CACHED,
        content_hash="h1", last_changed_at=T0, last_ingested_at=T0, cached_until=None,
    )
    return DataSource(**{**base, **changes})


def _repository(db_session) -> PostgresDataSourceRepository:
    return PostgresDataSourceRepository(get_sessionmaker(db_session.bind))


async def test_a_source_round_trips_with_its_interval_and_nullable_fields(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    source = _source(tenant_id)
    await repository.save(source, SourceVersion("forty-five days"))
    assert await repository.get(tenant_id, "policy", None) == source


async def test_changed_versions_are_recorded_and_confirmations_are_not(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    await repository.save(_source(tenant_id), SourceVersion("v1"))
    await repository.save(_source(tenant_id, last_ingested_at=T0 + HOUR))  # confirmation
    changed = _source(
        tenant_id, content_hash="h2", last_changed_at=T0 + DAY, last_ingested_at=T0 + DAY
    )
    await repository.save(changed, SourceVersion("v2"))

    assert await repository.version_times(tenant_id, changed.id) == [T0, T0 + DAY]
    assert await repository.current_content(tenant_id, changed.id) == "v2"


async def test_a_later_save_keeps_route_and_interval_and_only_migrate_changes_them(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    source = _source(tenant_id)
    await repository.save(source, SourceVersion("v1"))

    # A stale snapshot carrying another route and interval must not overwrite them.
    await repository.save(
        _source(tenant_id, route=RAG_ONLY, expected_change_interval=HOUR, cached_until=T0 + DAY)
    )
    stored = await repository.get(tenant_id, "policy", None)
    assert stored is not None
    assert (stored.route, stored.expected_change_interval, stored.cached_until) == (
        CACHED, source.expected_change_interval, T0 + DAY,
    )

    assert await repository.migrate(tenant_id, source.id, "h1", RAG_ONLY, HOUR, None) is True
    migrated = await repository.get(tenant_id, "policy", None)
    assert migrated is not None
    assert (migrated.route, migrated.expected_change_interval, migrated.cached_until) == (
        RAG_ONLY, HOUR, None,
    )
    assert await repository.list_sources(tenant_id) == [migrated]


async def test_conditional_writes_apply_only_while_the_source_is_unchanged(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    source = _source(tenant_id)
    await repository.save(source, SourceVersion("v1"))

    assert await repository.record_cached_until(tenant_id, source.id, "h1", T0 + DAY) is True
    assert await repository.record_cached_until(tenant_id, source.id, "stale", T0) is False
    assert await repository.record_cached_until(uuid.uuid4(), source.id, "h1", T0) is False

    await repository.mark_pending(tenant_id, source.id, "h2")
    pending = await repository.get(tenant_id, "policy", None)
    assert pending is not None
    assert (pending.pending_hash, pending.cached_until) == ("h2", T0 + DAY)
    assert await repository.record_cached_until(tenant_id, source.id, "h1", None) is False
    assert await repository.migrate(tenant_id, source.id, "h1", RAG_ONLY, HOUR, None) is False

    await repository.save(source)  # the save that records the change clears the marker
    cleared = await repository.get(tenant_id, "policy", None)
    assert cleared is not None
    assert cleared.pending_hash is None
    assert await repository.migrate(tenant_id, source.id, "h1", RAG_ONLY, HOUR, None) is True


async def test_a_version_without_text_is_recorded_but_never_returned_as_content(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    source = _source(tenant_id, "size", user_id)
    await repository.save(source, SourceVersion(None))

    assert await repository.version_times(tenant_id, source.id) == [T0]
    assert await repository.current_content(tenant_id, source.id) is None
    await set_tenant_context(db_session, tenant_id)
    stored_text = (
        await db_session.execute(
            text("SELECT content FROM data_source_versions WHERE data_source_id = :id"),
            {"id": source.id},
        )
    ).scalars().all()
    assert stored_text == [None]


async def test_versions_under_one_timestamp_keep_their_insertion_order(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    await repository.save(_source(tenant_id), SourceVersion("v1"))
    second = _source(tenant_id, content_hash="h2")  # the same last_changed_at as v1
    await repository.save(second, SourceVersion("v2"))

    assert await repository.version_times(tenant_id, second.id) == [T0, T0]
    assert await repository.current_content(tenant_id, second.id) == "v2"


async def test_a_user_source_and_a_tenant_source_with_one_key_are_distinct(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    tenant_source, user_source = _source(tenant_id, "size"), _source(tenant_id, "size", user_id)
    await repository.save(tenant_source, SourceVersion("t"))
    await repository.save(user_source, SourceVersion("u"))

    assert await repository.get(tenant_id, "size", None) == tenant_source
    assert await repository.get(tenant_id, "size", user_id) == user_source
    assert await repository.current_content(tenant_id, user_source.id) == "u"


async def test_deleting_a_user_removes_their_sources_and_versions(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    source = _source(tenant_id, "size", user_id)
    await repository.save(source, SourceVersion(None))

    await db_session.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})
    await db_session.commit()

    assert await repository.get(tenant_id, "size", user_id) is None
    assert await repository.version_times(tenant_id, source.id) == []


async def test_another_tenant_can_neither_see_list_nor_read_versions(db_session):
    repository, tenant_id, other = _repository(db_session), uuid.uuid4(), uuid.uuid4()
    source = _source(tenant_id)
    await repository.save(source, SourceVersion("v1"))
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
    await repository.save(_source(tenant_id), SourceVersion("v1"))
    duplicate = _source(tenant_id, id=uuid.uuid4())
    with pytest.raises(IntegrityError):
        await repository.save(duplicate)
