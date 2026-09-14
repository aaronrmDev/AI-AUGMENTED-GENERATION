import uuid
from datetime import datetime, timedelta
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.infrastructure.db import set_tenant_context
from src.orchestration.domain.entities import (
    DataSource,
    IngestionRoute,
    SourceScope,
    SourceVersion,
)
from src.orchestration.domain.ports import DataSourceRepository

_COLUMNS = (
    "id, tenant_id, user_id, source_key, scope, expected_change_interval, route, "
    "content_hash, last_changed_at, last_ingested_at, cached_until, pending_content_hash"
)
# A conditional write applies only to a source that hasn't changed since its caller read it.
_UNCHANGED = (
    "WHERE id = :id AND tenant_id = :tenant_id AND content_hash = :expected_hash "
    "AND pending_content_hash IS NULL"
)


def _to_source(row: Any) -> DataSource:
    return DataSource(
        id=row.id,
        tenant_id=row.tenant_id,
        user_id=row.user_id,
        source_key=row.source_key,
        scope=SourceScope(row.scope),
        expected_change_interval=row.expected_change_interval,
        route=IngestionRoute(row.route),
        content_hash=row.content_hash,
        last_changed_at=row.last_changed_at,
        last_ingested_at=row.last_ingested_at,
        cached_until=row.cached_until,
        pending_hash=row.pending_content_hash,
    )


def _matched_one(result: Any) -> bool:
    return cast(CursorResult[Any], result).rowcount == 1


class PostgresDataSourceRepository(DataSourceRepository):
    """data_sources and data_source_versions, under tenant_isolation RLS.

    Each call opens and commits its own short transaction and sets the tenant context
    inside it, as PostgresSessionBudgetRecorder does. Queries also filter on tenant_id
    explicitly, so a mistake in context-setting can't widen a read.

    Writes are column-targeted because ingestion, refresh, and review can run
    concurrently:
    - save, ingestion's write, never rewrites an existing source's route or interval;
    - record_cached_until and migrate are conditional on the content hash their caller
      listed, and on no change being pending, so neither can overwrite a newer version
      with a stale snapshot.
    """

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def get(
        self, tenant_id: uuid.UUID, source_key: str, user_id: uuid.UUID | None
    ) -> DataSource | None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            row = (
                await session.execute(
                    text(
                        f"SELECT {_COLUMNS} FROM data_sources WHERE tenant_id = :tenant_id "
                        "AND source_key = :source_key "
                        "AND user_id IS NOT DISTINCT FROM CAST(:user_id AS uuid)"
                    ),
                    {"tenant_id": tenant_id, "source_key": source_key, "user_id": user_id},
                )
            ).first()
        return None if row is None else _to_source(row)

    async def save(self, source: DataSource, version: SourceVersion | None = None) -> None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, source.tenant_id)
            await session.execute(
                text(
                    f"INSERT INTO data_sources ({_COLUMNS}) VALUES (:id, :tenant_id, :user_id, "
                    ":source_key, :scope, :expected_change_interval, :route, :content_hash, "
                    ":last_changed_at, :last_ingested_at, :cached_until, :pending_hash) "
                    "ON CONFLICT (id) DO UPDATE SET content_hash = EXCLUDED.content_hash, "
                    "last_changed_at = EXCLUDED.last_changed_at, "
                    "last_ingested_at = EXCLUDED.last_ingested_at, "
                    "cached_until = EXCLUDED.cached_until, "
                    "pending_content_hash = EXCLUDED.pending_content_hash, updated_at = now()"
                ),
                {
                    "id": source.id,
                    "tenant_id": source.tenant_id,
                    "user_id": source.user_id,
                    "source_key": source.source_key,
                    "scope": source.scope.value,
                    "expected_change_interval": source.expected_change_interval,
                    "route": source.route.value,
                    "content_hash": source.content_hash,
                    "last_changed_at": source.last_changed_at,
                    "last_ingested_at": source.last_ingested_at,
                    "cached_until": source.cached_until,
                    "pending_hash": source.pending_hash,
                },
            )
            if version is not None:
                await session.execute(
                    text(
                        "INSERT INTO data_source_versions "
                        "(data_source_id, tenant_id, content_hash, content, ingested_at) "
                        "VALUES (:source_id, :tenant_id, :content_hash, :content, :ingested_at)"
                    ),
                    {
                        "source_id": source.id,
                        "tenant_id": source.tenant_id,
                        "content_hash": source.content_hash,
                        "content": version.content,
                        "ingested_at": source.last_changed_at,
                    },
                )

    async def mark_pending(
        self, tenant_id: uuid.UUID, source_id: uuid.UUID, content_hash: str
    ) -> None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            await session.execute(
                text(
                    "UPDATE data_sources SET pending_content_hash = :content_hash, "
                    "updated_at = now() WHERE id = :id AND tenant_id = :tenant_id"
                ),
                {"content_hash": content_hash, "id": source_id, "tenant_id": tenant_id},
            )

    async def record_cached_until(
        self,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        expected_hash: str,
        cached_until: datetime | None,
    ) -> bool:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            result = await session.execute(
                text(
                    "UPDATE data_sources SET cached_until = :cached_until, updated_at = now() "
                    + _UNCHANGED
                ),
                {
                    "cached_until": cached_until,
                    "id": source_id,
                    "tenant_id": tenant_id,
                    "expected_hash": expected_hash,
                },
            )
            return _matched_one(result)

    async def migrate(
        self,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        expected_hash: str,
        route: IngestionRoute,
        interval: timedelta,
        cached_until: datetime | None,
    ) -> bool:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            result = await session.execute(
                text(
                    "UPDATE data_sources SET route = :route, "
                    "expected_change_interval = :interval, cached_until = :cached_until, "
                    "updated_at = now() " + _UNCHANGED
                ),
                {
                    "route": route.value,
                    "interval": interval,
                    "cached_until": cached_until,
                    "id": source_id,
                    "tenant_id": tenant_id,
                    "expected_hash": expected_hash,
                },
            )
            return _matched_one(result)

    async def version_times(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> list[datetime]:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            result = await session.execute(
                text(
                    "SELECT ingested_at FROM data_source_versions WHERE tenant_id = :tenant_id "
                    "AND data_source_id = :source_id ORDER BY ingested_at, seq"
                ),
                {"tenant_id": tenant_id, "source_id": source_id},
            )
            return [row.ingested_at for row in result]

    async def current_content(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> str | None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            value = (
                await session.execute(
                    text(
                        "SELECT content FROM data_source_versions WHERE tenant_id = :tenant_id "
                        "AND data_source_id = :source_id AND content IS NOT NULL "
                        "ORDER BY seq DESC LIMIT 1"
                    ),
                    {"tenant_id": tenant_id, "source_id": source_id},
                )
            ).scalar_one_or_none()
        return None if value is None else str(value)

    async def list_sources(self, tenant_id: uuid.UUID) -> list[DataSource]:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            result = await session.execute(
                text(
                    f"SELECT {_COLUMNS} FROM data_sources WHERE tenant_id = :tenant_id "
                    "ORDER BY source_key, user_id NULLS FIRST"
                ),
                {"tenant_id": tenant_id},
            )
            return [_to_source(row) for row in result]
