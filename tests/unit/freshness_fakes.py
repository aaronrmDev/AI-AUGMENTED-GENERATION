import dataclasses
import uuid
from datetime import datetime, timedelta

from src.orchestration.domain.entities import (
    CacheHit,
    DataSource,
    IngestionRoute,
    SourceVersion,
)
from src.orchestration.domain.ports import (
    DataSourceRepository,
    ExpiringCache,
    RagIndex,
    SessionFactWriter,
)
from src.orchestration.domain.sync_mixer import content_hash


class FakeDataSourceRepository(DataSourceRepository):
    """Mirrors PostgresDataSourceRepository's write semantics: a save never changes an
    existing source's route or interval, and the conditional writes apply only while
    the stored hash is the expected one and no change is pending."""

    def __init__(self) -> None:
        self.sources: dict[uuid.UUID, DataSource] = {}
        self.versions: dict[uuid.UUID, list[tuple[datetime, str | None]]] = {}

    async def get(
        self, tenant_id: uuid.UUID, source_key: str, user_id: uuid.UUID | None
    ) -> DataSource | None:
        return next(
            (
                s for s in self.sources.values()
                if (s.tenant_id, s.source_key, s.user_id) == (tenant_id, source_key, user_id)
            ),
            None,
        )

    async def save(self, source: DataSource, version: SourceVersion | None = None) -> None:
        stored = self.sources.get(source.id)
        if stored is not None:
            source = dataclasses.replace(
                source, route=stored.route, expected_change_interval=stored.expected_change_interval
            )
        self.sources[source.id] = source
        if version is not None:
            self.versions.setdefault(source.id, []).append(
                (source.last_changed_at, version.content)
            )

    async def mark_pending(
        self, tenant_id: uuid.UUID, source_id: uuid.UUID, content_hash: str
    ) -> None:
        if self._visible(tenant_id, source_id):
            self.sources[source_id] = dataclasses.replace(
                self.sources[source_id], pending_hash=content_hash
            )

    def _unchanged(self, tenant_id: uuid.UUID, source_id: uuid.UUID, expected_hash: str) -> bool:
        if not self._visible(tenant_id, source_id):
            return False
        stored = self.sources[source_id]
        return stored.content_hash == expected_hash and stored.pending_hash is None

    async def record_cached_until(
        self,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        expected_hash: str,
        cached_until: datetime | None,
    ) -> bool:
        if not self._unchanged(tenant_id, source_id, expected_hash):
            return False
        self.sources[source_id] = dataclasses.replace(
            self.sources[source_id], cached_until=cached_until
        )
        return True

    async def migrate(
        self,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        expected_hash: str,
        route: IngestionRoute,
        interval: timedelta,
        cached_until: datetime | None,
    ) -> bool:
        if not self._unchanged(tenant_id, source_id, expected_hash):
            return False
        self.sources[source_id] = dataclasses.replace(
            self.sources[source_id],
            route=route,
            expected_change_interval=interval,
            cached_until=cached_until,
        )
        return True

    def _visible(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> bool:
        source = self.sources.get(source_id)
        return source is not None and source.tenant_id == tenant_id

    async def version_times(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> list[datetime]:
        if not self._visible(tenant_id, source_id):
            return []
        return sorted(at for at, _ in self.versions.get(source_id, []))

    async def current_content(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> str | None:
        if not self._visible(tenant_id, source_id):
            return None
        texts = [text for _, text in self.versions.get(source_id, []) if text is not None]
        return texts[-1] if texts else None

    async def list_sources(self, tenant_id: uuid.UUID) -> list[DataSource]:
        return [s for s in self.sources.values() if s.tenant_id == tenant_id]


class FakeRagIndex(RagIndex):
    def __init__(self) -> None:
        self.replaced: list[tuple[uuid.UUID, uuid.UUID, str, str]] = []

    async def replace(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, title: str, text: str
    ) -> None:
        self.replaced.append((tenant_id, document_id, title, text))


class FakeExpiringCache(ExpiringCache):
    """Records calls and ignores expiry: ExpiringFrozenCache's own tests cover expiry."""

    def __init__(self) -> None:
        self._entries: dict[tuple[uuid.UUID, uuid.UUID], tuple[str, datetime | None]] = {}
        self.preload_calls: list[tuple[uuid.UUID, uuid.UUID, str, datetime | None]] = []
        self.renew_calls: list[tuple[uuid.UUID, uuid.UUID, datetime | None]] = []
        self.evict_calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def preload(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        self.preload_until(tenant_id, document_id, content, None)

    def preload_until(
        self,
        tenant_id: uuid.UUID,
        document_id: uuid.UUID,
        content: str,
        expires_at: datetime | None,
    ) -> None:
        self.preload_calls.append((tenant_id, document_id, content, expires_at))
        self._entries[(tenant_id, document_id)] = (content, expires_at)

    def renew(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, expires_at: datetime | None
    ) -> bool:
        key = (tenant_id, document_id)
        if key not in self._entries:
            return False
        self.renew_calls.append((tenant_id, document_id, expires_at))
        self._entries[key] = (self._entries[key][0], expires_at)
        return True

    def expiry(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> datetime | None:
        return self._entries[(tenant_id, document_id)][1]

    def lookup(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> CacheHit | None:
        entry = self._entries.get((tenant_id, document_id))
        return None if entry is None else CacheHit(content_hash(entry[0]), None)

    def evict(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        self.evict_calls.append((tenant_id, document_id))
        self._entries.pop((tenant_id, document_id), None)

    def contains(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        return (tenant_id, document_id) in self._entries


class FakeSessionFactWriter(SessionFactWriter):
    def __init__(self) -> None:
        self.records: list[tuple[uuid.UUID, uuid.UUID, str, str]] = []

    async def record(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, fact_key: str, fact_value: str
    ) -> None:
        self.records.append((tenant_id, user_id, fact_key, fact_value))
