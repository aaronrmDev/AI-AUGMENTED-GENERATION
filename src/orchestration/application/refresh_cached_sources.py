import dataclasses
import uuid
from datetime import datetime

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.domain.entities import FreshnessPolicy, IngestionRoute
from src.orchestration.domain.freshness_router import cache_ttl
from src.orchestration.domain.ports import DataSourceRepository, ExpiringCache


class RefreshCachedSources:
    """The batch pre-load for CAG_WITH_RAG_BACKUP sources, driven on whatever cadence a
    caller chooses.

    A source is pre-loaded only if it isn't cached and its content was confirmed by an
    ingestion within its TTL. The entry then expires one TTL after that confirmation,
    not after the pre-load. Re-pre-loading content the feed hasn't confirmed would
    re-serve exactly the stale copy the TTL exists to stop (spec decision 6), so such a
    source stays on RAG until an ingestion confirms it.
    """

    def __init__(
        self,
        repository: DataSourceRepository,
        cache: ExpiringCache,
        warmed: CacheWarmedRetrieve,
        *,
        policy: FreshnessPolicy | None = None,
    ) -> None:
        self._repository = repository
        self._cache = cache
        self._warmed = warmed
        self._policy = policy or FreshnessPolicy()

    async def run(self, tenant_id: uuid.UUID, now: datetime) -> list[str]:
        preloaded: list[str] = []
        for source in await self._repository.list_sources(tenant_id):
            if source.route is not IngestionRoute.CAG_WITH_RAG_BACKUP:
                continue
            if self._cache.contains(tenant_id, source.id):
                continue
            ttl = cache_ttl(source.expected_change_interval, self._policy)
            expires_at = None if ttl is None else source.last_ingested_at + ttl
            if expires_at is not None and expires_at <= now:
                continue
            content = await self._repository.current_content(tenant_id, source.id)
            if content is None:
                continue
            self._cache.preload_until(tenant_id, source.id, content, expires_at)
            self._warmed.note_warmed(tenant_id, source.id, content)
            await self._repository.save(dataclasses.replace(source, cached_until=expires_at))
            preloaded.append(source.source_key)
        return preloaded
