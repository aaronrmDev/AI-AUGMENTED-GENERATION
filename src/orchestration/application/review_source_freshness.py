import dataclasses
import uuid
from datetime import datetime

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.domain.entities import (
    FreshnessPolicy,
    IngestionRoute,
    SourceMigration,
    SourceScope,
)
from src.orchestration.domain.freshness_router import decide_migration
from src.orchestration.domain.ports import DataSourceRepository, ExpiringCache


class ReviewSourceFreshness:
    """Concept 9's "monitor & migrate", for tenant-scoped sources.

    A demotion to RAG_ONLY evicts and forgets the cached copy at once. A promotion only
    changes the route and interval, and the next RefreshCachedSources run pre-loads the
    source. Either way, the observed interval replaces the declared one, so later
    routing and TTLs follow the source's real behaviour. User-scoped sources are never
    reviewed: frequency can't move data into or out of a user's own store.
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

    async def run(self, tenant_id: uuid.UUID, now: datetime) -> list[SourceMigration]:
        migrations: list[SourceMigration] = []
        for source in await self._repository.list_sources(tenant_id):
            if source.scope is SourceScope.USER:
                continue
            times = await self._repository.version_times(tenant_id, source.id)
            decision = decide_migration(source, times, now, self._policy)
            if decision is None:
                continue
            cached_until = source.cached_until
            if decision.to_route is IngestionRoute.RAG_ONLY:
                self._cache.evict(tenant_id, source.id)
                self._warmed.forget(tenant_id, source.id)
                cached_until = None
            await self._repository.save(
                dataclasses.replace(
                    source,
                    route=decision.to_route,
                    expected_change_interval=decision.interval,
                    cached_until=cached_until,
                )
            )
            migrations.append(
                SourceMigration(
                    source.source_key, source.route, decision.to_route, decision.interval
                )
            )
        return migrations
