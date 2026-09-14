import dataclasses
import uuid
from collections.abc import Callable
from datetime import datetime

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.domain.entities import (
    DataSource,
    DataSourceProfile,
    FreshnessPolicy,
    IngestionResult,
    IngestionRoute,
    SourceScope,
)
from src.orchestration.domain.errors import ScopeMismatch
from src.orchestration.domain.freshness_router import cache_ttl, route_for, source_id_for
from src.orchestration.domain.ports import (
    DataSourceRepository,
    ExpiringCache,
    RagIndex,
    SessionFactWriter,
)
from src.orchestration.domain.sync_mixer import content_hash

RoutePolicy = Callable[[DataSourceProfile, FreshnessPolicy], IngestionRoute]


class IngestDataSource:
    """Concept 9's "route at ingestion", run once per delivered version of a source.

    - A new source is routed by route_policy from its declared profile. Later
      ingestions keep the stored route and interval, which ReviewSourceFreshness
      may have re-learned.
    - A change replaces RAG's copy and, for a cached source, evicts the CAG entry at
      once (Concept 9: "CAG price cache invalidated"). Pre-loading the new text is
      RefreshCachedSources' batch job.
    - A confirmation (identical content) renews a live cache entry for one TTL.

    External effects run before the record is saved. A failure part-way therefore
    leaves the stored hash on the previous version, the retry is seen as a change
    again, and source ids are deterministic, so it replaces the same document.
    """

    def __init__(
        self,
        repository: DataSourceRepository,
        rag_index: RagIndex,
        cache: ExpiringCache,
        warmed: CacheWarmedRetrieve,
        fact_writer: SessionFactWriter,
        *,
        policy: FreshnessPolicy | None = None,
        route_policy: RoutePolicy = route_for,
    ) -> None:
        self._repository = repository
        self._rag_index = rag_index
        self._cache = cache
        self._warmed = warmed
        self._fact_writer = fact_writer
        self._policy = policy or FreshnessPolicy()
        self._route_policy = route_policy

    async def execute(
        self,
        tenant_id: uuid.UUID,
        profile: DataSourceProfile,
        content: str,
        now: datetime,
        user_id: uuid.UUID | None = None,
    ) -> IngestionResult:
        if (profile.scope is SourceScope.USER) != (user_id is not None):
            raise ScopeMismatch(profile.source_key, profile.scope.value, user_id is not None)

        existing = await self._repository.get(tenant_id, profile.source_key, user_id)
        digest = content_hash(content)
        if existing is None:
            route = self._route_policy(profile, self._policy)
            if route is IngestionRoute.MAG and profile.scope is SourceScope.TENANT:
                raise ValueError("a tenant-scoped source cannot route to MAG")
            source = DataSource(
                id=source_id_for(tenant_id, profile.source_key, user_id),
                tenant_id=tenant_id,
                user_id=user_id,
                source_key=profile.source_key,
                scope=profile.scope,
                expected_change_interval=profile.expected_change_interval,
                route=route,
                content_hash=digest,
                last_changed_at=now,
                last_ingested_at=now,
            )
            changed = True
        else:
            changed = existing.content_hash != digest
            source = dataclasses.replace(
                existing,
                content_hash=digest,
                last_ingested_at=now,
                last_changed_at=now if changed else existing.last_changed_at,
            )

        cached_until = await self._apply_route(source, content, changed, now)
        await self._repository.save(
            dataclasses.replace(source, cached_until=cached_until),
            changed_content=content if changed else None,
        )
        return IngestionResult(source.id, source.route, changed)

    async def _apply_route(
        self, source: DataSource, content: str, changed: bool, now: datetime
    ) -> datetime | None:
        if source.route is IngestionRoute.MAG:
            if changed and source.user_id is not None:
                await self._fact_writer.record(
                    source.tenant_id, source.user_id, source.source_key, content
                )
            return None

        if changed:
            await self._rag_index.replace(source.tenant_id, source.id, source.source_key, content)
        if source.route is IngestionRoute.RAG_ONLY:
            return None

        if changed:
            self._cache.evict(source.tenant_id, source.id)
            self._warmed.forget(source.tenant_id, source.id)
            return None
        ttl = cache_ttl(source.expected_change_interval, self._policy)
        expires_at = None if ttl is None else now + ttl
        renewed = self._cache.renew(source.tenant_id, source.id, expires_at)
        return expires_at if renewed else None
