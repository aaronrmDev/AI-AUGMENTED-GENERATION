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
    SourceVersion,
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
      ingestions keep the stored route and interval, which ReviewSourceFreshness may
      have re-learned.
    - A change replaces RAG's copy and, for a cached source, evicts the CAG entry at
      once (Concept 9: "CAG price cache invalidated"). Pre-loading the new text is
      RefreshCachedSources' batch job.
    - A confirmation (identical content) renews a live cache entry for one TTL.

    A change to an existing source is marked pending before any effect runs, and the
    save that records it clears the marker. If an effect or the save fails, the marker
    stays: refresh and review skip the source, so neither can re-cache the superseded
    text. The next ingestion re-applies the change, even if the feed has meanwhile
    reverted to the stored content. Source ids are deterministic, so a retry replaces
    the same document.

    route_policy is the measurement's ablation seam. Placing a user-scoped source
    anywhere but MAG puts personal data in tenant-wide stores, so it has to be allowed
    explicitly with permit_user_scope_outside_mag.
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
        permit_user_scope_outside_mag: bool = False,
    ) -> None:
        self._repository = repository
        self._rag_index = rag_index
        self._cache = cache
        self._warmed = warmed
        self._fact_writer = fact_writer
        self._policy = policy or FreshnessPolicy()
        self._route_policy = route_policy
        self._permit_user_scope_outside_mag = permit_user_scope_outside_mag

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
            source = self._new_source(tenant_id, profile, digest, now, user_id)
            changed = reapply = True
        else:
            changed = existing.content_hash != digest
            reapply = changed or existing.pending_hash is not None
            source = dataclasses.replace(
                existing,
                content_hash=digest,
                last_ingested_at=now,
                last_changed_at=now if changed else existing.last_changed_at,
                pending_hash=None,
            )
            if reapply:
                await self._repository.mark_pending(tenant_id, existing.id, digest)

        cached_until = await self._apply_route(source, content, reapply, now)
        version = None
        if changed:
            version = SourceVersion(None if source.route is IngestionRoute.MAG else content)
        await self._repository.save(dataclasses.replace(source, cached_until=cached_until), version)
        return IngestionResult(source.id, source.route, changed)

    def _new_source(
        self,
        tenant_id: uuid.UUID,
        profile: DataSourceProfile,
        digest: str,
        now: datetime,
        user_id: uuid.UUID | None,
    ) -> DataSource:
        route = self._route_policy(profile, self._policy)
        if route is IngestionRoute.MAG and profile.scope is SourceScope.TENANT:
            raise ValueError("a tenant-scoped source cannot route to MAG")
        if (
            profile.scope is SourceScope.USER
            and route is not IngestionRoute.MAG
            and not self._permit_user_scope_outside_mag
        ):
            raise ValueError(
                "a user-scoped source routes to MAG; placing it elsewhere puts personal data "
                "in tenant-wide stores and must be permitted explicitly"
            )
        return DataSource(
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

    async def _apply_route(
        self, source: DataSource, content: str, reapply: bool, now: datetime
    ) -> datetime | None:
        if source.route is IngestionRoute.MAG:
            if reapply and source.user_id is not None:
                await self._fact_writer.record(
                    source.tenant_id, source.user_id, source.source_key, content
                )
            return None

        if reapply:
            await self._rag_index.replace(source.tenant_id, source.id, source.source_key, content)
        if source.route is IngestionRoute.RAG_ONLY:
            return None

        if reapply:
            self._cache.evict(source.tenant_id, source.id)
            self._warmed.forget(source.tenant_id, source.id)
            return None
        ttl = cache_ttl(source.expected_change_interval, self._policy)
        expires_at = None if ttl is None else now + ttl
        renewed = self._cache.renew(source.tenant_id, source.id, expires_at)
        return expires_at if renewed else None
