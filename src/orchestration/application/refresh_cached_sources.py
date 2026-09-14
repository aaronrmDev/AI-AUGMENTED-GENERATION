import uuid
from datetime import datetime

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.domain.entities import FreshnessPolicy, IngestionRoute
from src.orchestration.domain.freshness_router import cache_ttl
from src.orchestration.domain.ports import DataSourceRepository, ExpiringCache
from src.orchestration.domain.sync_mixer import content_hash


class RefreshCachedSources:
    """The batch pre-load for CAG_WITH_RAG_BACKUP sources, driven on whatever cadence a
    caller chooses.

    A source is pre-loaded only if it isn't cached, has no pending change, and its
    content was confirmed by an ingestion within its TTL. The entry expires one TTL
    after that confirmation, not after the pre-load. Re-pre-loading content the feed
    hasn't confirmed would re-serve exactly the stale copy the TTL exists to stop (spec
    decision 6).

    Ingestion can run concurrently, so every step re-checks the listing it started
    from:
    - text whose hash no longer matches the listed source isn't pre-loaded;
    - the expiry is recorded with a write conditional on that hash;
    - if a change landed after the pre-load, the write matches nothing, and the fresh
      copy is evicted again.
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
            if source.pending_hash is not None or self._cache.contains(tenant_id, source.id):
                continue
            ttl = cache_ttl(source.expected_change_interval, self._policy)
            expires_at = None if ttl is None else source.last_ingested_at + ttl
            if expires_at is not None and expires_at <= now:
                continue
            content = await self._repository.current_content(tenant_id, source.id)
            if content is None or content_hash(content) != source.content_hash:
                continue
            self._cache.preload_until(tenant_id, source.id, content, expires_at)
            self._warmed.note_warmed(tenant_id, source.id, content)
            recorded = await self._repository.record_cached_until(
                tenant_id, source.id, source.content_hash, expires_at
            )
            if not recorded:
                self._cache.evict(tenant_id, source.id)
                self._warmed.forget(tenant_id, source.id)
                continue
            preloaded.append(source.source_key)
        return preloaded
