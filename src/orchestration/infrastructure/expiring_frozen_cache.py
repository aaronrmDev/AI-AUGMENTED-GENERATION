import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from src.orchestration.domain.entities import CacheHit
from src.orchestration.domain.ports import ExpiringCache, FrozenCache

_Key = tuple[uuid.UUID, uuid.UUID]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ExpiringFrozenCache(ExpiringCache):
    """Adds Concept 9's TTL to any FrozenCache.

    An entry is expired from the instant the clock reaches its expiry. The first
    lookup or contains to see that evicts it from the wrapped cache and reports it
    absent. CacheWarmedRetrieve's confirmation then misses, and the orchestration
    cascade falls through to RAG with no sweep to wait for. Entries preloaded
    without an expiry never expire.

    CagTier reaches lookup from worker threads, so the expiry check and the eviction
    it triggers happen together under one lock, as do evict and renew. preload_until
    records the new expiry before the slow preload runs. A concurrent expiry check
    therefore sees the entry being written, never its expired predecessor, and can't
    evict the new copy.
    """

    def __init__(self, inner: FrozenCache, clock: Callable[[], datetime] = _utc_now) -> None:
        self._inner = inner
        self._clock = clock
        self._expiry: dict[_Key, datetime | None] = {}
        self._lock = threading.Lock()

    def preload(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        self.preload_until(tenant_id, document_id, content, None)

    def preload_until(
        self,
        tenant_id: uuid.UUID,
        document_id: uuid.UUID,
        content: str,
        expires_at: datetime | None,
    ) -> None:
        with self._lock:
            self._expiry[(tenant_id, document_id)] = expires_at
        self._inner.preload(tenant_id, document_id, content)

    def renew(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, expires_at: datetime | None
    ) -> bool:
        with self._lock:
            if self._expire_if_due(tenant_id, document_id):
                return False
            if not self._inner.contains(tenant_id, document_id):
                return False
            self._expiry[(tenant_id, document_id)] = expires_at
            return True

    def lookup(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> CacheHit | None:
        with self._lock:
            if self._expire_if_due(tenant_id, document_id):
                return None
        return self._inner.lookup(tenant_id, document_id)

    def evict(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        with self._lock:
            self._expiry.pop((tenant_id, document_id), None)
            self._inner.evict(tenant_id, document_id)

    def contains(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        with self._lock:
            if self._expire_if_due(tenant_id, document_id):
                return False
        return self._inner.contains(tenant_id, document_id)

    def _expire_if_due(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        """Must be called holding the lock."""
        key = (tenant_id, document_id)
        expires_at = self._expiry.get(key)
        if expires_at is None or self._clock() < expires_at:
            return False
        del self._expiry[key]
        self._inner.evict(tenant_id, document_id)
        return True
