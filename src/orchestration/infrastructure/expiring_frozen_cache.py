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

    The expiry map is lock-guarded, because CagTier reaches lookup from worker
    threads.
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
        self._inner.preload(tenant_id, document_id, content)
        with self._lock:
            self._expiry[(tenant_id, document_id)] = expires_at

    def renew(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, expires_at: datetime | None
    ) -> bool:
        if not self.contains(tenant_id, document_id):
            return False
        with self._lock:
            self._expiry[(tenant_id, document_id)] = expires_at
        return True

    def lookup(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> CacheHit | None:
        if self._expired(tenant_id, document_id):
            return None
        return self._inner.lookup(tenant_id, document_id)

    def evict(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        with self._lock:
            self._expiry.pop((tenant_id, document_id), None)
        self._inner.evict(tenant_id, document_id)

    def contains(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        if self._expired(tenant_id, document_id):
            return False
        return self._inner.contains(tenant_id, document_id)

    def _expired(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        with self._lock:
            expires_at = self._expiry.get((tenant_id, document_id))
        if expires_at is None or self._clock() < expires_at:
            return False
        self.evict(tenant_id, document_id)
        return True
