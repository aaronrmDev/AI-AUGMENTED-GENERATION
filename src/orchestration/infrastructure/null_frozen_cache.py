import uuid
from datetime import datetime

from src.orchestration.domain.entities import CacheHit
from src.orchestration.domain.ports import ExpiringCache


class NullFrozenCache(ExpiringCache):
    """A stand-in for a real, GPU-resident CAG cache, which no production process
    in this project serves yet (docs/architecture/OVERVIEW.md discloses this for
    the API's own unified pipeline). Every method is a safe no-op: nothing is ever
    cached, so IngestDataSource's cache-touching code paths (eviction on change,
    renewal on confirmation) do real, harmless nothing until a real cache exists.
    """

    def preload(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        return None

    def lookup(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> CacheHit | None:
        return None

    def evict(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        return None

    def contains(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        return False

    def preload_until(
        self,
        tenant_id: uuid.UUID,
        document_id: uuid.UUID,
        content: str,
        expires_at: datetime | None,
    ) -> None:
        return None

    def renew(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, expires_at: datetime | None
    ) -> bool:
        return False
