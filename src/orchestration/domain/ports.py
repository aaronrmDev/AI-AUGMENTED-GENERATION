import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from src.orchestration.domain.entities import CacheHit, WarmEntry


class AccessFrequencyTracker(ABC):
    # One shared tracking primitive behind both Cache-Warmed RAG (top-N
    # snapshot) and tiering (threshold-based promote/demote) -- OVERVIEW.md
    # describes the same underlying signal for both ("an analytics process
    # tracks which documents RAG retrieves most often" for Cache-Warmed RAG;
    # "access-pattern tracking for RAG-retrieved data" for tiering).
    # `now`/`at` are explicit parameters rather than wall-clock reads inside
    # the tracker, so tests can exercise real promotion/demotion decisions
    # over a controlled time window without sleeping through it.
    #
    # tenant_id scopes every method the same way RAG's own VectorStore and
    # every MAG port already do (a review finding caught the first version
    # of this port omitting it entirely, which would have let one tenant's
    # access counts and warmed content leak into another's query results
    # once a real caller wired a shared instance across tenants, matching
    # this project's own singleton-service DI convention).
    @abstractmethod
    def record_access(self, tenant_id: uuid.UUID, document_id: uuid.UUID, at: datetime) -> None: ...

    @abstractmethod
    def access_count(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, window: timedelta, now: datetime
    ) -> int: ...

    @abstractmethod
    def most_accessed(
        self, tenant_id: uuid.UUID, n: int, window: timedelta, now: datetime
    ) -> list[uuid.UUID]: ...


class FrozenCache(ABC):
    # tenant_id-scoped for the same reason as AccessFrequencyTracker above.
    @abstractmethod
    def preload(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None: ...

    @abstractmethod
    def lookup(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> CacheHit | None: ...

    @abstractmethod
    def evict(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None: ...

    @abstractmethod
    def contains(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool: ...


class UserScopedAccessFrequencyTracker(ABC):
    # Structurally close to AccessFrequencyTracker above but a genuinely
    # different port, not a subtype or a reused implementation: CAG's
    # frozen cache is a tenant-wide SHARED resource (no user concept), but
    # MAG's warm tier is inherently PERSONAL (every MAG table is
    # user_id-keyed) -- reusing the tenant-only tracker for MAG's tiering
    # would conflate different users' access patterns for the same
    # document, incorrectly promoting a document into user B's memory
    # because user A accessed it heavily. The extra user_id dimension
    # here is load-bearing, not decorative.
    @abstractmethod
    def record_access(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID, at: datetime
    ) -> None: ...

    @abstractmethod
    def access_count(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        window: timedelta,
        now: datetime,
    ) -> int: ...

    @abstractmethod
    def most_accessed(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, n: int, window: timedelta, now: datetime
    ) -> list[uuid.UUID]: ...


class WarmStore(ABC):
    # Same four-verb shape as FrozenCache (promote/preload, demote/evict)
    # because the underlying idea -- place content in a fast personal
    # store, check it, remove it -- really is the same idea FrozenCache
    # already named. A separate port because the physical realization (a
    # Postgres+Qdrant semantic fact, not a GPU-resident KV cache) and the
    # required scoping (user_id) both genuinely differ.
    @abstractmethod
    def promote(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID, content: str
    ) -> None: ...

    @abstractmethod
    def lookup(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> WarmEntry | None: ...

    @abstractmethod
    def demote(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> None: ...

    @abstractmethod
    def contains(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> bool: ...
