import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from src.mag.domain.entities import ScoredFact
from src.orchestration.domain.entities import (
    BudgetAllocation,
    CacheHit,
    DataSource,
    DataSourceProfile,
    IngestionRoute,
    JobStatus,
    Paradigm,
    SourceVersion,
    TierRequest,
    TierResult,
    WarmEntry,
)


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
    #
    # Async, unlike FrozenCache: HFFrozenCache's real implementation is
    # pure local CPU tensor computation (matching EmbeddingModel's own
    # sync convention), but SemanticMemoryWarmStore's real implementation
    # makes real network calls to Postgres and Qdrant (matching RAG's
    # VectorStore and every MAG repository/index port's own async
    # convention) -- the sync-vs-async split in this codebase tracks
    # whether a real implementation does I/O, not which paradigm a port
    # belongs to.
    @abstractmethod
    async def promote(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID, content: str
    ) -> None: ...

    @abstractmethod
    async def lookup(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> WarmEntry | None: ...

    @abstractmethod
    async def demote(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> None: ...

    @abstractmethod
    async def contains(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> bool: ...


class QueryClassifier(ABC):
    # Async because LlmQueryClassifier makes a network call -- a port's
    # sync/async shape tracks whether ANY real implementation does I/O
    # (WarmStore's comment above). query_embedding is passed in so a
    # classifier that needs it never embeds the query a second time;
    # UnifiedAnswerQuestion has already paid for it.
    @abstractmethod
    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]: ...


class CascadeTier(ABC):
    @property
    @abstractmethod
    def paradigm(self) -> Paradigm: ...

    @abstractmethod
    async def attempt(self, request: TierRequest) -> TierResult: ...


class SemanticFactSearch(ABC):
    # Each call is its own unit of work, owned by the implementation. The
    # cascade cancels a timed-out tier mid-flight, and a cancelled SQLAlchemy
    # query terminates the connection it ran on -- a tier sharing the request's
    # session would break every later use of it (measured in
    # tests/integration/test_orchestration_meta_layer.py).
    @abstractmethod
    async def search(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
    ) -> list[ScoredFact]: ...


class SessionBudgetRecorder(ABC):
    # user_id as well as tenant_id: RLS isolates tenants, and the write's own
    # WHERE clause keeps one user from overwriting another user's record.
    @abstractmethod
    async def record(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        allocation: BudgetAllocation,
        contributing: frozenset[Paradigm],
    ) -> None: ...


class DataSourceRepository(ABC):
    """The durable record of every freshness-routed data source and its versions.
    Every method is tenant-scoped; user-scoped sources are also keyed by user_id."""

    @abstractmethod
    async def get(
        self, tenant_id: uuid.UUID, source_key: str, user_id: uuid.UUID | None
    ) -> DataSource | None: ...

    @abstractmethod
    async def save(self, source: DataSource, version: SourceVersion | None = None) -> None:
        """Insert the source, or update an existing one's content fields: its hash,
        change and ingestion times, cache expiry, and pending marker. An existing source's
        route and interval are left alone; only migrate changes them. When version is
        given, it is recorded at source.last_changed_at in the same transaction."""

    @abstractmethod
    async def mark_pending(
        self, tenant_id: uuid.UUID, source_id: uuid.UUID, content_hash: str
    ) -> None:
        """Record that a change's effects are starting, before they run."""

    @abstractmethod
    async def record_cached_until(
        self,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        expected_hash: str,
        cached_until: datetime | None,
    ) -> bool:
        """Set the cache expiry only if the stored hash is still expected_hash and no
        change is pending. Returns False, writing nothing, otherwise."""

    @abstractmethod
    async def migrate(
        self,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        expected_hash: str,
        route: IngestionRoute,
        interval: timedelta,
        cached_until: datetime | None,
    ) -> bool:
        """Change the route, interval, and cache expiry only if the stored hash is still
        expected_hash and no change is pending. Returns False, writing nothing, otherwise."""

    @abstractmethod
    async def version_times(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> list[datetime]:
        """Every version's ingestion time, oldest first."""

    @abstractmethod
    async def current_content(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> str | None: ...

    @abstractmethod
    async def list_sources(self, tenant_id: uuid.UUID) -> list[DataSource]: ...


class RagIndex(ABC):
    """Holds one data source's current text in RAG's stores, as one document."""

    @abstractmethod
    async def replace(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, title: str, text: str
    ) -> None:
        """Make text the document's only content: no chunk of an earlier version may
        remain retrievable afterwards."""


class ExpiringCache(FrozenCache):
    """A FrozenCache whose entries can expire. lookup and contains report an expired
    entry as absent, so a cascade falls through to RAG without any sweep."""

    @abstractmethod
    def preload_until(
        self,
        tenant_id: uuid.UUID,
        document_id: uuid.UUID,
        content: str,
        expires_at: datetime | None,
    ) -> None: ...

    @abstractmethod
    def renew(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, expires_at: datetime | None
    ) -> bool:
        """Move a live entry's expiry. Returns False, changing nothing, when there is
        no live entry to renew."""


class SessionFactWriter(ABC):
    """Writes a user-scoped data source's text into MAG as that user's fact."""

    @abstractmethod
    async def record(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, fact_key: str, fact_value: str
    ) -> None: ...


class IngestionJobDispatcher(ABC):
    """Runs an ingestion asynchronously and reports on it. IngestDataSource never
    depends on this -- it has no idea whether it's being run synchronously in a
    test, from the evaluation harness, or inside a Celery task, and this port is
    what keeps it that way."""

    @abstractmethod
    async def dispatch(
        self,
        *,
        tenant_id: uuid.UUID,
        profile: DataSourceProfile,
        content: str,
        user_id: uuid.UUID | None,
    ) -> str:
        """Enqueues the ingestion and returns a task id the caller can poll."""

    @abstractmethod
    async def status(
        self, task_id: str, *, tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> JobStatus | None:
        """None means the task id doesn't exist or doesn't belong to this caller --
        the router maps that to the same 404 shape used everywhere else in this API."""
