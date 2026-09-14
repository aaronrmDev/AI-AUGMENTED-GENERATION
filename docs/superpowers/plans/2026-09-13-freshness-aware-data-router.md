# Orchestration Meta-Layer Batch B: Freshness-Aware Data Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and measure Concept 9's Freshness-Aware Data Router: classify each data source by change frequency, route it at ingestion to RAG, CAG with RAG backup, or MAG, expire cached copies with a TTL, and migrate sources whose real change rate drifts.

**Architecture:** The routing, TTL, and migration rules are pure functions in `src/orchestration/domain/freshness_router.py`. Three use cases in `src/orchestration/application/` apply them: `IngestDataSource`, `RefreshCachedSources`, and `ReviewSourceFreshness`. They depend on four new ports, whose adapters live in `src/orchestration/infrastructure/`:

- a Postgres source registry behind migration 0006;
- an expiring wrapper around any `FrozenCache`;
- a RAG index that replaces a source's chunks in Qdrant and Postgres;
- a writer that records user-scoped sources through MAG's `RecordSemanticFact`.

A simulated-clock runner measures the spec's five claims against real stores.

**Tech Stack:** Python 3.14 venv (ruff target py311, mypy strict on `src/`), pytest with `asyncio_mode = "auto"`, SQLAlchemy async + asyncpg, Alembic, TestContainers (`pgvector/pgvector:pg16`, `qdrant/qdrant:v1.16.2`, `neo4j:5-community`), sentence-transformers MiniLM, transformers `distilgpt2`.

**Spec:** `docs/superpowers/specs/2026-09-13-freshness-aware-data-router-design.md`

## Global Constraints

- **Location.** Work only inside `.worktrees/feature/155-freshness-aware-data-router/`. Every path below is relative to that directory.
- **Python.** Run it as `../../../.venv/Scripts/python.exe`.
- **Unit tests.** `../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider`. Baseline before this batch: 1037 passed.
- **Integration tests.** `../../../.venv/Scripts/python.exe -m pytest tests/integration/<file> -q -p no:cacheprovider`. They need Docker. Baseline: 231 passed, 8 skipped.
- **Lint and types.** `../../../.venv/Scripts/python.exe -m ruff check <changed files>` and `../../../.venv/Scripts/python.exe -m mypy src`. Line length is 100. The code below aims for it; wrap any line ruff still flags (E501) without changing behaviour.
- **Scope.** Every port method takes `tenant_id`. User-scoped sources also carry `user_id`.
- **Time.** Nothing in `src/` reads the wall clock inside a rule. Use cases take `now`; `ExpiringFrozenCache` takes a `clock`. Unit tests never sleep.
- **Defaults are disclosed, not tuned.** One day for the volatile boundary, 3 changes to demote, 7 quiet boundaries to promote, and a TTL factor of 0.5.
- **Blocking work.** CPU work on an async path goes through `asyncio.to_thread`: embedding inside `ChunkedRagIndex`, for example.
- **Commits.** Messages follow `.gitmessage` (Conventional Commits), cite `Refs #155` plus the task's issue, and end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## File map

| File | Responsibility |
|---|---|
| `src/orchestration/domain/entities.py` (modify) | `SourceScope`, `IngestionRoute`, `DataSourceProfile`, `FreshnessPolicy`, `DataSource`, `IngestionResult`, `MigrationDecision`, `SourceMigration` |
| `src/orchestration/domain/errors.py` (modify) | `ScopeMismatch` |
| `src/orchestration/domain/freshness_router.py` | `route_for`, `cache_ttl`, `observed_change_interval`, `decide_migration`, `source_id_for` |
| `src/orchestration/domain/ports.py` (modify) | `DataSourceRepository`, `RagIndex`, `ExpiringCache`, `SessionFactWriter` |
| `src/orchestration/application/cache_warmed_retrieve.py` (modify) | `forget` |
| `src/orchestration/application/ingest_data_source.py` | `IngestDataSource`, `RoutePolicy` |
| `src/orchestration/application/refresh_cached_sources.py` | `RefreshCachedSources` |
| `src/orchestration/application/review_source_freshness.py` | `ReviewSourceFreshness` |
| `src/orchestration/infrastructure/expiring_frozen_cache.py` | `ExpiringFrozenCache` |
| `alembic/versions/0006_data_sources.py` | `data_sources`, `data_source_versions`, RLS |
| `src/orchestration/infrastructure/postgres_data_source_repository.py` | `PostgresDataSourceRepository` |
| `src/rag/infrastructure/qdrant_vector_store.py` (modify) | `delete_document` |
| `src/rag/infrastructure/postgres_document_repository.py` (modify) | `delete_document` |
| `src/orchestration/infrastructure/chunked_rag_index.py` | `ChunkedRagIndex` |
| `src/orchestration/infrastructure/record_semantic_fact_writer.py` | `RecordSemanticFactWriter` |
| `tests/unit/freshness_fakes.py` | `FakeDataSourceRepository`, `FakeRagIndex`, `FakeExpiringCache`, `FakeSessionFactWriter` |
| `tests/integration/freshness_env.py` | shared real-store composition for the end-to-end tests |
| `evaluation/domain/freshness_metrics.py` | `ProbeObservation`, `PlacementTally`, `tally_probes`, `PreloadTracker`, `MigrationObservation` |
| `evaluation/infrastructure/freshness_report.py` | `render_freshness_measurements` |
| `evaluation/scenarios/freshness-router/run_freshness_measurements.py` | the simulated-clock runner |
| `evaluation/reports/freshness-router*.md` | the generated tables and the narrative report |

---

### Task 1: Domain entities, errors, and the freshness rules

**Files:**
- Modify: `src/orchestration/domain/entities.py` (append; add `from datetime import datetime, timedelta` to the imports)
- Modify: `src/orchestration/domain/errors.py` (append)
- Create: `src/orchestration/domain/freshness_router.py`
- Test: `tests/unit/test_freshness_router.py`, `tests/unit/test_freshness_entities.py`

**Interfaces:**
- **Produces for every later task:** the entities, `ScopeMismatch`, and the five functions below, with exactly these signatures.

- [ ] **Step 1: Write the failing entity tests** in `tests/unit/test_freshness_entities.py`

```python
from datetime import timedelta

import pytest

from src.orchestration.domain.entities import DataSourceProfile, FreshnessPolicy, SourceScope


def test_a_blank_source_key_is_rejected():
    with pytest.raises(ValueError):
        DataSourceProfile("  ", SourceScope.TENANT, timedelta(hours=1))


def test_a_non_positive_change_interval_is_rejected():
    with pytest.raises(ValueError):
        DataSourceProfile("prices", SourceScope.TENANT, timedelta(0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volatile_below": timedelta(0)},
        {"demote_after_changes": 0},
        {"demote_after_changes": 3, "promote_after_quiet_multiple": 3},
        {"ttl_factor": 0.0},
    ],
)
def test_an_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FreshnessPolicy(**kwargs)


def test_the_default_policy_matches_the_spec():
    policy = FreshnessPolicy()
    assert policy.volatile_below == timedelta(days=1)
    assert (policy.demote_after_changes, policy.promote_after_quiet_multiple) == (3, 7)
    assert policy.ttl_factor == 0.5


def test_ttl_can_be_disabled():
    assert FreshnessPolicy(ttl_factor=None).ttl_factor is None
```

- [ ] **Step 2: Write the failing rule tests** in `tests/unit/test_freshness_router.py`

```python
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.orchestration.domain.entities import (
    DataSource,
    DataSourceProfile,
    FreshnessPolicy,
    IngestionRoute,
    MigrationDecision,
    SourceScope,
)
from src.orchestration.domain.freshness_router import (
    cache_ttl,
    decide_migration,
    observed_change_interval,
    route_for,
    source_id_for,
)

POLICY = FreshnessPolicy()
NOW = datetime(2026, 1, 20, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)
TENANT, USER = uuid.uuid4(), uuid.uuid4()


def _source(route: IngestionRoute, scope: SourceScope = SourceScope.TENANT) -> DataSource:
    return DataSource(
        id=uuid.uuid4(), tenant_id=TENANT, user_id=USER if scope is SourceScope.USER else None,
        source_key="k", scope=scope, expected_change_interval=DAY, route=route,
        content_hash="h", last_changed_at=NOW, last_ingested_at=NOW,
    )


# Concept 9's freshness spectrum, each row declared as a profile.
@pytest.mark.parametrize(
    ("scope", "interval", "expected"),
    [
        (SourceScope.TENANT, timedelta(seconds=1), IngestionRoute.RAG_ONLY),  # stock prices
        (SourceScope.TENANT, HOUR, IngestionRoute.RAG_ONLY),  # news, social media
        (SourceScope.USER, timedelta(minutes=1), IngestionRoute.MAG),  # session state
        (SourceScope.TENANT, 3 * DAY, IngestionRoute.CAG_WITH_RAG_BACKUP),  # product catalog
        (SourceScope.TENANT, 60 * DAY, IngestionRoute.CAG_WITH_RAG_BACKUP),  # policies
        (SourceScope.TENANT, 3650 * DAY, IngestionRoute.CAG_WITH_RAG_BACKUP),  # textbooks
        # Code repositories change "per commit". One interval can't express the
        # source's main-branch/PR split, so a busy repository routes RAG_ONLY.
        (SourceScope.TENANT, 4 * HOUR, IngestionRoute.RAG_ONLY),
        (SourceScope.USER, 30 * DAY, IngestionRoute.MAG),  # long-term preferences
    ],
)
def test_concept_9_rows_route_as_declared(scope, interval, expected):
    assert route_for(DataSourceProfile("s", scope, interval), POLICY) is expected


def test_exactly_one_boundary_is_stable_and_just_under_is_volatile():
    at = DataSourceProfile("s", SourceScope.TENANT, DAY)
    under = DataSourceProfile("s", SourceScope.TENANT, DAY - timedelta(seconds=1))
    assert route_for(at, POLICY) is IngestionRoute.CAG_WITH_RAG_BACKUP
    assert route_for(under, POLICY) is IngestionRoute.RAG_ONLY


def test_user_scope_routes_to_mag_whatever_its_interval():
    profile = DataSourceProfile("s", SourceScope.USER, 3650 * DAY)
    assert route_for(profile, POLICY) is IngestionRoute.MAG


def test_ttl_is_half_the_interval_by_default_and_none_when_disabled():
    assert cache_ttl(7 * DAY, POLICY) == timedelta(days=3.5)
    assert cache_ttl(7 * DAY, FreshnessPolicy(ttl_factor=None)) is None


def test_observed_interval_needs_two_timestamps_and_averages_uneven_gaps():
    assert observed_change_interval([NOW]) is None
    times = [NOW, NOW + HOUR, NOW + 4 * HOUR]  # gaps of 1h and 3h
    assert observed_change_interval(list(reversed(times))) == 2 * HOUR


def _demote_times(first_gap_before_now: timedelta) -> list[datetime]:
    # Four versions: the one before the last three changes, then three hourly changes.
    start = NOW - first_gap_before_now
    return [start, start + HOUR, start + 2 * HOUR, start + 3 * HOUR]


def test_three_recent_fast_changes_demote_a_cached_source():
    decision = decide_migration(
        _source(IngestionRoute.CAG_WITH_RAG_BACKUP), _demote_times(DAY), NOW, POLICY
    )
    assert decision == MigrationDecision(IngestionRoute.RAG_ONLY, HOUR)


def test_two_changes_are_not_enough_to_demote():
    times = [NOW - 3 * HOUR, NOW - 2 * HOUR, NOW - HOUR]
    source = _source(IngestionRoute.CAG_WITH_RAG_BACKUP)
    assert decide_migration(source, times, NOW, POLICY) is None


def test_a_version_exactly_on_the_window_edge_does_not_demote():
    start = NOW - 3 * DAY
    times = [start, start + DAY, start + 2 * DAY, NOW]  # daily changes: exactly the boundary
    source = _source(IngestionRoute.CAG_WITH_RAG_BACKUP)
    assert decide_migration(source, times, NOW, POLICY) is None


def test_fast_changes_that_ended_before_the_window_do_not_demote():
    times = _demote_times(10 * DAY)  # fast, but ten days ago
    source = _source(IngestionRoute.CAG_WITH_RAG_BACKUP)
    assert decide_migration(source, times, NOW, POLICY) is None


def test_seven_quiet_boundaries_promote_a_rag_only_source_with_the_quiet_span():
    last_change = NOW - 7 * DAY
    decision = decide_migration(
        _source(IngestionRoute.RAG_ONLY), [last_change - HOUR, last_change], NOW, POLICY
    )
    assert decision == MigrationDecision(IngestionRoute.CAG_WITH_RAG_BACKUP, 7 * DAY)


def test_one_second_short_of_seven_quiet_boundaries_does_not_promote():
    last_change = NOW - 7 * DAY + timedelta(seconds=1)
    assert decide_migration(_source(IngestionRoute.RAG_ONLY), [last_change], NOW, POLICY) is None


def test_mag_sources_never_migrate():
    source = _source(IngestionRoute.MAG, SourceScope.USER)
    assert decide_migration(source, _demote_times(DAY), NOW, POLICY) is None
    assert decide_migration(source, [NOW - 30 * DAY], NOW, POLICY) is None


def test_versions_after_now_are_ignored():
    times = [NOW - 8 * DAY, NOW + HOUR]
    decision = decide_migration(_source(IngestionRoute.RAG_ONLY), times, NOW, POLICY)
    assert decision == MigrationDecision(IngestionRoute.CAG_WITH_RAG_BACKUP, 8 * DAY)


def test_source_ids_are_deterministic_and_distinguish_scope():
    tenant_source = source_id_for(TENANT, "prices", None)
    assert tenant_source == source_id_for(TENANT, "prices", None)
    assert tenant_source != source_id_for(TENANT, "prices", USER)
    assert tenant_source != source_id_for(uuid.uuid4(), "prices", None)
```

- [ ] **Step 3: Run both to verify they fail**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_freshness_entities.py tests/unit/test_freshness_router.py -q -p no:cacheprovider`
Expected: collection errors, from `ImportError` on `DataSourceProfile` and `freshness_router`.

- [ ] **Step 4: Append the entities** to `src/orchestration/domain/entities.py`, adding `from datetime import datetime, timedelta` to its imports

```python
class SourceScope(Enum):
    TENANT = "tenant"
    USER = "user"


class IngestionRoute(Enum):
    RAG_ONLY = "rag_only"
    CAG_WITH_RAG_BACKUP = "cag_with_rag_backup"
    MAG = "mag"


@dataclass(frozen=True)
class DataSourceProfile:
    """What a caller declares about a source the first time it is ingested."""

    source_key: str
    scope: SourceScope
    expected_change_interval: timedelta

    def __post_init__(self) -> None:
        if not self.source_key.strip():
            raise ValueError("source_key must not be blank")
        if self.expected_change_interval <= timedelta(0):
            raise ValueError("expected_change_interval must be positive")


@dataclass(frozen=True)
class FreshnessPolicy:
    """The router's disclosed defaults (spec decisions 3, 6, and 7)."""

    volatile_below: timedelta = timedelta(days=1)
    demote_after_changes: int = 3
    promote_after_quiet_multiple: int = 7
    ttl_factor: float | None = 0.5

    def __post_init__(self) -> None:
        if self.volatile_below <= timedelta(0):
            raise ValueError("volatile_below must be positive")
        if self.demote_after_changes < 1:
            raise ValueError("demote_after_changes must be at least 1")
        if self.promote_after_quiet_multiple <= self.demote_after_changes:
            # Otherwise one change on the window's edge could satisfy both rules.
            raise ValueError("promote_after_quiet_multiple must exceed demote_after_changes")
        if self.ttl_factor is not None and self.ttl_factor <= 0:
            raise ValueError("ttl_factor must be positive, or None for no expiry")


@dataclass(frozen=True)
class DataSource:
    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID | None
    source_key: str
    scope: SourceScope
    expected_change_interval: timedelta
    route: IngestionRoute
    content_hash: str
    last_changed_at: datetime
    last_ingested_at: datetime
    cached_until: datetime | None = None


@dataclass(frozen=True)
class IngestionResult:
    source_id: uuid.UUID
    route: IngestionRoute
    changed: bool


@dataclass(frozen=True)
class MigrationDecision:
    to_route: IngestionRoute
    interval: timedelta


@dataclass(frozen=True)
class SourceMigration:
    source_key: str
    from_route: IngestionRoute
    to_route: IngestionRoute
    observed_interval: timedelta
```

- [ ] **Step 5: Append the error** to `src/orchestration/domain/errors.py`

```python
class ScopeMismatch(Exception):
    """A user-scoped source was ingested without a user_id, or a tenant-scoped one with one."""

    def __init__(self, source_key: str, scope_name: str, has_user: bool) -> None:
        super().__init__(
            f"source {source_key!r} is {scope_name}-scoped but was ingested "
            f"{'with' if has_user else 'without'} a user_id"
        )
        self.source_key = source_key
```

- [ ] **Step 6: Create `src/orchestration/domain/freshness_router.py`**

```python
"""Concept 9's Freshness-Aware Data Router, as pure rules: where a source lives, how
long a cached copy is trusted, and when observed change should move it."""
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta

from src.orchestration.domain.entities import (
    DataSource,
    DataSourceProfile,
    FreshnessPolicy,
    IngestionRoute,
    MigrationDecision,
    SourceScope,
)

_SOURCE_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_OID, "orchestration:data_source")


def source_id_for(tenant_id: uuid.UUID, source_key: str, user_id: uuid.UUID | None) -> uuid.UUID:
    """Deterministic, so a retried ingestion replaces the same RAG document and cache
    entry instead of orphaning the previous ones under a fresh id."""
    return uuid.uuid5(_SOURCE_NAMESPACE, f"{tenant_id}:{user_id or ''}:{source_key}")


def route_for(profile: DataSourceProfile, policy: FreshnessPolicy) -> IngestionRoute:
    if profile.scope is SourceScope.USER:
        return IngestionRoute.MAG
    if profile.expected_change_interval < policy.volatile_below:
        return IngestionRoute.RAG_ONLY
    return IngestionRoute.CAG_WITH_RAG_BACKUP


def cache_ttl(interval: timedelta, policy: FreshnessPolicy) -> timedelta | None:
    if policy.ttl_factor is None:
        return None
    return interval * policy.ttl_factor


def observed_change_interval(times: Sequence[datetime]) -> timedelta | None:
    if len(times) < 2:
        return None
    ordered = sorted(times)
    return (ordered[-1] - ordered[0]) / (len(ordered) - 1)


def decide_migration(
    source: DataSource,
    version_times: Sequence[datetime],
    now: datetime,
    policy: FreshnessPolicy,
) -> MigrationDecision | None:
    """version_times holds every version's ingestion time, the first included.

    Demote a cached source when its last demote_after_changes changes, and the
    version before them, all fall strictly inside that many boundaries before now:
    their mean gap is then under the boundary, so the re-learned interval routes
    RAG_ONLY. Promote a RAG_ONLY source once it has gone promote_after_quiet_multiple
    boundaries without a new version. MAG sources never migrate.
    """
    if source.route is IngestionRoute.MAG:
        return None
    history = sorted(t for t in version_times if t <= now)
    if not history:
        return None
    boundary = policy.volatile_below
    if source.route is IngestionRoute.CAG_WITH_RAG_BACKUP:
        needed = policy.demote_after_changes + 1
        recent = history[-needed:]
        if len(recent) < needed or recent[0] <= now - boundary * policy.demote_after_changes:
            return None
        interval = observed_change_interval(recent)
        if interval is None:
            return None
        return MigrationDecision(IngestionRoute.RAG_ONLY, interval)
    quiet = now - history[-1]
    if quiet >= boundary * policy.promote_after_quiet_multiple:
        return MigrationDecision(IngestionRoute.CAG_WITH_RAG_BACKUP, quiet)
    return None
```

- [ ] **Step 7: Run the tests, lint, and types**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_freshness_entities.py tests/unit/test_freshness_router.py -q -p no:cacheprovider`
Expected: all pass.

Then run ruff on the four changed files, and `mypy src`. Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/orchestration/domain/entities.py src/orchestration/domain/errors.py src/orchestration/domain/freshness_router.py tests/unit/test_freshness_entities.py tests/unit/test_freshness_router.py
git commit  # feat: add the freshness router's routing, TTL, and migration rules
```

### Task 2: Ports, fakes, and `CacheWarmedRetrieve.forget`

**Files:**
- Modify: `src/orchestration/domain/ports.py` (append; extend the entities import with `DataSource`)
- Modify: `src/orchestration/application/cache_warmed_retrieve.py` (add `forget`)
- Create: `tests/unit/freshness_fakes.py`
- Test: `tests/unit/test_cache_warmed_retrieve.py` (append), `tests/unit/test_freshness_fakes.py`

**Interfaces:**
- Consumes: `DataSource` (Task 1).
- Produces:
  - `DataSourceRepository`:
    - `get(tenant_id, source_key, user_id) -> DataSource | None`
    - `save(source, changed_content: str | None = None) -> None`
    - `version_times(tenant_id, source_id) -> list[datetime]`
    - `current_content(tenant_id, source_id) -> str | None`
    - `list_sources(tenant_id) -> list[DataSource]`
  - `RagIndex.replace(tenant_id, document_id, title, text) -> None`
  - `ExpiringCache(FrozenCache)`:
    - `preload_until(tenant_id, document_id, content, expires_at: datetime | None) -> None`
    - `renew(tenant_id, document_id, expires_at: datetime | None) -> bool`
  - `SessionFactWriter.record(tenant_id, user_id, fact_key, fact_value) -> None`
  - `CacheWarmedRetrieve.forget(tenant_id, document_id) -> None`
  - The fakes: `FakeDataSourceRepository`, `FakeRagIndex`, `FakeExpiringCache`, `FakeSessionFactWriter`.

- [ ] **Step 1: Write the failing `forget` tests** (append to `tests/unit/test_cache_warmed_retrieve.py`)

```python
def test_forgetting_an_evicted_document_stops_it_shadowing_a_valid_one():
    retriever, cache, _ = _build()
    embedder = FakeBagOfWordsEmbeddingModel()
    query = embedder.embed(_MATCHING_QUERY)
    evicted, valid = uuid.uuid4(), uuid.uuid4()
    other = "the policy desk opens at nine"
    # Precondition, checked rather than assumed: the evicted document is the best candidate.
    assert cosine_similarity(query, embedder.embed(_WARMED_CONTENT)) > cosine_similarity(
        query, embedder.embed(other)
    )
    for document_id, content in ((evicted, _WARMED_CONTENT), (valid, other)):
        cache.preload(_TENANT, document_id, content)
        retriever.note_warmed(_TENANT, document_id, content)
    cache.evict(_TENANT, evicted)

    # The best candidate fails its FrozenCache confirmation, so nothing is returned...
    assert retriever.best_warmed_match(_TENANT, query) is None

    retriever.forget(_TENANT, evicted)

    # ...until it is forgotten, and the valid document is matched instead.
    result = retriever.best_warmed_match(_TENANT, query)
    assert result is not None
    assert result.document_id == valid


def test_forgetting_something_never_warmed_is_a_no_op():
    retriever, _, _ = _build()
    retriever.forget(_TENANT, uuid.uuid4())
    retriever.forget(uuid.uuid4(), uuid.uuid4())
```

- [ ] **Step 2: Run to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_cache_warmed_retrieve.py -q -p no:cacheprovider`
Expected: 2 failures, each with `AttributeError: 'CacheWarmedRetrieve' object has no attribute 'forget'`.

- [ ] **Step 3: Add `forget`** to `CacheWarmedRetrieve`, directly after `note_warmed`

```python
    def forget(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        """Drop a document's warmed memo once its cache entry is gone.

        best_warmed_match confirms only its single best candidate, so an evicted
        document left here would keep shadowing every valid warmed document it
        outscores.
        """
        entries = self._warmed.get(tenant_id)
        if entries is not None:
            entries.pop(document_id, None)
```

- [ ] **Step 4: Append the ports** to `src/orchestration/domain/ports.py`

```python
class DataSourceRepository(ABC):
    """The durable record of every freshness-routed data source and its versions.
    Every method is tenant-scoped; user-scoped sources are also keyed by user_id."""

    @abstractmethod
    async def get(
        self, tenant_id: uuid.UUID, source_key: str, user_id: uuid.UUID | None
    ) -> DataSource | None: ...

    @abstractmethod
    async def save(self, source: DataSource, changed_content: str | None = None) -> None:
        """Insert or update the source by id. When changed_content is given, also record
        it as a new version at source.last_changed_at, in the same transaction."""

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
```

- [ ] **Step 5: Write the fakes' own tests** in `tests/unit/test_freshness_fakes.py`

```python
import uuid
from datetime import UTC, datetime, timedelta

from src.orchestration.domain.entities import DataSource, IngestionRoute, SourceScope
from tests.unit.freshness_fakes import FakeDataSourceRepository, FakeExpiringCache

T0 = datetime(2026, 1, 1, tzinfo=UTC)
TENANT = uuid.uuid4()


def _source(**changes) -> DataSource:
    base = DataSource(
        id=uuid.uuid4(), tenant_id=TENANT, user_id=None, source_key="k",
        scope=SourceScope.TENANT, expected_change_interval=timedelta(days=1),
        route=IngestionRoute.RAG_ONLY, content_hash="h1", last_changed_at=T0, last_ingested_at=T0,
    )
    return DataSource(**{**base.__dict__, **changes})


async def test_the_fake_repository_records_versions_only_for_changed_content():
    repository = FakeDataSourceRepository()
    source = _source()
    await repository.save(source, changed_content="v1")
    await repository.save(source)  # a confirmation: no new version
    later = _source(id=source.id, content_hash="h2", last_changed_at=T0 + timedelta(hours=1))
    await repository.save(later, changed_content="v2")

    assert await repository.version_times(TENANT, source.id) == [T0, T0 + timedelta(hours=1)]
    assert await repository.current_content(TENANT, source.id) == "v2"
    assert await repository.get(TENANT, "k", None) == later
    assert await repository.version_times(uuid.uuid4(), source.id) == []
    assert await repository.list_sources(uuid.uuid4()) == []


def test_the_fake_cache_renews_only_live_entries():
    cache = FakeExpiringCache()
    document = uuid.uuid4()
    assert cache.renew(TENANT, document, T0) is False
    cache.preload_until(TENANT, document, "text", T0)
    assert cache.renew(TENANT, document, T0 + timedelta(days=1)) is True
    assert cache.expiry(TENANT, document) == T0 + timedelta(days=1)
```

- [ ] **Step 6: Create `tests/unit/freshness_fakes.py`**

```python
import uuid
from datetime import datetime

from src.orchestration.domain.entities import CacheHit, DataSource
from src.orchestration.domain.ports import (
    DataSourceRepository,
    ExpiringCache,
    RagIndex,
    SessionFactWriter,
)
from src.orchestration.domain.sync_mixer import content_hash


class FakeDataSourceRepository(DataSourceRepository):
    def __init__(self) -> None:
        self.sources: dict[uuid.UUID, DataSource] = {}
        self.versions: dict[uuid.UUID, list[tuple[datetime, str]]] = {}

    async def get(
        self, tenant_id: uuid.UUID, source_key: str, user_id: uuid.UUID | None
    ) -> DataSource | None:
        return next(
            (
                s for s in self.sources.values()
                if (s.tenant_id, s.source_key, s.user_id) == (tenant_id, source_key, user_id)
            ),
            None,
        )

    async def save(self, source: DataSource, changed_content: str | None = None) -> None:
        self.sources[source.id] = source
        if changed_content is not None:
            self.versions.setdefault(source.id, []).append(
                (source.last_changed_at, changed_content)
            )

    def _visible(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> bool:
        source = self.sources.get(source_id)
        return source is not None and source.tenant_id == tenant_id

    async def version_times(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> list[datetime]:
        if not self._visible(tenant_id, source_id):
            return []
        return sorted(at for at, _ in self.versions.get(source_id, []))

    async def current_content(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> str | None:
        versions = self.versions.get(source_id) if self._visible(tenant_id, source_id) else None
        return max(versions, key=lambda version: version[0])[1] if versions else None

    async def list_sources(self, tenant_id: uuid.UUID) -> list[DataSource]:
        return [s for s in self.sources.values() if s.tenant_id == tenant_id]


class FakeRagIndex(RagIndex):
    def __init__(self) -> None:
        self.replaced: list[tuple[uuid.UUID, uuid.UUID, str, str]] = []

    async def replace(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, title: str, text: str
    ) -> None:
        self.replaced.append((tenant_id, document_id, title, text))


class FakeExpiringCache(ExpiringCache):
    """Records calls and ignores expiry: ExpiringFrozenCache's own tests cover expiry."""

    def __init__(self) -> None:
        self._entries: dict[tuple[uuid.UUID, uuid.UUID], tuple[str, datetime | None]] = {}
        self.preload_calls: list[tuple[uuid.UUID, uuid.UUID, str, datetime | None]] = []
        self.renew_calls: list[tuple[uuid.UUID, uuid.UUID, datetime | None]] = []
        self.evict_calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def preload(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        self.preload_until(tenant_id, document_id, content, None)

    def preload_until(
        self,
        tenant_id: uuid.UUID,
        document_id: uuid.UUID,
        content: str,
        expires_at: datetime | None,
    ) -> None:
        self.preload_calls.append((tenant_id, document_id, content, expires_at))
        self._entries[(tenant_id, document_id)] = (content, expires_at)

    def renew(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, expires_at: datetime | None
    ) -> bool:
        key = (tenant_id, document_id)
        if key not in self._entries:
            return False
        self.renew_calls.append((tenant_id, document_id, expires_at))
        self._entries[key] = (self._entries[key][0], expires_at)
        return True

    def expiry(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> datetime | None:
        return self._entries[(tenant_id, document_id)][1]

    def lookup(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> CacheHit | None:
        entry = self._entries.get((tenant_id, document_id))
        return None if entry is None else CacheHit(content_hash(entry[0]), None)

    def evict(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        self.evict_calls.append((tenant_id, document_id))
        self._entries.pop((tenant_id, document_id), None)

    def contains(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        return (tenant_id, document_id) in self._entries


class FakeSessionFactWriter(SessionFactWriter):
    def __init__(self) -> None:
        self.records: list[tuple[uuid.UUID, uuid.UUID, str, str]] = []

    async def record(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, fact_key: str, fact_value: str
    ) -> None:
        self.records.append((tenant_id, user_id, fact_key, fact_value))
```

- [ ] **Step 7: Run and verify**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_cache_warmed_retrieve.py tests/unit/test_freshness_fakes.py -q -p no:cacheprovider`
Expected: all pass. Then run ruff on the changed files, and `mypy src`.

- [ ] **Step 8: Commit**

```bash
git add src/orchestration/domain/ports.py src/orchestration/application/cache_warmed_retrieve.py tests/unit/freshness_fakes.py tests/unit/test_cache_warmed_retrieve.py tests/unit/test_freshness_fakes.py
git commit  # feat: add the freshness router's ports and let warmed documents be forgotten
```

---

### Task 3: `ExpiringFrozenCache`

**Files:**
- Create: `src/orchestration/infrastructure/expiring_frozen_cache.py`
- Test: `tests/unit/test_expiring_frozen_cache.py`

**Interfaces:**
- Consumes: `ExpiringCache` and `FrozenCache` (Task 2); `FakeFrozenCache` from `tests/unit/orchestration_fakes.py`.
- Produces: `ExpiringFrozenCache(inner: FrozenCache, clock: Callable[[], datetime] = <UTC now>)`, implementing `ExpiringCache`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_expiring_frozen_cache.py`

```python
import uuid
from datetime import UTC, datetime, timedelta

from src.orchestration.infrastructure.expiring_frozen_cache import ExpiringFrozenCache
from tests.unit.orchestration_fakes import FakeFrozenCache

T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
TENANT, OTHER_TENANT, DOC = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _cache() -> tuple[ExpiringFrozenCache, FakeFrozenCache, _Clock]:
    inner, clock = FakeFrozenCache(), _Clock(T0)
    return ExpiringFrozenCache(inner, clock), inner, clock


def test_an_entry_is_served_until_its_expiry_and_absent_from_that_instant():
    cache, inner, clock = _cache()
    cache.preload_until(TENANT, DOC, "text", T0 + HOUR)

    clock.now = T0 + HOUR - timedelta(microseconds=1)
    assert cache.contains(TENANT, DOC)
    assert cache.lookup(TENANT, DOC) is not None

    clock.now = T0 + HOUR
    assert cache.lookup(TENANT, DOC) is None
    assert not cache.contains(TENANT, DOC)
    # Evicted from the wrapped cache the first time the expiry was seen, and only once.
    assert inner.evict_calls == [(TENANT, DOC)]


def test_a_plain_preload_and_a_none_expiry_never_expire():
    cache, _, clock = _cache()
    cache.preload(TENANT, DOC, "text")
    other = uuid.uuid4()
    cache.preload_until(TENANT, other, "text", None)
    clock.now = T0 + timedelta(days=3650)
    assert cache.contains(TENANT, DOC)
    assert cache.contains(TENANT, other)


def test_renewing_a_live_entry_moves_its_expiry():
    cache, _, clock = _cache()
    cache.preload_until(TENANT, DOC, "text", T0 + HOUR)
    assert cache.renew(TENANT, DOC, T0 + 2 * HOUR) is True
    clock.now = T0 + 90 * timedelta(minutes=1)
    assert cache.contains(TENANT, DOC)


def test_renewing_a_missing_or_expired_entry_changes_nothing():
    cache, _, clock = _cache()
    assert cache.renew(TENANT, DOC, T0 + HOUR) is False
    cache.preload_until(TENANT, DOC, "text", T0 + HOUR)
    clock.now = T0 + HOUR
    assert cache.renew(TENANT, DOC, T0 + 5 * HOUR) is False
    assert not cache.contains(TENANT, DOC)


def test_re_preloading_replaces_the_expiry():
    cache, _, clock = _cache()
    cache.preload_until(TENANT, DOC, "v1", T0 + HOUR)
    cache.preload_until(TENANT, DOC, "v2", None)
    clock.now = T0 + 10 * HOUR
    assert cache.contains(TENANT, DOC)


def test_evict_removes_the_entry_and_its_expiry():
    cache, inner, _ = _cache()
    cache.preload_until(TENANT, DOC, "text", T0 + HOUR)
    cache.evict(TENANT, DOC)
    assert not cache.contains(TENANT, DOC)
    assert not inner.contains(TENANT, DOC)


def test_expiry_is_scoped_by_tenant():
    cache, _, clock = _cache()
    cache.preload_until(TENANT, DOC, "text", T0 + HOUR)
    cache.preload_until(OTHER_TENANT, DOC, "text", None)
    clock.now = T0 + HOUR
    assert not cache.contains(TENANT, DOC)
    assert cache.contains(OTHER_TENANT, DOC)
```

- [ ] **Step 2: Run to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_expiring_frozen_cache.py -q -p no:cacheprovider`
Expected: a collection error, `ModuleNotFoundError: src.orchestration.infrastructure.expiring_frozen_cache`.

- [ ] **Step 3: Create `src/orchestration/infrastructure/expiring_frozen_cache.py`**

```python
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

    Thread-safe for the expiry map: CagTier reaches lookup from worker threads.
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
```

- [ ] **Step 4: Run and verify**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_expiring_frozen_cache.py -q -p no:cacheprovider`
Expected: 7 passed. Then run ruff on both files, and `mypy src`.

- [ ] **Step 5: Revert-check the boundary.** Temporarily change `<` to `<=` in `_expired`, and confirm `test_an_entry_is_served_until_its_expiry_and_absent_from_that_instant` fails. Then restore it.

- [ ] **Step 6: Commit**

```bash
git add src/orchestration/infrastructure/expiring_frozen_cache.py tests/unit/test_expiring_frozen_cache.py
git commit  # feat: expire frozen-cache entries at lookup time
```

---

### Task 4: `IngestDataSource`

**Files:**
- Create: `src/orchestration/application/ingest_data_source.py`
- Test: `tests/unit/test_ingest_data_source.py`

**Interfaces:**
- Consumes:
  - Task 1: `route_for`, `cache_ttl`, `source_id_for`, the entities, and `ScopeMismatch`.
  - Task 2: the ports, the fakes, and `CacheWarmedRetrieve.forget`.
- Produces:
  - `RoutePolicy = Callable[[DataSourceProfile, FreshnessPolicy], IngestionRoute]`.
  - `IngestDataSource(repository, rag_index, cache, warmed, fact_writer, *, policy: FreshnessPolicy | None = None, route_policy: RoutePolicy = route_for)`.
  - `async execute(tenant_id, profile, content, now, user_id=None) -> IngestionResult`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_ingest_data_source.py`

```python
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.ingest_data_source import IngestDataSource
from src.orchestration.domain.entities import (
    DataSourceProfile,
    FreshnessPolicy,
    IngestionRoute,
    SourceScope,
)
from src.orchestration.domain.errors import ScopeMismatch
from src.orchestration.domain.freshness_router import source_id_for
from tests.unit.freshness_fakes import (
    FakeDataSourceRepository,
    FakeExpiringCache,
    FakeRagIndex,
    FakeSessionFactWriter,
)
from tests.unit.orchestration_fakes import FakeBagOfWordsEmbeddingModel
from tests.unit.rag_fakes import FakeRetriever

T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)
TENANT, USER = uuid.uuid4(), uuid.uuid4()
PRICES = DataSourceProfile("prices", SourceScope.TENANT, HOUR)
POLICY_DOC = DataSourceProfile("return-policy", SourceScope.TENANT, 7 * DAY)
SIZE = DataSourceProfile("size", SourceScope.USER, 30 * DAY)


def _build(policy: FreshnessPolicy | None = None, route_policy=None):
    repository, rag, cache = FakeDataSourceRepository(), FakeRagIndex(), FakeExpiringCache()
    writer = FakeSessionFactWriter()
    warmed = CacheWarmedRetrieve(FakeBagOfWordsEmbeddingModel(), cache, FakeRetriever(), 0.3)
    kwargs = {"policy": policy}
    if route_policy is not None:
        kwargs["route_policy"] = route_policy
    use_case = IngestDataSource(repository, rag, cache, warmed, writer, **kwargs)
    return use_case, repository, rag, cache, warmed, writer


async def test_a_new_volatile_source_goes_to_rag_only_under_a_deterministic_id():
    use_case, repository, rag, cache, _, writer = _build()

    result = await use_case.execute(TENANT, PRICES, "costs 41 dollars", T0)

    source_id = source_id_for(TENANT, "prices", None)
    assert (result.source_id, result.route, result.changed) == (
        source_id, IngestionRoute.RAG_ONLY, True,
    )
    assert rag.replaced == [(TENANT, source_id, "prices", "costs 41 dollars")]
    assert cache.preload_calls == []
    assert writer.records == []
    assert await repository.version_times(TENANT, source_id) == [T0]


async def test_a_new_stable_source_is_indexed_in_rag_and_left_for_the_batch_to_cache():
    use_case, repository, rag, cache, _, _ = _build()

    result = await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0)

    assert result.route is IngestionRoute.CAG_WITH_RAG_BACKUP
    assert rag.replaced == [(TENANT, result.source_id, "return-policy", "forty-five days")]
    assert cache.preload_calls == []
    stored = await repository.get(TENANT, "return-policy", None)
    assert stored is not None
    assert stored.cached_until is None


async def test_a_user_scoped_source_is_written_to_mag_and_nowhere_else():
    use_case, _, rag, cache, _, writer = _build()

    result = await use_case.execute(TENANT, SIZE, "wears size 10", T0, user_id=USER)

    assert result.route is IngestionRoute.MAG
    assert writer.records == [(TENANT, USER, "size", "wears size 10")]
    assert rag.replaced == []
    assert cache.preload_calls == []


async def test_identical_content_is_a_confirmation_not_a_change():
    use_case, repository, rag, _, _, _ = _build()
    await use_case.execute(TENANT, PRICES, "costs 41 dollars", T0)

    result = await use_case.execute(TENANT, PRICES, "costs 41 dollars", T0 + HOUR)

    assert result.changed is False
    assert len(rag.replaced) == 1
    stored = await repository.get(TENANT, "prices", None)
    assert stored is not None
    assert (stored.last_changed_at, stored.last_ingested_at) == (T0, T0 + HOUR)
    assert await repository.version_times(TENANT, result.source_id) == [T0]


async def test_a_change_replaces_the_rag_text_and_records_a_version():
    use_case, repository, rag, _, _, _ = _build()
    await use_case.execute(TENANT, PRICES, "costs 41 dollars", T0)

    result = await use_case.execute(TENANT, PRICES, "costs 43 dollars", T0 + HOUR)

    assert result.changed is True
    assert rag.replaced[-1][3] == "costs 43 dollars"
    assert await repository.version_times(TENANT, result.source_id) == [T0, T0 + HOUR]
    assert await repository.current_content(TENANT, result.source_id) == "costs 43 dollars"


async def test_a_change_to_a_cached_source_evicts_and_forgets_it_at_once():
    use_case, repository, _, cache, warmed, _ = _build()
    first = await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0)
    cache.preload_until(TENANT, first.source_id, "forty-five days", T0 + DAY)
    warmed.note_warmed(TENANT, first.source_id, "forty-five days")

    await use_case.execute(TENANT, POLICY_DOC, "sixty days", T0 + HOUR)

    assert (TENANT, first.source_id) in cache.evict_calls
    # Re-preloading the old text proves the memo is gone: nothing matches it now.
    cache.preload(TENANT, first.source_id, "forty-five days")
    embedding = FakeBagOfWordsEmbeddingModel().embed("forty-five days")
    assert warmed.best_warmed_match(TENANT, embedding) is None
    stored = await repository.get(TENANT, "return-policy", None)
    assert stored is not None
    assert stored.cached_until is None


async def test_a_confirmation_renews_a_cached_entry_for_one_ttl_from_now():
    use_case, repository, _, cache, _, _ = _build()
    first = await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0)
    cache.preload_until(TENANT, first.source_id, "forty-five days", T0 + DAY)

    await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0 + 2 * DAY)

    expected = T0 + 2 * DAY + timedelta(days=3.5)  # 7-day interval x 0.5
    assert cache.expiry(TENANT, first.source_id) == expected
    stored = await repository.get(TENANT, "return-policy", None)
    assert stored is not None
    assert stored.cached_until == expected


async def test_a_confirmation_of_an_uncached_source_caches_nothing():
    use_case, repository, _, cache, _, _ = _build()
    await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0)

    await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0 + DAY)

    assert cache.preload_calls == []
    assert cache.renew_calls == []
    stored = await repository.get(TENANT, "return-policy", None)
    assert stored is not None
    assert stored.cached_until is None


async def test_with_ttl_disabled_a_confirmation_renews_with_no_expiry():
    use_case, _, _, cache, _, _ = _build(policy=FreshnessPolicy(ttl_factor=None))
    first = await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0)
    cache.preload_until(TENANT, first.source_id, "forty-five days", T0 + DAY)

    await use_case.execute(TENANT, POLICY_DOC, "forty-five days", T0 + HOUR)

    assert cache.renew_calls == [(TENANT, first.source_id, None)]


@pytest.mark.parametrize(
    ("profile", "user_id"), [(SIZE, None), (PRICES, USER)]
)
async def test_scope_and_user_id_must_agree(profile, user_id):
    use_case, repository, rag, _, _, writer = _build()
    with pytest.raises(ScopeMismatch):
        await use_case.execute(TENANT, profile, "text", T0, user_id=user_id)
    assert await repository.list_sources(TENANT) == []
    assert (rag.replaced, writer.records) == ([], [])


async def test_an_injected_route_policy_can_cache_a_user_scoped_source():
    # The measurement's "cache everything" baseline: the same code path, one route.
    use_case, _, rag, _, _, writer = _build(
        route_policy=lambda profile, policy: IngestionRoute.CAG_WITH_RAG_BACKUP
    )
    result = await use_case.execute(TENANT, SIZE, "wears size 10", T0, user_id=USER)
    assert result.route is IngestionRoute.CAG_WITH_RAG_BACKUP
    assert rag.replaced == [(TENANT, result.source_id, "size", "wears size 10")]
    assert writer.records == []


async def test_a_route_policy_cannot_send_tenant_data_to_mag():
    use_case, _, _, _, _, _ = _build(route_policy=lambda profile, policy: IngestionRoute.MAG)
    with pytest.raises(ValueError):
        await use_case.execute(TENANT, PRICES, "text", T0)


async def test_later_ingestions_keep_the_stored_interval():
    use_case, repository, _, _, _, _ = _build()
    await use_case.execute(TENANT, PRICES, "v1", T0)
    redeclared = DataSourceProfile("prices", SourceScope.TENANT, 30 * DAY)

    result = await use_case.execute(TENANT, redeclared, "v2", T0 + HOUR)

    assert result.route is IngestionRoute.RAG_ONLY
    stored = await repository.get(TENANT, "prices", None)
    assert stored is not None
    assert stored.expected_change_interval == HOUR


async def test_the_same_key_in_two_tenants_is_two_sources():
    use_case, repository, _, _, _, _ = _build()
    other = uuid.uuid4()
    a = await use_case.execute(TENANT, PRICES, "v1", T0)
    b = await use_case.execute(other, PRICES, "v1", T0)
    assert a.source_id != b.source_id
    assert len(await repository.list_sources(TENANT)) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_ingest_data_source.py -q -p no:cacheprovider`
Expected: a collection error, `ModuleNotFoundError: src.orchestration.application.ingest_data_source`.

- [ ] **Step 3: Create `src/orchestration/application/ingest_data_source.py`**

```python
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
```

- [ ] **Step 4: Run and verify**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_ingest_data_source.py -q -p no:cacheprovider`
Expected: 15 passed. Then run ruff on both files, and `mypy src`.

- [ ] **Step 5: Revert-check eviction.** Temporarily delete the `self._warmed.forget(...)` line, and confirm `test_a_change_to_a_cached_source_evicts_and_forgets_it_at_once` fails. Then restore it.

- [ ] **Step 6: Commit**

```bash
git add src/orchestration/application/ingest_data_source.py tests/unit/test_ingest_data_source.py
git commit  # feat: route data sources at ingestion by expected change frequency
```

---

### Task 5: `RefreshCachedSources` and `ReviewSourceFreshness`

**Files:**
- Create: `src/orchestration/application/refresh_cached_sources.py`
- Create: `src/orchestration/application/review_source_freshness.py`
- Test: `tests/unit/test_refresh_cached_sources.py`
- Test: `tests/unit/test_review_source_freshness.py`

**Interfaces:**
- Consumes:
  - Task 1: `cache_ttl`, `decide_migration`, and the entities.
  - Task 2: the ports and the fakes.
- Produces:
  - `RefreshCachedSources(repository, cache, warmed, *, policy=None).run(tenant_id, now) -> list[str]`, returning the source keys it pre-loaded.
  - `ReviewSourceFreshness(repository, cache, warmed, *, policy=None).run(tenant_id, now) -> list[SourceMigration]`.

- [ ] **Step 1: Write the shared test helper** at the top of both new test files

```python
import uuid
from datetime import UTC, datetime, timedelta

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.domain.entities import (
    DataSource,
    FreshnessPolicy,
    IngestionRoute,
    SourceScope,
)
from tests.unit.freshness_fakes import FakeDataSourceRepository, FakeExpiringCache
from tests.unit.orchestration_fakes import FakeBagOfWordsEmbeddingModel
from tests.unit.rag_fakes import FakeRetriever

T0 = datetime(2026, 1, 20, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)
TENANT = uuid.uuid4()


def _fixtures():
    repository, cache = FakeDataSourceRepository(), FakeExpiringCache()
    warmed = CacheWarmedRetrieve(FakeBagOfWordsEmbeddingModel(), cache, FakeRetriever(), 0.3)
    return repository, cache, warmed


async def _seed(
    repository: FakeDataSourceRepository,
    key: str,
    route: IngestionRoute,
    versions: list[tuple[datetime, str]],
    *,
    interval: timedelta = 7 * DAY,
    last_ingested_at: datetime | None = None,
    scope: SourceScope = SourceScope.TENANT,
    tenant_id: uuid.UUID = TENANT,
) -> DataSource:
    source = None
    for at, text in versions:
        source = DataSource(
            id=uuid.uuid5(uuid.NAMESPACE_OID, f"{tenant_id}:{key}"), tenant_id=tenant_id,
            user_id=uuid.uuid4() if scope is SourceScope.USER else None, source_key=key,
            scope=scope, expected_change_interval=interval, route=route, content_hash=text,
            last_changed_at=at, last_ingested_at=last_ingested_at or at,
        )
        await repository.save(source, changed_content=text)
    assert source is not None
    return source
```

- [ ] **Step 2: Write the failing refresh tests** in `tests/unit/test_refresh_cached_sources.py`, below the helper

```python
from src.orchestration.application.refresh_cached_sources import RefreshCachedSources


async def test_an_uncached_confirmed_stable_source_is_preloaded_until_one_ttl_after_confirmation():
    repository, cache, warmed = _fixtures()
    source = await _seed(
        repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP,
        [(T0 - 10 * DAY, "forty-five days")], last_ingested_at=T0 - DAY,
    )

    preloaded = await RefreshCachedSources(repository, cache, warmed).run(TENANT, T0)

    expires_at = T0 - DAY + timedelta(days=3.5)
    assert preloaded == ["policy"]
    assert cache.preload_calls == [(TENANT, source.id, "forty-five days", expires_at)]
    match = warmed.best_warmed_match(TENANT, FakeBagOfWordsEmbeddingModel().embed("forty-five days"))
    assert match is not None
    assert match.document_id == source.id
    stored = await repository.get(TENANT, "policy", None)
    assert stored is not None
    assert stored.cached_until == expires_at


async def test_an_already_cached_source_is_not_preloaded_again():
    repository, cache, warmed = _fixtures()
    source = await _seed(repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP, [(T0, "v1")])
    cache.preload_until(TENANT, source.id, "v1", None)
    cache.preload_calls.clear()

    assert await RefreshCachedSources(repository, cache, warmed).run(TENANT, T0) == []
    assert cache.preload_calls == []


async def test_a_source_unconfirmed_for_longer_than_its_ttl_stays_on_rag():
    repository, cache, warmed = _fixtures()
    await _seed(
        repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP, [(T0 - 10 * DAY, "v1")],
        last_ingested_at=T0 - timedelta(days=3.5),  # exactly one TTL ago: expired
    )
    assert await RefreshCachedSources(repository, cache, warmed).run(TENANT, T0) == []


async def test_rag_only_and_mag_sources_are_never_preloaded():
    repository, cache, warmed = _fixtures()
    await _seed(repository, "prices", IngestionRoute.RAG_ONLY, [(T0, "v1")])
    await _seed(repository, "size", IngestionRoute.MAG, [(T0, "v1")], scope=SourceScope.USER)
    assert await RefreshCachedSources(repository, cache, warmed).run(TENANT, T0) == []


async def test_with_ttl_disabled_age_never_blocks_a_preload():
    repository, cache, warmed = _fixtures()
    source = await _seed(
        repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP, [(T0 - 400 * DAY, "v1")]
    )
    refresh = RefreshCachedSources(repository, cache, warmed, policy=FreshnessPolicy(ttl_factor=None))
    assert await refresh.run(TENANT, T0) == ["policy"]
    assert cache.preload_calls == [(TENANT, source.id, "v1", None)]


async def test_another_tenants_sources_are_untouched():
    repository, cache, warmed = _fixtures()
    await _seed(
        repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP, [(T0, "v1")], tenant_id=uuid.uuid4()
    )
    assert await RefreshCachedSources(repository, cache, warmed).run(TENANT, T0) == []
```

- [ ] **Step 3: Write the failing review tests** in `tests/unit/test_review_source_freshness.py`, below the helper

```python
from src.orchestration.application.review_source_freshness import ReviewSourceFreshness
from src.orchestration.domain.entities import SourceMigration


async def test_a_cached_source_changing_fast_is_demoted_evicted_and_forgotten():
    repository, cache, warmed = _fixtures()
    start = T0 - 4 * HOUR
    source = await _seed(
        repository, "catalog", IngestionRoute.CAG_WITH_RAG_BACKUP,
        [(start + i * HOUR, f"v{i}") for i in range(4)],
    )
    cache.preload_until(TENANT, source.id, "v3", None)
    warmed.note_warmed(TENANT, source.id, "v3")

    migrations = await ReviewSourceFreshness(repository, cache, warmed).run(TENANT, T0)

    assert migrations == [
        SourceMigration("catalog", IngestionRoute.CAG_WITH_RAG_BACKUP, IngestionRoute.RAG_ONLY, HOUR)
    ]
    assert (TENANT, source.id) in cache.evict_calls
    assert warmed.best_warmed_match(TENANT, FakeBagOfWordsEmbeddingModel().embed("v3")) is None
    stored = await repository.get(TENANT, "catalog", None)
    assert stored is not None
    assert (stored.route, stored.expected_change_interval, stored.cached_until) == (
        IngestionRoute.RAG_ONLY, HOUR, None,
    )


async def test_a_quiet_rag_only_source_is_promoted_without_a_preload():
    repository, cache, warmed = _fixtures()
    await _seed(repository, "sale", IngestionRoute.RAG_ONLY, [(T0 - 8 * DAY, "v1")], interval=HOUR)

    migrations = await ReviewSourceFreshness(repository, cache, warmed).run(TENANT, T0)

    assert migrations == [
        SourceMigration("sale", IngestionRoute.RAG_ONLY, IngestionRoute.CAG_WITH_RAG_BACKUP, 8 * DAY)
    ]
    assert cache.preload_calls == []
    stored = await repository.get(TENANT, "sale", None)
    assert stored is not None
    assert (stored.route, stored.expected_change_interval) == (
        IngestionRoute.CAG_WITH_RAG_BACKUP, 8 * DAY,
    )


async def test_user_scoped_sources_are_never_reviewed():
    repository, cache, warmed = _fixtures()
    start = T0 - 4 * HOUR
    await _seed(
        repository, "size", IngestionRoute.CAG_WITH_RAG_BACKUP,
        [(start + i * HOUR, f"v{i}") for i in range(4)], scope=SourceScope.USER,
    )
    assert await ReviewSourceFreshness(repository, cache, warmed).run(TENANT, T0) == []


async def test_a_source_with_no_migration_is_left_exactly_as_stored():
    repository, cache, warmed = _fixtures()
    source = await _seed(
        repository, "policy", IngestionRoute.CAG_WITH_RAG_BACKUP, [(T0 - DAY, "v1")]
    )
    assert await ReviewSourceFreshness(repository, cache, warmed).run(TENANT, T0) == []
    assert await repository.get(TENANT, "policy", None) == source
    assert cache.evict_calls == []
```

- [ ] **Step 4: Run to verify both fail**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_refresh_cached_sources.py tests/unit/test_review_source_freshness.py -q -p no:cacheprovider`
Expected: collection errors, `ModuleNotFoundError` for both modules.

- [ ] **Step 5: Create `src/orchestration/application/refresh_cached_sources.py`**

```python
import dataclasses
import uuid
from datetime import datetime

from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.domain.entities import FreshnessPolicy, IngestionRoute
from src.orchestration.domain.freshness_router import cache_ttl
from src.orchestration.domain.ports import DataSourceRepository, ExpiringCache


class RefreshCachedSources:
    """The batch pre-load for CAG_WITH_RAG_BACKUP sources, driven on whatever
    cadence a caller chooses.

    A source is pre-loaded only if it isn't cached and its content was confirmed by
    an ingestion within its TTL. The entry then expires one TTL after that
    confirmation, not after the pre-load. Re-pre-loading content the feed hasn't
    confirmed would re-serve exactly the stale copy the TTL exists to stop (spec
    decision 6); such a source stays on RAG until an ingestion confirms it.
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
```

- [ ] **Step 6: Create `src/orchestration/application/review_source_freshness.py`**

```python
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

    A demotion to RAG_ONLY evicts and forgets the cached copy at once. A promotion
    only changes the route and interval, and the next RefreshCachedSources run
    pre-loads the source. Either way, the observed interval replaces the declared
    one, so later routing and TTLs follow the source's real behaviour. User-scoped
    sources are never reviewed: frequency can't move data into or out of a user's
    own store.
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
                SourceMigration(source.source_key, source.route, decision.to_route, decision.interval)
            )
        return migrations
```

- [ ] **Step 7: Run and verify**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_refresh_cached_sources.py tests/unit/test_review_source_freshness.py -q -p no:cacheprovider`
Expected: 10 passed. Then run the whole unit suite, ruff on the changed files, and `mypy src`.

- [ ] **Step 8: Commit**

```bash
git add src/orchestration/application/refresh_cached_sources.py src/orchestration/application/review_source_freshness.py tests/unit/test_refresh_cached_sources.py tests/unit/test_review_source_freshness.py
git commit  # feat: batch-preload confirmed stable sources and migrate drifting ones
```

---

### Task 6: Migration 0006 and `PostgresDataSourceRepository`

**Files:**
- Create: `alembic/versions/0006_data_sources.py`
- Create: `src/orchestration/infrastructure/postgres_data_source_repository.py`
- Modify: `tests/integration/test_migration.py` (append)
- Test: `tests/integration/test_postgres_data_source_repository.py`

**Interfaces:**
- Consumes: `DataSource`, `SourceScope`, and `IngestionRoute` (Task 1); `DataSourceRepository` (Task 2).
- Produces: the `data_sources` and `data_source_versions` tables, and `PostgresDataSourceRepository(sessionmaker: async_sessionmaker[AsyncSession])`.

- [ ] **Step 1: Write the failing schema tests** (append to `tests/integration/test_migration.py`, following its one-test-per-RLS-table convention)

```python
async def test_data_source_tables_exist_with_rls_enabled_and_forced(db_session):
    result = await db_session.execute(
        text(
            "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
            "WHERE relname IN ('data_sources', 'data_source_versions')"
        )
    )
    flags = {row.relname: (row.relrowsecurity, row.relforcerowsecurity) for row in result}
    assert flags == {"data_sources": (True, True), "data_source_versions": (True, True)}


async def test_tenant_isolation_policy_exists_on_both_data_source_tables(db_session):
    result = await db_session.execute(
        text(
            "SELECT tablename FROM pg_policies WHERE policyname = 'tenant_isolation' "
            "AND tablename IN ('data_sources', 'data_source_versions')"
        )
    )
    assert {row.tablename for row in result} == {"data_sources", "data_source_versions"}
```

- [ ] **Step 2: Write the failing repository tests** in `tests/integration/test_postgres_data_source_repository.py`

```python
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.orchestration.domain.entities import DataSource, IngestionRoute, SourceScope
from src.orchestration.domain.freshness_router import source_id_for
from src.orchestration.infrastructure.postgres_data_source_repository import (
    PostgresDataSourceRepository,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)


async def _create_user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id) "
            "VALUES (:id, :email, :hashed_password, :tenant_id)"
        ),
        {"id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
         "tenant_id": tenant_id},
    )
    await db_session.commit()
    return user_id


def _source(tenant_id: uuid.UUID, key: str = "policy", user_id: uuid.UUID | None = None,
            **changes) -> DataSource:
    base = dict(
        id=source_id_for(tenant_id, key, user_id), tenant_id=tenant_id, user_id=user_id,
        source_key=key, scope=SourceScope.USER if user_id else SourceScope.TENANT,
        expected_change_interval=7 * DAY + 90 * timedelta(seconds=1),
        route=IngestionRoute.MAG if user_id else IngestionRoute.CAG_WITH_RAG_BACKUP,
        content_hash="h1", last_changed_at=T0, last_ingested_at=T0, cached_until=None,
    )
    return DataSource(**{**base, **changes})


def _repository(db_session) -> PostgresDataSourceRepository:
    return PostgresDataSourceRepository(get_sessionmaker(db_session.bind))


async def test_a_source_round_trips_with_its_interval_and_nullable_fields(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    source = _source(tenant_id)
    await repository.save(source, changed_content="forty-five days")
    assert await repository.get(tenant_id, "policy", None) == source


async def test_changed_content_becomes_versions_and_confirmations_do_not(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    await repository.save(_source(tenant_id), changed_content="v1")
    await repository.save(_source(tenant_id, last_ingested_at=T0 + HOUR))  # confirmation
    changed = _source(tenant_id, content_hash="h2", last_changed_at=T0 + DAY,
                      last_ingested_at=T0 + DAY)
    await repository.save(changed, changed_content="v2")

    source_id = changed.id
    assert await repository.version_times(tenant_id, source_id) == [T0, T0 + DAY]
    assert await repository.current_content(tenant_id, source_id) == "v2"


async def test_saving_again_updates_route_interval_and_cache_expiry(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    await repository.save(_source(tenant_id), changed_content="v1")
    migrated = _source(tenant_id, route=IngestionRoute.RAG_ONLY, expected_change_interval=HOUR,
                       cached_until=T0 + 3 * DAY)
    await repository.save(migrated)
    assert await repository.get(tenant_id, "policy", None) == migrated
    assert await repository.list_sources(tenant_id) == [migrated]


async def test_a_user_source_and_a_tenant_source_with_one_key_are_distinct(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    tenant_source, user_source = _source(tenant_id, "size"), _source(tenant_id, "size", user_id)
    await repository.save(tenant_source, changed_content="t")
    await repository.save(user_source, changed_content="u")

    assert await repository.get(tenant_id, "size", None) == tenant_source
    assert await repository.get(tenant_id, "size", user_id) == user_source
    assert await repository.current_content(tenant_id, user_source.id) == "u"


async def test_another_tenant_can_neither_see_list_nor_read_versions(db_session):
    repository, tenant_id, other = _repository(db_session), uuid.uuid4(), uuid.uuid4()
    source = _source(tenant_id)
    await repository.save(source, changed_content="v1")
    assert await repository.get(other, "policy", None) is None
    assert await repository.list_sources(other) == []
    assert await repository.version_times(other, source.id) == []
    assert await repository.current_content(other, source.id) is None


@pytest.mark.parametrize(
    ("scope", "with_user", "interval"),
    [("user", False, "1 hour"), ("tenant", True, "1 hour"), ("tenant", False, "0 seconds")],
)
async def test_the_schema_refuses_inconsistent_rows(db_session, scope, with_user, interval):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id) if with_user else None
    await set_tenant_context(db_session, tenant_id)
    with pytest.raises(IntegrityError):
        await db_session.execute(
            text(
                "INSERT INTO data_sources (id, tenant_id, user_id, source_key, scope, "
                "expected_change_interval, route, content_hash, last_changed_at, "
                "last_ingested_at) VALUES (:id, :tenant_id, :user_id, 'k', :scope, "
                "CAST(:interval AS interval), 'rag_only', 'h', now(), now())"
            ),
            {"id": uuid.uuid4(), "tenant_id": tenant_id, "user_id": user_id, "scope": scope,
             "interval": interval},
        )
    await db_session.rollback()


async def test_the_schema_refuses_a_second_tenant_source_with_the_same_key(db_session):
    repository, tenant_id = _repository(db_session), uuid.uuid4()
    await repository.save(_source(tenant_id), changed_content="v1")
    duplicate = _source(tenant_id, id=uuid.uuid4())
    with pytest.raises(IntegrityError):
        await repository.save(duplicate)
```

- [ ] **Step 3: Run to verify they fail**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_migration.py tests/integration/test_postgres_data_source_repository.py -q -p no:cacheprovider`
Expected: the two new migration tests fail (no rows), and the repository file fails to import.

- [ ] **Step 4: Create `alembic/versions/0006_data_sources.py`**

```python
"""freshness-aware data router: data_sources and data_source_versions

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-13

"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

_TABLES = ("data_sources", "data_source_versions")


def upgrade() -> None:
    op.create_table(
        "data_sources",
        # Not server-defaulted: ids are derived from (tenant, user, key) so a retried
        # ingestion replaces the same RAG document (freshness_router.source_id_for).
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True
        ),
        sa.Column("source_key", sa.String, nullable=False),
        sa.Column("scope", sa.String, nullable=False),
        sa.Column("expected_change_interval", sa.Interval, nullable=False),
        sa.Column("route", sa.String, nullable=False),
        sa.Column("content_hash", sa.String, nullable=False),
        sa.Column("last_changed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cached_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.CheckConstraint("scope IN ('tenant', 'user')", name="ck_data_sources_scope"),
        sa.CheckConstraint(
            "route IN ('rag_only', 'cag_with_rag_backup', 'mag')", name="ck_data_sources_route"
        ),
        sa.CheckConstraint(
            "expected_change_interval > interval '0'", name="ck_data_sources_interval_positive"
        ),
        # A user-scoped source belongs to exactly one user; a tenant-scoped one to none.
        sa.CheckConstraint(
            "(scope = 'user') = (user_id IS NOT NULL)", name="ck_data_sources_scope_user"
        ),
    )
    # NULLS NOT DISTINCT (PostgreSQL 15+): a tenant-scoped key is unique per tenant even
    # though its user_id is NULL, while each user can hold a source under the same key.
    op.execute(
        "CREATE UNIQUE INDEX uq_data_sources_tenant_user_key "
        "ON data_sources (tenant_id, user_id, source_key) NULLS NOT DISTINCT"
    )
    op.create_index("ix_data_sources_tenant_id", "data_sources", ["tenant_id"])

    op.create_table(
        "data_source_versions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "data_source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("data_sources.id"),
            nullable=False,
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content_hash", sa.String, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_data_source_versions_source_ingested",
        "data_source_versions",
        ["data_source_id", "ingested_at"],
    )
    op.create_index("ix_data_source_versions_tenant_id", "data_source_versions", ["tenant_id"])

    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
            USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
            """
        )
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON data_sources, data_source_versions TO app_user")


def downgrade() -> None:
    op.execute("REVOKE ALL ON data_sources, data_source_versions FROM app_user")
    op.drop_table("data_source_versions")
    op.drop_table("data_sources")
```

- [ ] **Step 5: Create `src/orchestration/infrastructure/postgres_data_source_repository.py`**

```python
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.infrastructure.db import set_tenant_context
from src.orchestration.domain.entities import DataSource, IngestionRoute, SourceScope
from src.orchestration.domain.ports import DataSourceRepository

_COLUMNS = (
    "id, tenant_id, user_id, source_key, scope, expected_change_interval, route, "
    "content_hash, last_changed_at, last_ingested_at, cached_until"
)


def _to_source(row: Any) -> DataSource:
    return DataSource(
        id=row.id,
        tenant_id=row.tenant_id,
        user_id=row.user_id,
        source_key=row.source_key,
        scope=SourceScope(row.scope),
        expected_change_interval=row.expected_change_interval,
        route=IngestionRoute(row.route),
        content_hash=row.content_hash,
        last_changed_at=row.last_changed_at,
        last_ingested_at=row.last_ingested_at,
        cached_until=row.cached_until,
    )


class PostgresDataSourceRepository(DataSourceRepository):
    """data_sources and data_source_versions, under tenant_isolation RLS.

    Each call opens and commits its own short transaction and sets the tenant
    context inside it, as PostgresSessionBudgetRecorder does. Ingestion, refresh,
    and review each own their unit of work, and a source and its new version are
    written together or not at all. Queries also filter on tenant_id explicitly,
    so a mistake in context-setting can't widen a read.
    """

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def get(
        self, tenant_id: uuid.UUID, source_key: str, user_id: uuid.UUID | None
    ) -> DataSource | None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            row = (
                await session.execute(
                    text(
                        f"SELECT {_COLUMNS} FROM data_sources WHERE tenant_id = :tenant_id "
                        "AND source_key = :source_key "
                        "AND user_id IS NOT DISTINCT FROM CAST(:user_id AS uuid)"
                    ),
                    {"tenant_id": tenant_id, "source_key": source_key, "user_id": user_id},
                )
            ).first()
        return None if row is None else _to_source(row)

    async def save(self, source: DataSource, changed_content: str | None = None) -> None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, source.tenant_id)
            await session.execute(
                text(
                    f"INSERT INTO data_sources ({_COLUMNS}) VALUES (:id, :tenant_id, :user_id, "
                    ":source_key, :scope, :expected_change_interval, :route, :content_hash, "
                    ":last_changed_at, :last_ingested_at, :cached_until) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "expected_change_interval = EXCLUDED.expected_change_interval, "
                    "route = EXCLUDED.route, content_hash = EXCLUDED.content_hash, "
                    "last_changed_at = EXCLUDED.last_changed_at, "
                    "last_ingested_at = EXCLUDED.last_ingested_at, "
                    "cached_until = EXCLUDED.cached_until, updated_at = now()"
                ),
                {
                    "id": source.id,
                    "tenant_id": source.tenant_id,
                    "user_id": source.user_id,
                    "source_key": source.source_key,
                    "scope": source.scope.value,
                    "expected_change_interval": source.expected_change_interval,
                    "route": source.route.value,
                    "content_hash": source.content_hash,
                    "last_changed_at": source.last_changed_at,
                    "last_ingested_at": source.last_ingested_at,
                    "cached_until": source.cached_until,
                },
            )
            if changed_content is not None:
                await session.execute(
                    text(
                        "INSERT INTO data_source_versions "
                        "(data_source_id, tenant_id, content_hash, content, ingested_at) "
                        "VALUES (:source_id, :tenant_id, :content_hash, :content, :ingested_at)"
                    ),
                    {
                        "source_id": source.id,
                        "tenant_id": source.tenant_id,
                        "content_hash": source.content_hash,
                        "content": changed_content,
                        "ingested_at": source.last_changed_at,
                    },
                )

    async def version_times(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> list[datetime]:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            result = await session.execute(
                text(
                    "SELECT ingested_at FROM data_source_versions WHERE tenant_id = :tenant_id "
                    "AND data_source_id = :source_id ORDER BY ingested_at"
                ),
                {"tenant_id": tenant_id, "source_id": source_id},
            )
            return [row.ingested_at for row in result]

    async def current_content(self, tenant_id: uuid.UUID, source_id: uuid.UUID) -> str | None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            value = (
                await session.execute(
                    text(
                        "SELECT content FROM data_source_versions WHERE tenant_id = :tenant_id "
                        "AND data_source_id = :source_id ORDER BY ingested_at DESC LIMIT 1"
                    ),
                    {"tenant_id": tenant_id, "source_id": source_id},
                )
            ).scalar_one_or_none()
        return None if value is None else str(value)

    async def list_sources(self, tenant_id: uuid.UUID) -> list[DataSource]:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            result = await session.execute(
                text(
                    f"SELECT {_COLUMNS} FROM data_sources WHERE tenant_id = :tenant_id "
                    "ORDER BY source_key, user_id NULLS FIRST"
                ),
                {"tenant_id": tenant_id},
            )
            return [_to_source(row) for row in result]
```

- [ ] **Step 6: Run and verify**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_migration.py tests/integration/test_postgres_data_source_repository.py -q -p no:cacheprovider`
Expected: all pass. If asyncpg can't infer a NULL parameter's type in the INSERT, wrap that parameter in `CAST(... AS <type>)`, the same fix `get` already applies to `user_id`. Record the change in the execution notes. Then run ruff on the changed files, and `mypy src`.

- [ ] **Step 7: Commit**

```bash
git add alembic/versions/0006_data_sources.py src/orchestration/infrastructure/postgres_data_source_repository.py tests/integration/test_migration.py tests/integration/test_postgres_data_source_repository.py
git commit  # feat: persist data sources and their version history under RLS
```

---

### Task 7: RAG deletes and `ChunkedRagIndex`

**Files:**
- Modify: `src/rag/infrastructure/qdrant_vector_store.py` (add `delete_document`)
- Modify: `src/rag/infrastructure/postgres_document_repository.py` (add `delete_document`)
- Create: `src/orchestration/infrastructure/chunked_rag_index.py`
- Test: `tests/integration/test_chunked_rag_index.py`

**Interfaces:**
- Consumes: `RagIndex` (Task 2); the existing `QdrantVectorStore`, `PostgresDocumentRepository`, `Chunker`, `EmbeddingModel`, and `set_tenant_context`.
- Produces:
  - `QdrantVectorStore.delete_document(document_id, tenant_id) -> None`
  - `PostgresDocumentRepository.delete_document(document_id, tenant_id) -> None`
  - `ChunkedRagIndex(sessionmaker, vector_store: QdrantVectorStore, chunker: Chunker, embedder: EmbeddingModel)`

- [ ] **Step 1: Write the failing tests** in `tests/integration/test_chunked_rag_index.py`

```python
import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.orchestration.infrastructure.chunked_rag_index import ChunkedRagIndex
from src.rag.domain.entities import Chunk, Document
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore

V1 = " ".join(["The return window for unopened items is forty-five days."] * 6)
V2 = " ".join(["The return window for unopened items is sixty days."] * 6)


async def _stores(db_session, qdrant_url, embedding_model):
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    chunker = FixedSizeChunker(chunk_size_tokens=16, overlap_ratio=0.0)
    index = ChunkedRagIndex(get_sessionmaker(db_session.bind), vector_store, chunker, embedding_model)
    return index, vector_store


async def _postgres_chunks(db_session, tenant_id, document_id) -> list[str]:
    await set_tenant_context(db_session, tenant_id)
    result = await db_session.execute(
        text("SELECT content FROM chunks WHERE document_id = :id"), {"id": document_id}
    )
    return [row.content for row in result]


async def test_replacing_a_source_leaves_only_its_new_text_in_both_stores(
    db_session, qdrant_url, embedding_model
):
    index, vector_store = await _stores(db_session, qdrant_url, embedding_model)
    tenant_id, document_id = uuid.uuid4(), uuid.uuid4()

    await index.replace(tenant_id, document_id, "return-policy", V1)
    await index.replace(tenant_id, document_id, "return-policy", V2)

    stored = await _postgres_chunks(db_session, tenant_id, document_id)
    assert len(stored) > 1  # really chunked, so every chunk had to be replaced
    assert all("sixty" in chunk or "forty" not in chunk for chunk in stored)
    assert not any("forty-five" in chunk for chunk in stored)
    rows = (
        await db_session.execute(
            text("SELECT chunk_count FROM documents WHERE id = :id"), {"id": document_id}
        )
    ).all()
    assert [row.chunk_count for row in rows] == [len(stored)]

    hits = await vector_store.search(embedding_model.embed(V1), tenant_id, top_k=50)
    assert hits
    assert all(hit.document_id == document_id for hit in hits)
    assert not any("forty-five" in hit.content for hit in hits)


async def test_qdrant_delete_is_scoped_by_tenant(db_session, qdrant_url, embedding_model):
    index, vector_store = await _stores(db_session, qdrant_url, embedding_model)
    tenant_id, other_tenant, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    foreign = Chunk(uuid.uuid4(), document_id, "another tenant's text", embedding_model.embed("x"))
    await vector_store.upsert(foreign, other_tenant)

    await index.replace(tenant_id, document_id, "return-policy", V2)

    hits = await vector_store.search(embedding_model.embed("x"), other_tenant, top_k=5)
    assert [hit.chunk_id for hit in hits] == [foreign.id]


async def test_postgres_delete_document_removes_chunks_then_the_row(db_session, embedding_model):
    tenant_id, document_id = uuid.uuid4(), uuid.uuid4()
    await set_tenant_context(db_session, tenant_id)
    repository = PostgresDocumentRepository(db_session)
    await repository.save_document(
        Document(document_id, tenant_id, "f", "text/plain", "p", 1, "completed")
    )
    await repository.save_chunks(
        [Chunk(uuid.uuid4(), document_id, "text", embedding_model.embed("text"))], tenant_id
    )

    await repository.delete_document(document_id, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    counts = (
        await db_session.execute(
            text(
                "SELECT (SELECT count(*) FROM chunks WHERE document_id = :id) AS chunks, "
                "(SELECT count(*) FROM documents WHERE id = :id) AS documents"
            ),
            {"id": document_id},
        )
    ).one()
    assert (counts.chunks, counts.documents) == (0, 0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_chunked_rag_index.py -q -p no:cacheprovider`
Expected: a collection error, `ModuleNotFoundError: src.orchestration.infrastructure.chunked_rag_index`.

- [ ] **Step 3: Add `QdrantVectorStore.delete_document`**

```python
    async def delete_document(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        """Delete every point of one document, filtered by tenant as well as document,
        so one tenant can never delete another's points."""
        await self._client.delete(
            collection_name=_COLLECTION_NAME,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="tenant_id", match=qmodels.MatchValue(value=str(tenant_id))
                        ),
                        qmodels.FieldCondition(
                            key="document_id", match=qmodels.MatchValue(value=str(document_id))
                        ),
                    ]
                )
            ),
            wait=True,
        )
```

- [ ] **Step 4: Add `PostgresDocumentRepository.delete_document`**

```python
    async def delete_document(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        # Chunks first: chunks.document_id references documents.id. tenant_id is matched
        # explicitly as well as through RLS, like the chunk inserts above.
        params = {"id": document_id, "tenant_id": tenant_id}
        await self._session.execute(
            text("DELETE FROM chunks WHERE document_id = :id AND tenant_id = :tenant_id"), params
        )
        await self._session.execute(
            text("DELETE FROM documents WHERE id = :id AND tenant_id = :tenant_id"), params
        )
        await self._session.flush()
```

- [ ] **Step 5: Create `src/orchestration/infrastructure/chunked_rag_index.py`**

```python
import asyncio
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.infrastructure.db import set_tenant_context
from src.orchestration.domain.ports import RagIndex
from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import Chunker, EmbeddingModel
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore


class ChunkedRagIndex(RagIndex):
    """Keeps one data source's current text in RAG as a single document.

    replace deletes the previous version from both stores before writing the new
    one. A superseded chunk left in Qdrant would still be retrieved by vector
    search, and one left in Postgres by hybrid RAG's BM25KeywordSearch.

    The Postgres half is one transaction, and the Qdrant half follows it. Between
    the two, vector search can briefly return the old version; a replace that
    fails there leaves the stored hash unchanged, so the router retries it.
    Embedding runs on a worker thread, because this shares an event loop with the
    query path.
    """

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        vector_store: QdrantVectorStore,
        chunker: Chunker,
        embedder: EmbeddingModel,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._vector_store = vector_store
        self._chunker = chunker
        self._embedder = embedder

    async def replace(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, title: str, text: str
    ) -> None:
        chunks = [
            Chunk(
                id=uuid.uuid4(),
                document_id=document_id,
                content=piece,
                embedding=await asyncio.to_thread(self._embedder.embed, piece),
            )
            for piece in self._chunker.chunk(text)
        ]
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            documents = PostgresDocumentRepository(session)
            await documents.delete_document(document_id, tenant_id)
            await documents.save_document(
                Document(
                    id=document_id,
                    tenant_id=tenant_id,
                    filename=title,
                    mime_type="text/plain",
                    storage_path=f"data-source:{title}",
                    chunk_count=len(chunks),
                    status="completed",
                )
            )
            await documents.save_chunks(chunks, tenant_id)
        await self._vector_store.delete_document(document_id, tenant_id)
        for chunk in chunks:
            await self._vector_store.upsert(chunk, tenant_id)
```

- [ ] **Step 6: Run and verify**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_chunked_rag_index.py -q -p no:cacheprovider`
Expected: 3 passed. Then run the existing RAG store tests (`tests/integration/test_qdrant_vector_store.py` and `tests/integration/test_rag_rls_tenant_isolation.py`, if present), ruff on the changed files, and `mypy src`.

- [ ] **Step 7: Commit**

```bash
git add src/rag/infrastructure/qdrant_vector_store.py src/rag/infrastructure/postgres_document_repository.py src/orchestration/infrastructure/chunked_rag_index.py tests/integration/test_chunked_rag_index.py
git commit  # feat: replace a data source's chunks in both RAG stores
```

---

### Task 8: `RecordSemanticFactWriter`

**Files:**
- Create: `src/orchestration/infrastructure/record_semantic_fact_writer.py`
- Test: `tests/integration/test_record_semantic_fact_writer.py`

**Interfaces:**
- Consumes: `SessionFactWriter` (Task 2); the existing `RecordSemanticFact`, `PostgresSemanticMemoryRepository`, `SemanticMemoryIndex`, `MemoryGraphRepository`, and `EmbeddingModel`.
- Produces: `RecordSemanticFactWriter(sessionmaker, semantic_index, embedder, graph, *, source: str = "freshness-router")`.

- [ ] **Step 1: Write the failing test** in `tests/integration/test_record_semantic_fact_writer.py`

```python
import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.orchestration.infrastructure.record_semantic_fact_writer import RecordSemanticFactWriter

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"


async def _create_user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id) "
            "VALUES (:id, :email, :hashed_password, :tenant_id)"
        ),
        {"id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
         "tenant_id": tenant_id},
    )
    await db_session.commit()
    return user_id


async def _facts(db_session, tenant_id, user_id):
    await set_tenant_context(db_session, tenant_id)
    result = await db_session.execute(
        text("SELECT fact_key, fact_value, source FROM semantic_memory WHERE user_id = :user_id"),
        {"user_id": user_id},
    )
    return [tuple(row) for row in result]


async def test_a_user_scoped_source_becomes_one_updatable_fact_visible_only_in_its_tenant(
    db_session, qdrant_url, neo4j_url, embedding_model
):
    tenant_id = uuid.uuid4()
    user_id = await _create_user(db_session, tenant_id)
    index = QdrantSemanticMemoryIndex(qdrant_url)
    await index.ensure_collection()
    url, username, password = neo4j_url
    graph = Neo4jMemoryGraphRepository(url, auth=(username, password))
    await graph.ensure_schema()
    writer = RecordSemanticFactWriter(
        get_sessionmaker(db_session.bind), index, embedding_model, graph
    )
    try:
        await writer.record(tenant_id, user_id, "size-preference", "wears size 10")
        await writer.record(tenant_id, user_id, "size-preference", "wears size 11")
    finally:
        await graph.close()

    assert await _facts(db_session, tenant_id, user_id) == [
        ("size-preference", "wears size 11", "freshness-router")
    ]
    assert await _facts(db_session, uuid.uuid4(), user_id) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_record_semantic_fact_writer.py -q -p no:cacheprovider`
Expected: a collection error, `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/orchestration/infrastructure/record_semantic_fact_writer.py`**

```python
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.domain.ports import MemoryGraphRepository, SemanticMemoryIndex
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.domain.ports import SessionFactWriter
from src.rag.domain.ports import EmbeddingModel


class RecordSemanticFactWriter(SessionFactWriter):
    """Writes a user-scoped data source into MAG through RecordSemanticFact, unmodified.

    The source key is the fact key, so RecordSemanticFact's upsert on
    (user_id, fact_key) makes each new version overwrite the last. The Postgres
    repository flushes into a session, and this writer owns that session's
    transaction: the same unit-of-work rule the cascade's MAG search follows.
    """

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        semantic_index: SemanticMemoryIndex,
        embedder: EmbeddingModel,
        graph: MemoryGraphRepository,
        *,
        source: str = "freshness-router",
    ) -> None:
        self._sessionmaker = sessionmaker
        self._index = semantic_index
        self._embedder = embedder
        self._graph = graph
        self._source = source

    async def record(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, fact_key: str, fact_value: str
    ) -> None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            command = RecordSemanticFact(
                PostgresSemanticMemoryRepository(session), self._index, self._embedder, self._graph
            )
            await command.execute(tenant_id, user_id, fact_key, fact_value, source=self._source)
```

- [ ] **Step 4: Run and verify**

Run the test again. Expected: 1 passed. Then run ruff on both files, and `mypy src`.

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/infrastructure/record_semantic_fact_writer.py tests/integration/test_record_semantic_fact_writer.py
git commit  # feat: record user-scoped data sources as MAG facts
```

---

### Task 9: End-to-end against real stores and Batch A's cascade

**Files:**
- Create: `tests/integration/freshness_env.py` (not collected: no `test_` prefix)
- Test: `tests/integration/test_freshness_router_end_to_end.py`

**Interfaces:**
- Consumes:
  - Tasks 1–8.
  - Batch A's `LatencyCascade`, `TierTimeouts`, `CagTier`, `MagTier`, `RagTier`, and `SessionScopedSemanticFactSearch`.
  - `CachingEmbeddingModel`, `SearchDocuments`, `HFFrozenCache`, and `FixedSizeChunker`.
  - The thresholds `CAG_HIT`, `CAG_PARTIAL`, `MAG_HIT`, and `MAG_PARTIAL` from `evaluation/scenarios/orchestration_meta_layer_thresholds.py`.
- Produces:
  - `SimulatedClock`.
  - `freshness_env(db_session, qdrant_url, neo4j_url, embedding_model, tokenizer, model)`, an async context manager yielding a `FreshnessEnv` with `ingest`, `refresh`, `review`, `clock`, `tenant_id`, `owner_id`, `other_user_id`, and `ask(question, user_id=None) -> CascadeResult`.
  - Content constants reused by the runner in Task 11.

- [ ] **Step 1: Create `tests/integration/freshness_env.py`**

```python
"""Real-store composition for the freshness router's end-to-end tests:
- Postgres with RLS, and Qdrant;
- MiniLM behind CachingEmbeddingModel;
- a distilgpt2 HFFrozenCache behind ExpiringFrozenCache, on a clock the test moves;
- MAG writes through RecordSemanticFact with Neo4j;
- Batch A's unrouted cascade to ask questions. Concept 5 as drawn trusts a CAG hit,
  which is exactly what stale placement exploits.

Not collected by pytest (no test_ prefix)."""
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from evaluation.scenarios.orchestration_meta_layer_thresholds import (
    CAG_HIT,
    CAG_PARTIAL,
    MAG_HIT,
    MAG_PARTIAL,
)
from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.cascade_tiers import CagTier, MagTier, RagTier
from src.orchestration.application.ingest_data_source import IngestDataSource
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.refresh_cached_sources import RefreshCachedSources
from src.orchestration.application.review_source_freshness import ReviewSourceFreshness
from src.orchestration.domain.entities import (
    CascadeResult,
    DataSourceProfile,
    SourceScope,
    TierRequest,
)
from src.orchestration.infrastructure.chunked_rag_index import ChunkedRagIndex
from src.orchestration.infrastructure.expiring_frozen_cache import ExpiringFrozenCache
from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache
from src.orchestration.infrastructure.postgres_data_source_repository import (
    PostgresDataSourceRepository,
)
from src.orchestration.infrastructure.record_semantic_fact_writer import RecordSemanticFactWriter
from src.orchestration.infrastructure.session_scoped_semantic_fact_search import (
    SessionScopedSemanticFactSearch,
)
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.ports import EmbeddingModel
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)

PRICES = DataSourceProfile("backpack-price", SourceScope.TENANT, HOUR)
PRICE_QUESTION = "How much does the blue hiking backpack cost today?"


def price_text(dollars: int) -> str:
    return f"The blue hiking backpack costs {dollars} dollars today."


POLICY = DataSourceProfile("return-policy", SourceScope.TENANT, 90 * DAY)
POLICY_QUESTION = "What is the return policy for unopened items?"


def policy_text(days: str) -> str:
    return (
        f"Our return policy allows customers to return unopened items within {days} days "
        "of purchase for a full refund."
    )


SIZE = DataSourceProfile("size-preference", SourceScope.USER, 30 * DAY)
SIZE_QUESTION = "What shoe size do I wear?"
SIZE_TEXT = "The user wears size 10 running shoes."

# Generous on purpose: these tests check placement, not latency.
CORRECTNESS_TIMEOUTS = TierTimeouts(cag=5.0, mag=5.0, rag=10.0)


class SimulatedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@dataclass
class FreshnessEnv:
    tenant_id: uuid.UUID
    owner_id: uuid.UUID
    other_user_id: uuid.UUID
    clock: SimulatedClock
    embedder: EmbeddingModel
    ingest: IngestDataSource
    refresh: RefreshCachedSources
    review: ReviewSourceFreshness
    cascade: LatencyCascade

    async def ask(self, question: str, user_id: uuid.UUID | None = None) -> CascadeResult:
        request = TierRequest(
            self.tenant_id, user_id or self.owner_id, uuid.uuid4(), question,
            self.embedder.embed(question),
        )
        return await self.cascade.run(request)


async def create_user(db_session: Any, tenant_id: uuid.UUID) -> uuid.UUID:
    await set_tenant_context(db_session, tenant_id)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id) "
            "VALUES (:id, :email, :hashed_password, :tenant_id)"
        ),
        {"id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
         "tenant_id": tenant_id},
    )
    await db_session.commit()
    return user_id


@asynccontextmanager
async def freshness_env(
    db_session: Any,
    qdrant_url: str,
    neo4j_url: tuple[str, str, str],
    embedding_model: EmbeddingModel,
    tokenizer: Any,
    model: Any,
) -> AsyncIterator[FreshnessEnv]:
    tenant_id = uuid.uuid4()
    owner_id = await create_user(db_session, tenant_id)
    other_user_id = await create_user(db_session, tenant_id)
    sessionmaker = get_sessionmaker(db_session.bind)
    embedder = CachingEmbeddingModel(embedding_model)
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    semantic_index = QdrantSemanticMemoryIndex(qdrant_url)
    await semantic_index.ensure_collection()
    url, username, password = neo4j_url
    graph = Neo4jMemoryGraphRepository(url, auth=(username, password))
    await graph.ensure_schema()

    clock = SimulatedClock(T0)
    cache = ExpiringFrozenCache(HFFrozenCache(tokenizer=tokenizer, model=model), clock)
    search = SearchDocuments(embedder, vector_store)
    warmed = CacheWarmedRetrieve(embedder, cache, search, similarity_threshold=CAG_HIT)
    repository = PostgresDataSourceRepository(sessionmaker)
    rag_index = ChunkedRagIndex(sessionmaker, vector_store, FixedSizeChunker(), embedder)
    writer = RecordSemanticFactWriter(sessionmaker, semantic_index, embedder, graph)
    cascade = LatencyCascade(
        [
            CagTier(warmed, hit_threshold=CAG_HIT, partial_threshold=CAG_PARTIAL),
            MagTier(
                SessionScopedSemanticFactSearch(sessionmaker),
                hit_threshold=MAG_HIT,
                partial_threshold=MAG_PARTIAL,
            ),
            RagTier(search, top_k=2),
        ],
        CORRECTNESS_TIMEOUTS,
    )
    try:
        yield FreshnessEnv(
            tenant_id=tenant_id,
            owner_id=owner_id,
            other_user_id=other_user_id,
            clock=clock,
            embedder=embedder,
            ingest=IngestDataSource(repository, rag_index, cache, warmed, writer),
            refresh=RefreshCachedSources(repository, cache, warmed),
            review=ReviewSourceFreshness(repository, cache, warmed),
            cascade=cascade,
        )
    finally:
        await cascade.drain()
        await graph.close()
```

- [ ] **Step 2: Write the tests** in `tests/integration/test_freshness_router_end_to_end.py`

```python
from src.orchestration.domain.entities import IngestionRoute, Paradigm, TierOutcome
from tests.integration.freshness_env import (
    DAY,
    HOUR,
    POLICY,
    POLICY_QUESTION,
    PRICE_QUESTION,
    PRICES,
    SIZE,
    SIZE_QUESTION,
    SIZE_TEXT,
    T0,
    freshness_env,
    policy_text,
    price_text,
)

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


def _context(result) -> str:
    return "\n".join(item.content for item in result.items)


def _paradigms(result) -> set[Paradigm]:
    return {item.paradigm for item in result.items}


async def test_a_volatile_source_is_answered_from_rag_and_never_cached(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        await env.ingest.execute(env.tenant_id, PRICES, price_text(41), T0)
        assert await env.refresh.run(env.tenant_id, T0) == []
        first = await env.ask(PRICE_QUESTION)
        await env.ingest.execute(env.tenant_id, PRICES, price_text(43), T0 + HOUR)
        second = await env.ask(PRICE_QUESTION)

    assert CAG not in _paradigms(first)
    assert "41 dollars" in _context(first)
    assert "43 dollars" in _context(second)
    assert "41 dollars" not in _context(second)


async def test_a_stable_source_is_served_from_cag_once_the_batch_has_preloaded_it(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        result = await env.ingest.execute(env.tenant_id, POLICY, policy_text("forty-five"), T0)
        before = await env.ask(POLICY_QUESTION)
        assert await env.refresh.run(env.tenant_id, T0) == ["return-policy"]
        after = await env.ask(POLICY_QUESTION)

    assert result.route is IngestionRoute.CAG_WITH_RAG_BACKUP
    assert CAG not in _paradigms(before)
    assert [(a.paradigm, a.outcome) for a in after.attempts] == [(CAG, TierOutcome.HIT)]
    assert "forty-five days" in _context(after)


async def test_a_changed_cached_source_is_served_from_rag_until_the_next_refresh(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        await env.ingest.execute(env.tenant_id, POLICY, policy_text("forty-five"), T0)
        await env.refresh.run(env.tenant_id, T0)
        env.clock.now = T0 + DAY
        await env.ingest.execute(env.tenant_id, POLICY, policy_text("sixty"), T0 + DAY)
        between = await env.ask(POLICY_QUESTION)
        await env.refresh.run(env.tenant_id, T0 + DAY)
        refreshed = await env.ask(POLICY_QUESTION)

    assert CAG not in _paradigms(between)
    assert "sixty days" in _context(between)
    assert "forty-five days" not in _context(between)
    assert CAG in _paradigms(refreshed)
    assert "sixty days" in _context(refreshed)
    assert "forty-five days" not in _context(refreshed)


async def test_an_expired_entry_falls_back_to_rag_and_is_not_preloaded_unconfirmed(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        await env.ingest.execute(env.tenant_id, POLICY, policy_text("forty-five"), T0)
        await env.refresh.run(env.tenant_id, T0)
        fresh = await env.ask(POLICY_QUESTION)
        env.clock.now = T0 + 45 * DAY  # 90-day interval x 0.5: exactly one TTL later
        expired = await env.ask(POLICY_QUESTION)
        preloaded = await env.refresh.run(env.tenant_id, env.clock.now)

    assert CAG in _paradigms(fresh)
    assert CAG not in _paradigms(expired)
    assert RAG in _paradigms(expired)
    assert preloaded == []


async def test_a_user_scoped_source_reaches_its_owner_through_mag_and_no_one_else(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        result = await env.ingest.execute(env.tenant_id, SIZE, SIZE_TEXT, T0, user_id=env.owner_id)
        owner = await env.ask(SIZE_QUESTION, env.owner_id)
        other = await env.ask(SIZE_QUESTION, env.other_user_id)

    assert result.route is IngestionRoute.MAG
    assert MAG in _paradigms(owner)
    assert "size 10" in _context(owner)
    assert "size 10" not in _context(other)


async def test_review_demotes_a_cached_source_that_starts_changing_hourly(
    db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    async with freshness_env(
        db_session, qdrant_url, neo4j_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    ) as env:
        await env.ingest.execute(env.tenant_id, POLICY, policy_text("45"), T0)
        await env.refresh.run(env.tenant_id, T0)
        for hour, days in ((1, "46"), (2, "47"), (3, "48")):
            await env.ingest.execute(env.tenant_id, POLICY, policy_text(days), T0 + hour * HOUR)
        env.clock.now = T0 + 4 * HOUR
        migrations = await env.review.run(env.tenant_id, env.clock.now)
        preloaded = await env.refresh.run(env.tenant_id, env.clock.now)
        answer = await env.ask(POLICY_QUESTION)

    assert [(m.source_key, m.to_route) for m in migrations] == [
        ("return-policy", IngestionRoute.RAG_ONLY)
    ]
    assert migrations[0].observed_interval == HOUR
    assert preloaded == []
    assert CAG not in _paradigms(answer)
    assert "48 days" in _context(answer)
```

- [ ] **Step 3: Run**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_freshness_router_end_to_end.py -q -p no:cacheprovider`
Expected: 6 passed.

These tests depend on real MiniLM similarity clearing the thresholds Batch A measured: CAG 0.43 hit / 0.35 partial, MAG 0.42 / 0.37. If an assertion about which tier answered fails, print each question's score against each stored text before changing anything. Reword the question until its own text clears the hit threshold and every other text stays below the partial one. **Never change the thresholds.** Record the measured scores in a comment in `freshness_env.py`, and the change in the plan's execution notes.

- [ ] **Step 4: Revert-check TTL fallback.** Temporarily construct the cache as `ExpiringFrozenCache(..., clock=lambda: T0)` so it never advances. Confirm `test_an_expired_entry_falls_back_to_rag_and_is_not_preloaded_unconfirmed` fails, then restore.

- [ ] **Step 5: Lint and commit**

Run ruff on both files.

```bash
git add tests/integration/freshness_env.py tests/integration/test_freshness_router_end_to_end.py
git commit  # test: validate freshness routing end to end against real stores
```

---

### Task 10: Measurement metrics and report renderer

**Files:**
- Create: `evaluation/domain/freshness_metrics.py`
- Create: `evaluation/infrastructure/freshness_report.py`
- Test: `tests/unit/test_freshness_metrics.py`, `tests/unit/test_freshness_report.py`

**Interfaces:**
- Produces:
  - `ProbeObservation(arm, source_key, context, current_marker, superseded_markers: tuple[str, ...], served_from_cag: bool, foreign_markers: tuple[str, ...] = ())`.
  - `PlacementTally`, with fields `arm, source_key, probes, stale_only, mixed, fresh_only, neither, cag_served, stale_from_cag, foreign_exposures` and the properties `stale_rate` and `cag_share` (both NaN over zero probes).
  - `tally_probes(observations) -> list[PlacementTally]`, in first-seen `(arm, source_key)` order.
  - `PreloadTracker`, with `preloaded(arm, key)`, `served(arm, key)`, and `counts(arm, key) -> tuple[int, int]`, which returns `(preloads, never_served)`.
  - `MigrationObservation(source_key, from_route, to_route, shift_at, migrated_at)`, with the property `lag`.
  - `RouteRow(data_type, declared_change, scope, source_paradigm, route, agrees: bool)`.
  - `render_freshness_measurements(routes, tallies, preloads: dict[tuple[str, str], tuple[int, int]], migrations, ttl_tallies, notes) -> str`.

- [ ] **Step 1: Write the failing metric tests** in `tests/unit/test_freshness_metrics.py`

```python
import math
from datetime import UTC, datetime, timedelta

from evaluation.domain.freshness_metrics import (
    MigrationObservation,
    PreloadTracker,
    ProbeObservation,
    tally_probes,
)


def _probe(context: str, *, cag: bool = False, arm: str = "a", key: str = "k",
           foreign: tuple[str, ...] = ()) -> ProbeObservation:
    return ProbeObservation(arm, key, context, "v3", ("v1", "v2"), cag, foreign)


def test_contexts_are_classified_by_which_versions_they_hold():
    [tally] = tally_probes([
        _probe("v1"), _probe("v2 and v3", cag=True), _probe("v3", cag=True), _probe("nothing"),
    ])
    assert (tally.probes, tally.stale_only, tally.mixed, tally.fresh_only, tally.neither) == (
        4, 1, 1, 1, 1,
    )
    assert (tally.cag_served, tally.stale_from_cag) == (2, 1)
    assert tally.stale_rate == 0.25
    assert tally.cag_share == 0.5


def test_foreign_text_reaching_a_probe_is_counted():
    [tally] = tally_probes([
        _probe("v3 size 10", foreign=("size 10",)), _probe("v3", foreign=("size 10",)),
    ])
    assert tally.foreign_exposures == 1


def test_tallies_are_grouped_by_arm_and_source_in_first_seen_order():
    tallies = tally_probes([
        _probe("v3", arm="b"), _probe("v3", arm="a"), _probe("v1", arm="b"),
    ])
    assert [(t.arm, t.probes) for t in tallies] == [("b", 2), ("a", 1)]


def test_rates_are_nan_over_zero_probes():
    from evaluation.domain.freshness_metrics import PlacementTally

    empty = PlacementTally("a", "k", 0, 0, 0, 0, 0, 0, 0, 0)
    assert math.isnan(empty.stale_rate)
    assert math.isnan(empty.cag_share)


def test_a_preload_counts_as_used_once_any_probe_is_served_from_it():
    tracker = PreloadTracker()
    tracker.preloaded("a", "k")
    tracker.served("a", "k")
    tracker.served("a", "k")
    tracker.preloaded("a", "k")  # replaced before serving anything
    tracker.served("b", "k")  # no preload for this arm: ignored
    assert tracker.counts("a", "k") == (2, 1)
    assert tracker.counts("b", "k") == (0, 0)


def test_migration_lag_is_migration_time_minus_shift_time():
    shift = datetime(2026, 1, 11, tzinfo=UTC)
    observation = MigrationObservation("catalog", "cag_with_rag_backup", "rag_only", shift,
                                       shift + timedelta(hours=21))
    assert observation.lag == timedelta(hours=21)
```

- [ ] **Step 2: Write the failing renderer test** in `tests/unit/test_freshness_report.py`

```python
from datetime import UTC, datetime, timedelta

from evaluation.domain.freshness_metrics import MigrationObservation, PlacementTally, RouteRow
from evaluation.infrastructure.freshness_report import render_freshness_measurements


def test_the_report_renders_every_section_with_its_numbers():
    shift = datetime(2026, 1, 11, tzinfo=UTC)
    report = render_freshness_measurements(
        routes=[RouteRow("Stock prices", "seconds", "tenant", "RAG", "rag_only", True)],
        tallies=[PlacementTally("freshness-aware", "prices", 8, 0, 0, 8, 0, 0, 0, 0)],
        preloads={("freshness-aware", "prices"): (0, 0)},
        migrations=[MigrationObservation("catalog", "cag_with_rag_backup", "rag_only", shift,
                                         shift + timedelta(hours=21))],
        ttl_tallies=[PlacementTally("ttl 0.5", "warranty", 240, 20, 0, 200, 20, 20, 20, 0)],
        notes="Corpus notes.",
    )
    assert report.startswith("# Freshness-Aware Data Router — Measurements")
    assert "Corpus notes." in report
    assert "| Stock prices | seconds | tenant | RAG | rag_only | yes |" in report
    assert "| freshness-aware | prices | 8 | 0 | 0 | 8 | 0 | 0% | 0% | 0 | 0 | 0 |" in report
    assert "| catalog | cag_with_rag_backup → rag_only | 2026-01-11 00:00 | 2026-01-11 21:00 | 21.0 h |" in report
    assert "| ttl 0.5 | warranty | 240 | 20 |" in report
```

- [ ] **Step 3: Run to verify both fail**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_freshness_metrics.py tests/unit/test_freshness_report.py -q -p no:cacheprovider`
Expected: collection errors, `ModuleNotFoundError`.

- [ ] **Step 4: Create `evaluation/domain/freshness_metrics.py`**

```python
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class ProbeObservation:
    arm: str
    source_key: str
    context: str
    current_marker: str
    superseded_markers: tuple[str, ...]
    served_from_cag: bool
    # Text that must never reach this probe, such as another user's personal fact.
    foreign_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlacementTally:
    arm: str
    source_key: str
    probes: int
    stale_only: int
    mixed: int
    fresh_only: int
    neither: int
    cag_served: int
    stale_from_cag: int
    foreign_exposures: int

    @property
    def stale_rate(self) -> float:
        return self.stale_only / self.probes if self.probes else math.nan

    @property
    def cag_share(self) -> float:
        return self.cag_served / self.probes if self.probes else math.nan


def tally_probes(observations: Sequence[ProbeObservation]) -> list[PlacementTally]:
    groups: dict[tuple[str, str], list[ProbeObservation]] = {}
    for observation in observations:
        groups.setdefault((observation.arm, observation.source_key), []).append(observation)
    tallies: list[PlacementTally] = []
    for (arm, key), group in groups.items():
        counts = {"stale_only": 0, "mixed": 0, "fresh_only": 0, "neither": 0}
        stale_from_cag = 0
        for o in group:
            stale = any(marker in o.context for marker in o.superseded_markers)
            fresh = o.current_marker in o.context
            label = ("mixed" if fresh else "stale_only") if stale else (
                "fresh_only" if fresh else "neither"
            )
            counts[label] += 1
            stale_from_cag += int(stale and o.served_from_cag)
        tallies.append(
            PlacementTally(
                arm=arm,
                source_key=key,
                probes=len(group),
                cag_served=sum(o.served_from_cag for o in group),
                stale_from_cag=stale_from_cag,
                foreign_exposures=sum(
                    any(m in o.context for m in o.foreign_markers) for o in group
                ),
                **counts,
            )
        )
    return tallies


class PreloadTracker:
    """Counts pre-loads per (arm, source), and how many were never served from."""

    def __init__(self) -> None:
        self._generations: dict[tuple[str, str], list[bool]] = {}

    def preloaded(self, arm: str, source_key: str) -> None:
        self._generations.setdefault((arm, source_key), []).append(False)

    def served(self, arm: str, source_key: str) -> None:
        generations = self._generations.get((arm, source_key))
        if generations:
            generations[-1] = True

    def counts(self, arm: str, source_key: str) -> tuple[int, int]:
        generations = self._generations.get((arm, source_key), [])
        return len(generations), generations.count(False)


@dataclass(frozen=True)
class MigrationObservation:
    source_key: str
    from_route: str
    to_route: str
    shift_at: datetime
    migrated_at: datetime

    @property
    def lag(self) -> timedelta:
        return self.migrated_at - self.shift_at


@dataclass(frozen=True)
class RouteRow:
    data_type: str
    declared_change: str
    scope: str
    source_paradigm: str
    route: str
    agrees: bool
```

- [ ] **Step 5: Create `evaluation/infrastructure/freshness_report.py`**

```python
from __future__ import annotations

import math

from evaluation.domain.freshness_metrics import MigrationObservation, PlacementTally, RouteRow


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.0%}"


def render_freshness_measurements(
    routes: list[RouteRow],
    tallies: list[PlacementTally],
    preloads: dict[tuple[str, str], tuple[int, int]],
    migrations: list[MigrationObservation],
    ttl_tallies: list[PlacementTally],
    notes: str,
) -> str:
    lines = [
        "# Freshness-Aware Data Router — Measurements",
        "",
        notes,
        "",
        "## Concept 9's freshness spectrum, routed",
        "",
        "| Data type | Declared change | Scope | Source's paradigm | Route | Agrees |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.data_type} | {r.declared_change} | {r.scope} | {r.source_paradigm} "
        f"| {r.route} | {'yes' if r.agrees else 'no'} |"
        for r in routes
    ]
    lines += [
        "",
        "## Placement ablation over 30 simulated days",
        "",
        "| Arm | Source | Probes | Stale only | Mixed | Current only | Neither | Stale rate "
        "| Served from CAG | Pre-loads | Never-served pre-loads | Foreign exposures |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in tallies:
        preloaded, unused = preloads.get((t.arm, t.source_key), (0, 0))
        lines.append(
            f"| {t.arm} | {t.source_key} | {t.probes} | {t.stale_only} | {t.mixed} "
            f"| {t.fresh_only} | {t.neither} | {_pct(t.stale_rate)} | {_pct(t.cag_share)} "
            f"| {preloaded} | {unused} | {t.foreign_exposures} |"
        )
    lines += [
        "",
        "## Migration lag",
        "",
        "| Source | Migration | Pattern shift | Migrated | Lag |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {m.source_key} | {m.from_route} → {m.to_route} "
        f"| {m.shift_at:%Y-%m-%d %H:%M} | {m.migrated_at:%Y-%m-%d %H:%M} "
        f"| {m.lag.total_seconds() / 3600:.1f} h |"
        for m in migrations
    ]
    lines += [
        "",
        "## TTL bound on a silently changed cached source",
        "",
        "| Arm | Source | Probes | Superseded text served from CAG |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {t.arm} | {t.source_key} | {t.probes} | {t.stale_from_cag} |" for t in ttl_tallies
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 6: Run and verify**

Run the two test files. Expected: 7 passed. Then run ruff on the four files.

- [ ] **Step 7: Commit**

```bash
git add evaluation/domain/freshness_metrics.py evaluation/infrastructure/freshness_report.py tests/unit/test_freshness_metrics.py tests/unit/test_freshness_report.py
git commit  # feat: add freshness placement metrics and their report renderer
```

---

### Task 11: The measurement runner

**Files:**
- Create: `evaluation/scenarios/freshness-router/run_freshness_measurements.py` (no `__init__.py`, matching the other hyphenated scenario directories)
- Output: `evaluation/reports/freshness-router-measurements.md`

**Interfaces:**
- Consumes everything from Tasks 1–10, plus Batch A's cascade and thresholds.
- Produces the generated report. The runner imports nothing from `tests/`: Batch A's review moved shared thresholds out of test code for exactly that reason.

- [ ] **Step 1: Create the runner**

```python
"""Freshness-Aware Data Router measurements, on a simulated clock against real stores.

Parts (select with --parts):
- spectrum: Concept 9's eight data types, declared as profiles and routed.
- ablation: 30 simulated days, six sources, four placement arms. Every source is
  ingested hourly; refresh (and review, in the freshness-aware arm) runs at
  midnight; every source is probed every 3 hours through Batch A's unrouted
  cascade, which trusts a CAG hit exactly as Concept 5 draws it.
- ttl: a cached source changed in RAG behind the router's back while its feed has
  stalled, with a TTL and without one.

Starts its own Postgres, Qdrant, and Neo4j containers. No LLM: staleness is judged
on the assembled context, using version-specific markers. Before any arm runs, the
questions are checked against the tier thresholds, so a wording problem stops the
run instead of skewing it. Not pytest-collected.

Usage (from the repository root):
    PYTHONPATH=. python evaluation/scenarios/freshness-router/run_freshness_measurements.py
"""
import argparse
import asyncio
import io
import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.qdrant import QdrantContainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from alembic import command
from evaluation.domain.freshness_metrics import (
    MigrationObservation,
    PreloadTracker,
    ProbeObservation,
    RouteRow,
    tally_probes,
)
from evaluation.infrastructure.freshness_report import render_freshness_measurements
from evaluation.scenarios.orchestration_meta_layer_thresholds import (
    CAG_HIT,
    CAG_PARTIAL,
    MAG_HIT,
    MAG_PARTIAL,
)
from src.identity.infrastructure.db import get_engine, get_sessionmaker, set_tenant_context
from src.mag.infrastructure.neo4j_memory_graph_repository import Neo4jMemoryGraphRepository
from src.mag.infrastructure.qdrant_semantic_memory_index import QdrantSemanticMemoryIndex
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.cascade_tiers import CagTier, MagTier, RagTier
from src.orchestration.application.ingest_data_source import IngestDataSource, RoutePolicy
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.refresh_cached_sources import RefreshCachedSources
from src.orchestration.application.review_source_freshness import ReviewSourceFreshness
from src.orchestration.domain.entities import (
    DataSourceProfile,
    FreshnessPolicy,
    IngestionRoute,
    Paradigm,
    SourceScope,
    TierRequest,
)
from src.orchestration.domain.freshness_router import route_for, source_id_for
from src.orchestration.domain.similarity import cosine_similarity
from src.orchestration.infrastructure.chunked_rag_index import ChunkedRagIndex
from src.orchestration.infrastructure.expiring_frozen_cache import ExpiringFrozenCache
from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache
from src.orchestration.infrastructure.postgres_data_source_repository import (
    PostgresDataSourceRepository,
)
from src.orchestration.infrastructure.record_semantic_fact_writer import RecordSemanticFactWriter
from src.orchestration.infrastructure.session_scoped_semantic_fact_search import (
    SessionScopedSemanticFactSearch,
)
from src.rag.application.search_documents import SearchDocuments
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder

_REPORT = Path("evaluation/reports/freshness-router-measurements.md")
_PARTS = ("spectrum", "ablation", "ttl")
_APP_DB_PASSWORD = "evaluation-only-app-user-password"
_SEED_PASSWORD_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
T0 = datetime(2026, 1, 1, tzinfo=UTC)  # day 1, 00:00
HOUR, DAY = timedelta(hours=1), timedelta(days=1)
HORIZON_HOURS = 30 * 24
PROBE_EVERY_HOURS = 3
_TIMEOUTS = TierTimeouts(cag=5.0, mag=5.0, rag=10.0)  # placement, not latency, is measured


def _hours(now: datetime) -> int:
    return int((now - T0) / HOUR)


@dataclass(frozen=True)
class Feed:
    profile: DataSourceProfile
    question: str
    version_at: Callable[[datetime], int]
    text: Callable[[int], str]
    marker: Callable[[int], str]


def _catalog_version(now: datetime) -> int:
    hours = _hours(now)  # daily until day 10 (hour 216), hourly after
    return hours // 24 if hours < 216 else 9 + (hours - 216)


_POLICY_DAYS = ("forty-five", "sixty")
FEEDS = [
    Feed(
        DataSourceProfile("backpack-price", SourceScope.TENANT, HOUR),
        "How much does the blue hiking backpack cost today?",
        _hours,
        lambda n: f"The blue hiking backpack costs {40 + n} dollars today.",
        lambda n: f"costs {40 + n} dollars",
    ),
    Feed(
        DataSourceProfile("return-policy", SourceScope.TENANT, 90 * DAY),
        "What is the return policy for unopened items?",
        lambda now: int(now >= T0 + 11 * DAY + 9 * HOUR),  # day 12, 09:00
        lambda n: (
            "Our return policy allows customers to return unopened items within "
            f"{_POLICY_DAYS[n]} days of purchase for a full refund."
        ),
        lambda n: f"within {_POLICY_DAYS[n]} days",
    ),
    Feed(
        DataSourceProfile("shoe-catalog", SourceScope.TENANT, 7 * DAY),
        "How many trail running shoe models does the catalog list?",
        _catalog_version,
        lambda n: f"The trail running shoe catalog lists {100 + n} models this season.",
        lambda n: f"lists {100 + n} models",
    ),
    Feed(
        DataSourceProfile("shipping-guide", SourceScope.TENANT, 365 * DAY),
        "How long does standard shipping take?",
        lambda now: 0,
        lambda n: "Standard shipping takes five to seven business days.",
        lambda n: "five to seven",
    ),
    Feed(
        DataSourceProfile("flash-sale", SourceScope.TENANT, HOUR),
        "Which code gives the flash sale discount on camping tents?",
        lambda now: min(_hours(now), 96),  # hourly until day 5 (hour 96), then quiet
        lambda n: f"Flash sale code SALE{n} takes 20 percent off every camping tent.",
        lambda n: f"SALE{n} ",
    ),
]
SIZE_FEED = Feed(
    DataSourceProfile("size-preference", SourceScope.USER, 30 * DAY),
    "What shoe size do I wear?",
    lambda now: int(now >= T0 + 14 * DAY + 9 * HOUR),  # day 15, 09:00
    lambda n: f"The user wears size {10 + n} running shoes.",
    lambda n: f"size {10 + n} ",
)
_WARRANTY_YEARS = ("two", "three")
WARRANTY = Feed(
    DataSourceProfile("warranty-terms", SourceScope.TENANT, 7 * DAY),
    "How long is the product warranty?",
    lambda now: int(now >= T0 + 7 * DAY),  # day 8: changed in RAG behind the router's back
    lambda n: f"Every product carries a {_WARRANTY_YEARS[n]}-year limited warranty.",
    lambda n: f"{_WARRANTY_YEARS[n]}-year",
)
# When each drifting source's real behaviour changed, for the migration-lag table.
SHIFTS = {"shoe-catalog": T0 + 9 * DAY, "flash-sale": T0 + 4 * DAY}
SPECTRUM = [
    ("Stock prices, live scores", "seconds", SourceScope.TENANT, timedelta(seconds=1), "RAG"),
    ("News, social media", "minutes-hours", SourceScope.TENANT, HOUR, "RAG"),
    ("User session state", "every turn", SourceScope.USER, timedelta(minutes=1), "MAG"),
    ("Product catalog", "daily-weekly", SourceScope.TENANT, 3 * DAY, "CAG + RAG hybrid"),
    ("Company policies, manuals", "monthly-quarterly", SourceScope.TENANT, 60 * DAY, "CAG"),
    ("Textbooks, reference", "never", SourceScope.TENANT, 3650 * DAY, "CAG"),
    ("Code repositories", "per commit", SourceScope.TENANT, 4 * HOUR, "CAG + RAG"),
    ("User long-term preferences", "gradually", SourceScope.USER, 30 * DAY, "MAG"),
]
# Spec decision 2: the source's CAG rows are the stable route, which keeps RAG as backup.
_SOURCE_ROUTE = {
    "RAG": IngestionRoute.RAG_ONLY,
    "MAG": IngestionRoute.MAG,
    "CAG": IngestionRoute.CAG_WITH_RAG_BACKUP,
    "CAG + RAG hybrid": IngestionRoute.CAG_WITH_RAG_BACKUP,
    "CAG + RAG": IngestionRoute.CAG_WITH_RAG_BACKUP,
}


def _always_cag(profile: DataSourceProfile, policy: FreshnessPolicy) -> IngestionRoute:
    return IngestionRoute.CAG_WITH_RAG_BACKUP


def _always_rag(profile: DataSourceProfile, policy: FreshnessPolicy) -> IngestionRoute:
    return IngestionRoute.RAG_ONLY


NAIVE = "cache everything, batch refresh only"
ARMS: list[tuple[str, RoutePolicy, bool]] = [
    (NAIVE, _always_cag, False),
    ("cache everything, invalidate on change", _always_cag, False),
    ("RAG only", _always_rag, False),
    ("freshness-aware", route_for, True),
]


class SimulatedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@dataclass
class Stores:
    sessionmaker: async_sessionmaker[AsyncSession]
    vector_store: QdrantVectorStore
    semantic_index: QdrantSemanticMemoryIndex
    graph: Neo4jMemoryGraphRepository
    embedder: CachingEmbeddingModel
    tokenizer: Any
    model: Any


@dataclass
class Rig:
    name: str
    tenant_id: uuid.UUID
    owner_id: uuid.UUID
    other_id: uuid.UUID
    clock: SimulatedClock
    cache: ExpiringFrozenCache
    warmed: CacheWarmedRetrieve
    rag_index: ChunkedRagIndex
    cascade: LatencyCascade
    ingest: IngestDataSource
    refresh: RefreshCachedSources
    review: ReviewSourceFreshness | None


async def _build_rig(
    stores: Stores,
    name: str,
    route_policy: RoutePolicy,
    review: bool,
    policy: FreshnessPolicy | None = None,
) -> Rig:
    tenant_id, owner_id, other_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with stores.sessionmaker() as session, session.begin():
        await set_tenant_context(session, tenant_id)
        for user_id in (owner_id, other_id):
            await session.execute(
                text(
                    "INSERT INTO users (id, email, hashed_password, tenant_id) "
                    "VALUES (:id, :email, :hashed_password, :tenant_id)"
                ),
                {"id": user_id, "email": f"{user_id}@example.com",
                 "hashed_password": _SEED_PASSWORD_HASH, "tenant_id": tenant_id},
            )
    effective = policy or FreshnessPolicy()
    clock = SimulatedClock(T0)
    cache = ExpiringFrozenCache(
        HFFrozenCache(tokenizer=stores.tokenizer, model=stores.model), clock
    )
    search = SearchDocuments(stores.embedder, stores.vector_store)
    warmed = CacheWarmedRetrieve(stores.embedder, cache, search, similarity_threshold=CAG_HIT)
    repository = PostgresDataSourceRepository(stores.sessionmaker)
    rag_index = ChunkedRagIndex(
        stores.sessionmaker, stores.vector_store, FixedSizeChunker(), stores.embedder
    )
    writer = RecordSemanticFactWriter(
        stores.sessionmaker, stores.semantic_index, stores.embedder, stores.graph
    )
    cascade = LatencyCascade(
        [
            CagTier(warmed, hit_threshold=CAG_HIT, partial_threshold=CAG_PARTIAL),
            MagTier(
                SessionScopedSemanticFactSearch(stores.sessionmaker),
                hit_threshold=MAG_HIT,
                partial_threshold=MAG_PARTIAL,
            ),
            RagTier(search, top_k=2),
        ],
        _TIMEOUTS,
    )
    return Rig(
        name=name, tenant_id=tenant_id, owner_id=owner_id, other_id=other_id, clock=clock,
        cache=cache, warmed=warmed, rag_index=rag_index, cascade=cascade,
        ingest=IngestDataSource(
            repository, rag_index, cache, warmed, writer, policy=effective,
            route_policy=route_policy,
        ),
        refresh=RefreshCachedSources(repository, cache, warmed, policy=effective),
        review=ReviewSourceFreshness(repository, cache, warmed, policy=effective)
        if review else None,
    )


class NaivePlacement:
    """The batch-refresh-only baseline: every source's latest text goes to RAG at once,
    and into CAG only at the nightly re-warm, with no invalidation in between."""

    def __init__(self, rig: Rig) -> None:
        self._rig = rig
        self._latest: dict[uuid.UUID, tuple[str, str]] = {}

    async def ingest(self, feed: Feed, content: str, user_id: uuid.UUID | None) -> None:
        key = feed.profile.source_key
        source_id = source_id_for(self._rig.tenant_id, key, user_id)
        if self._latest.get(source_id, ("", ""))[1] != content:
            await self._rig.rag_index.replace(self._rig.tenant_id, source_id, key, content)
            self._latest[source_id] = (key, content)

    def rewarm(self) -> list[str]:
        for source_id, (_, content) in self._latest.items():
            self._rig.cache.preload(self._rig.tenant_id, source_id, content)
            self._rig.warmed.note_warmed(self._rig.tenant_id, source_id, content)
        return [key for key, _ in self._latest.values()]


def _probe_keys(source_key: str) -> list[str]:
    if source_key == SIZE_FEED.profile.source_key:
        return [f"{source_key} (owner)", f"{source_key} (other user)"]
    return [source_key]


async def _probe(
    stores: Stores, rig: Rig, feed: Feed, now: datetime, user_id: uuid.UUID, row: str,
    foreign: tuple[str, ...] = (),
) -> ProbeObservation:
    version = feed.version_at(now)
    request = TierRequest(
        rig.tenant_id, user_id, uuid.uuid4(), feed.question, stores.embedder.embed(feed.question)
    )
    result = await rig.cascade.run(request)
    return ProbeObservation(
        arm=rig.name,
        source_key=row,
        context="\n".join(item.content for item in result.items),
        current_marker=feed.marker(version),
        superseded_markers=tuple(feed.marker(v) for v in range(version)),
        served_from_cag=any(item.paradigm is Paradigm.CAG for item in result.items),
        foreign_markers=foreign,
    )


async def _run_ablation_arm(
    stores: Stores,
    arm: tuple[str, RoutePolicy, bool],
    tracker: PreloadTracker,
    migrations: list[MigrationObservation],
) -> list[ProbeObservation]:
    name, route_policy, review = arm
    rig = await _build_rig(stores, name, route_policy, review)
    naive = NaivePlacement(rig) if name == NAIVE else None
    size_markers = (SIZE_FEED.marker(0), SIZE_FEED.marker(1))
    observations: list[ProbeObservation] = []
    for tick in range(HORIZON_HOURS + 1):
        now = T0 + tick * HOUR
        rig.clock.now = now
        for feed, user_id in [*((f, None) for f in FEEDS), (SIZE_FEED, rig.owner_id)]:
            content = feed.text(feed.version_at(now))
            if naive is not None:
                await naive.ingest(feed, content, user_id)
            else:
                await rig.ingest.execute(
                    rig.tenant_id, feed.profile, content, now, user_id=user_id
                )
        if now.hour == 0:
            if rig.review is not None:
                for m in await rig.review.run(rig.tenant_id, now):
                    migrations.append(
                        MigrationObservation(
                            m.source_key, m.from_route.value, m.to_route.value,
                            SHIFTS.get(m.source_key, now), now,
                        )
                    )
            keys = naive.rewarm() if naive is not None else await rig.refresh.run(rig.tenant_id, now)
            for key in keys:
                for row in _probe_keys(key):
                    tracker.preloaded(name, row)
        if tick % PROBE_EVERY_HOURS == 0:
            probes: list[ProbeObservation] = []
            for feed in FEEDS:
                probes.append(
                    await _probe(stores, rig, feed, now, rig.owner_id, feed.profile.source_key)
                )
            owner_row, other_row = _probe_keys(SIZE_FEED.profile.source_key)
            probes.append(await _probe(stores, rig, SIZE_FEED, now, rig.owner_id, owner_row))
            probes.append(
                await _probe(stores, rig, SIZE_FEED, now, rig.other_id, other_row, size_markers)
            )
            for observation in probes:
                if observation.served_from_cag:
                    tracker.served(name, observation.source_key)
            observations.extend(probes)
    await rig.cascade.drain()
    return observations

async def _run_ttl_arm(stores: Stores, name: str, policy: FreshnessPolicy) -> list[ProbeObservation]:
    rig = await _build_rig(stores, name, route_for, review=False, policy=policy)
    source_id = source_id_for(rig.tenant_id, WARRANTY.profile.source_key, None)
    observations: list[ProbeObservation] = []
    for tick in range(HORIZON_HOURS + 1):
        now = T0 + tick * HOUR
        rig.clock.now = now
        if now.hour == 0 and now <= T0 + 6 * DAY:  # the feed confirms daily, and stalls after day 7
            await rig.ingest.execute(rig.tenant_id, WARRANTY.profile, WARRANTY.text(0), now)
        if now == T0 + 7 * DAY:  # day 8: the text changes in RAG behind the router's back
            await rig.rag_index.replace(
                rig.tenant_id, source_id, WARRANTY.profile.source_key, WARRANTY.text(1)
            )
        if now.hour == 0:
            await rig.refresh.run(rig.tenant_id, now)
        if tick % PROBE_EVERY_HOURS == 0:
            observations.append(
                await _probe(stores, rig, WARRANTY, now, rig.owner_id, WARRANTY.profile.source_key)
            )
    await rig.cascade.drain()
    return observations


def _spectrum() -> list[RouteRow]:
    policy = FreshnessPolicy()
    rows = []
    for data_type, change, scope, interval, paradigm in SPECTRUM:
        route = route_for(DataSourceProfile(data_type, scope, interval), policy)
        rows.append(
            RouteRow(data_type, change, scope.value, paradigm, route.value,
                     route is _SOURCE_ROUTE[paradigm])
        )
    return rows


def _calibrate(embedder: CachingEmbeddingModel) -> str:
    """Each question must reach its own source's text at both tiers' hit thresholds,
    and stay below the partial thresholds against every other source."""
    feeds = [*FEEDS, SIZE_FEED, WARRANTY]
    texts = {f.profile.source_key: embedder.embed(f.text(0)) for f in feeds}
    hit, partial = max(CAG_HIT, MAG_HIT), min(CAG_PARTIAL, MAG_PARTIAL)
    lines, failed = [], False
    for feed in feeds:
        question = embedder.embed(feed.question)
        scores = {key: cosine_similarity(question, vector) for key, vector in texts.items()}
        own = scores.pop(feed.profile.source_key)
        best_other = max(scores.values())
        failed |= own < hit or best_other >= partial
        lines.append(f"{feed.profile.source_key}: own {own:.2f}, best other {best_other:.2f}")
    if failed:
        raise SystemExit(
            f"question wording fails calibration (need own >= {hit}, other < {partial}):\n"
            + "\n".join(lines)
        )
    return "; ".join(lines)


async def _measure(app_url: str, qdrant_url: str, neo4j: tuple[str, str, str],
                   parts: frozenset[str]) -> None:
    engine = get_engine(app_url)
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    semantic_index = QdrantSemanticMemoryIndex(qdrant_url)
    await semantic_index.ensure_collection()
    graph = Neo4jMemoryGraphRepository(neo4j[0], auth=(neo4j[1], neo4j[2]))
    await graph.ensure_schema()
    stores = Stores(
        sessionmaker=get_sessionmaker(engine),
        vector_store=vector_store,
        semantic_index=semantic_index,
        graph=graph,
        embedder=CachingEmbeddingModel(SentenceTransformersEmbedder()),
        tokenizer=AutoTokenizer.from_pretrained("distilgpt2"),
        model=AutoModelForCausalLM.from_pretrained("distilgpt2"),
    )
    calibration = _calibrate(stores.embedder)
    routes = _spectrum() if "spectrum" in parts else []
    tracker, migrations, observations = PreloadTracker(), [], []
    if "ablation" in parts:
        for arm in ARMS:
            observations.extend(await _run_ablation_arm(stores, arm, tracker, migrations))
    tallies = tally_probes(observations)
    ttl_observations = []
    if "ttl" in parts:
        ttl_observations += await _run_ttl_arm(stores, "TTL factor 0.5", FreshnessPolicy())
        ttl_observations += await _run_ttl_arm(
            stores, "no TTL", FreshnessPolicy(ttl_factor=None)
        )
    notes = (
        f"Simulated clock from {T0:%Y-%m-%d} (day 1) for 30 days. Every source is ingested "
        f"hourly. Refresh runs at midnight, and review too in the freshness-aware arm. Every "
        f"source is probed every {PROBE_EVERY_HOURS} hours through Batch A's unrouted cascade "
        f"(CAG -> MAG -> RAG, first hit wins), with tier thresholds carried over unchanged: CAG "
        f"{CAG_HIT}/{CAG_PARTIAL}, MAG {MAG_HIT}/{MAG_PARTIAL}. Staleness is read from the "
        "assembled context using version-specific markers; no LLM runs. Schedules: prices "
        "change hourly; the return policy changes once, day 12 09:00; the catalog changes "
        "daily and hourly from day 10; the shipping guide never changes; the flash sale "
        "changes hourly until day 5 and then goes quiet; the owner's size preference changes "
        "day 15 09:00 and is probed by its owner and by another user of the same tenant. TTL "
        "part: warranty terms declared at 7 days, confirmed daily through day 7, changed "
        "directly in RAG on day 8. Question calibration: " + calibration + ". The change "
        "schedules are synthetic, one author wrote them and the rules, the cache is a CPU "
        "distilgpt2 proxy, and tier latency is cited from Batch A rather than re-measured "
        "here."
    )
    preloads = {
        (t.arm, t.source_key): tracker.counts(t.arm, t.source_key) for t in tallies
    }
    _REPORT.write_text(
        render_freshness_measurements(
            routes, tallies, preloads, migrations, tally_probes(ttl_observations), notes
        ),
        encoding="utf-8",
    )
    print(_REPORT.read_text(encoding="utf-8"))
    await graph.close()
    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freshness-Aware Data Router measurements.")
    parser.add_argument("--parts", default=",".join(_PARTS),
                        help="comma-separated subset of: " + ", ".join(_PARTS))
    parts = frozenset(
        part.strip() for part in parser.parse_args().parts.split(",") if part.strip()
    )
    unknown = parts - set(_PARTS)
    if unknown:
        parser.error(f"unknown parts: {sorted(unknown)}")
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # the report contains non-cp1252 characters
    with (
        PostgresContainer("pgvector/pgvector:pg16") as postgres,
        QdrantContainer("qdrant/qdrant:v1.16.2") as qdrant,
        Neo4jContainer("neo4j:5-community") as neo4j,
    ):
        # 127.0.0.1, not "localhost": asyncpg via "localhost" stalls on this Windows host.
        database_url = (
            make_url(postgres.get_connection_url())
            .set(drivername="postgresql+asyncpg", host="127.0.0.1")
            .render_as_string(hide_password=False)
        )
        os.environ["DATABASE_URL"] = database_url
        os.environ["APP_DB_PASSWORD"] = _APP_DB_PASSWORD
        command.upgrade(Config("alembic.ini"), "head")
        app_url = (
            make_url(database_url)
            .set(username="app_user", password=_APP_DB_PASSWORD)
            .render_as_string(hide_password=False)
        )
        neo4j_url = f"bolt://127.0.0.1:{neo4j.get_exposed_port(neo4j.port)}"
        asyncio.run(
            _measure(
                app_url,
                f"http://127.0.0.1:{qdrant.get_exposed_port(6333)}",
                (neo4j_url, neo4j.username, neo4j.password),
                parts,
            )
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint and dry-run the cheap part**

Run ruff on the runner. Then run: `PYTHONPATH=. ../../../.venv/Scripts/python.exe evaluation/scenarios/freshness-router/run_freshness_measurements.py --parts spectrum`

Expected: the spectrum table, 7 of 8 rows agreeing (code repositories disagree), and the calibration passing.

If calibration fails, reword the failing questions or source texts, never the thresholds. Repeat until it passes, and record the measured scores in the plan's execution notes.

- [ ] **Step 3: Run the full measurement in the background**

Run: `PYTHONPATH=. ../../../.venv/Scripts/python.exe evaluation/scenarios/freshness-router/run_freshness_measurements.py`

Run nothing CPU-heavy alongside it. Expect it to take tens of minutes.

Check the report against the arithmetic below, and investigate any mismatch with the systematic-debugging skill before writing about it.

**Ablation expectations:**
- **The naive arm:** stale-only and mixed contexts for `backpack-price`, `flash-sale` before day 5, and `shoe-catalog`.
- **The freshness-aware arm:** no stale-only context for any source.
- **Pre-loads:** the invalidate-on-change arm's never-served pre-loads exceed the freshness-aware arm's for `backpack-price`.
- **RAG only:** a 0% served-from-CAG share.
- **Foreign exposures:** above 0 for the size preference in every arm except freshness-aware.

**Migration expectations:**
- **Catalog:** demoted at day 11 00:00; its daily changes are exactly on the boundary, so it isn't demoted before its hourly phase.
- **Flash sale:** promoted at day 12 00:00, seven quiet days after its last change at day 5 00:00.

**TTL expectations** (20 or 185 stale CAG probes, bracketed by the half-open expiry):
- **With TTL factor 0.5:** the entry expires 3.5 days after the day-7 confirmation. That is about 2.5 days after the day-8 change, so about 20 probes.
- **With no TTL:** superseded text is served from CAG through day 30.

- [ ] **Step 4: Commit the runner and its report**

```bash
git add evaluation/scenarios/freshness-router/run_freshness_measurements.py evaluation/reports/freshness-router-measurements.md
git commit  # test: measure freshness routing over 30 simulated days against real stores
```

---

### Task 12: Report, documentation, review, and integration

**Files:**
- Create: `evaluation/reports/freshness-router.md` (the narrative report)
- Modify: `docs/architecture/OVERVIEW.md`, `docs/database/DATABASE.md`, `docs/testing/TESTING.md`, `docs/architecture/CONTEXT_GRAPH.md`, `README.md`
- Modify: `CLAUDE.md` (only through `claude-md-sync` proposals)
- Modify: this plan (execution notes)

- [ ] **Step 1: Full verification before touching docs**

Run, in order, and read every result:
1. `../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider`: all pass. Note the new count.
2. `../../../.venv/Scripts/python.exe -m pytest tests/integration -q -p no:cacheprovider -rs`: all pass except the 8 documented vLLM skips. Note the counts.
3. `../../../.venv/Scripts/python.exe -m ruff check` on every file this batch created or changed: clean.
4. `../../../.venv/Scripts/python.exe -m mypy src`: clean.

- [ ] **Step 2: Write `evaluation/reports/freshness-router.md`**

Structure it the way `evaluation/reports/orchestration-meta-layer.md` is structured, and take every number from `freshness-router-measurements.md`.

**Scope:** the issues; the spec, plan, and generated report paths.

**What this batch is:** the three use cases, the four ports, and what existing code changed:
- `CacheWarmedRetrieve.forget`;
- the concrete RAG deletes;
- migration 0006.

**One section per claim:**
- **Volatile data in a frozen cache.** Stale-only and mixed counts for the naive arm against the freshness-aware arm.
- **Stable data in RAG.** The served-from-CAG share for RAG only against freshness-aware, citing Batch A's tier latencies rather than re-measuring them.
- **Session data in MAG.** Foreign exposures per arm.
- **Migration.** The two lags, stated as bounded by the nightly review cadence and the hysteresis rules.
- **The TTL bound.** Stale CAG probes with the TTL against without it.

**Pre-load churn:** never-served pre-loads per arm, and what they cost in forward passes on this proxy.

**What building it changed:** every execution note recorded in this plan.

**What review caught:** filled after Step 6.

**What these measurements cannot show:**
- synthetic schedules;
- one author;
- a CPU cache proxy;
- a simulated clock;
- probes judged on context, not model answers;
- latency not re-measured.

**What this batch does not do:** the spec's list.

**The numbers:** unit and integration counts, and static checks.

Every claim cites the generated table it rests on. Differences too small for this run to distinguish from noise are not called effects. A claim the data doesn't support gets removed, not softened.

- [ ] **Step 3: Update the architecture, database, and testing docs**

**`docs/architecture/OVERVIEW.md`:**
- **The Freshness-Aware Data Router section.** Add what is built:
  - the module paths;
  - the three routes and the one-day boundary;
  - change handling: eviction on change, batch pre-load;
  - TTL anchoring;
  - the migration hysteresis.

  Cite `evaluation/reports/freshness-router.md` for the measured results.
- **The Phase 1 module blueprint.** Record that `freshness_router.py` and its use cases are built, so all five meta-layer components exist. What remains: an HTTP endpoint for ingestion and for the unified per-query path, `src/workers/` for scheduling refresh and review, the frontend, and Kubernetes.

**`docs/database/DATABASE.md`:**
- Describe `data_sources` and `data_source_versions`:
  - columns;
  - the scope/user `CHECK`;
  - the `NULLS NOT DISTINCT` uniqueness;
  - deterministic ids;
  - RLS from creation;
  - the version table as the change history migration reads.
- Update the table count: eight becomes ten.
- Add rows to the column tables.

**`docs/testing/TESTING.md`:**
- Refresh the unit and integration counts and file counts.
- Name the new real-store tests, including the MAG writer's and the end-to-end test's Neo4j dependency.
- Record that the freshness runner needs Docker for Postgres, Qdrant, and Neo4j, but no LLM.

**`README.md`:** the built list, what remains ahead, and the test counts.

- [ ] **Step 4: Regenerate the context graph**

Rerun graphify's AST pipeline over `src/`, under a `__main__` guard (see the memory note on running graphify on Windows).

Redraw `REAL_ORCH`:
- **Class counts:** cross-check them with a standard-library `ast` count.
- **The new nodes:** `freshness_router.py`, the three use cases, and the four adapters.
- **The `FUTURE` node:** narrowed to the HTTP endpoints, workers, and frontend.
- **Generation note and counts:** updated, disclosing graphify's sensitive-file skip of the two token-named identity files.

- [ ] **Step 5: Sync CLAUDE.md**

Invoke `claude-md-sync`. Apply each proposal grounded in this batch's history (test counts, the built and unbuilt list, the graph generation, the list of spec-first batches), and say in the commit message that the user granted full control for this session.

- [ ] **Step 6: Commit the docs, then review**

1. Commit the docs (`docs: record the built freshness router and its measured behaviour`).
2. Invoke `superpowers:requesting-code-review` over `develop..HEAD`.
3. Check each finding against the code.
4. Fix confirmed findings test-first, one commit per concern.
5. Rerun Step 1, and a measurement part if a fix touches it.
6. Update the narrative report's review section, and commit.

- [ ] **Step 7: Integrate**

Invoke `superpowers:finishing-a-development-branch` and take the standing path:
1. Merge to `develop` locally with a `merge:` commit.
2. Rerun the unit suite on the merged result.
3. Remove the worktree and delete the branch.
4. Close this batch's GitHub issues with a comment naming the merge commit and the report.
5. Close Story #155, and close Epic #150 once #168 is its only open child; if #168 is still open, leave the epic open with a comment saying why.
