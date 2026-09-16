import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from typing import Any


@dataclass(frozen=True)
class CacheHit:
    # kv_cache is an opaque handle -- only the FrozenCache implementation
    # that produced it knows its real shape (a transformers DynamicCache in
    # this batch's one implementation). The domain layer never inspects it,
    # only passes it through, the same "opaque payload, typed at the
    # infrastructure boundary" shape CAG Batch B's CompressedKV established.
    content_hash: str
    kv_cache: Any


@dataclass(frozen=True)
class SyncConflict:
    # A real, detected disagreement between what the hot tier's cached
    # entry for document_id holds and what the paradigm-specific
    # authoritative source currently says -- RAG for SyncCycle and
    # MagSyncCycle, MAG for CagMagSyncCycle (see domain/sync_mixer.py,
    # which doesn't itself know or care which side is authoritative).
    # Always produced already-resolved (by eviction), never a pending
    # conflict a caller still has to act on.
    document_id: uuid.UUID
    cached_content_hash: str
    authoritative_content_hash: str


class TierDecision(Enum):
    PROMOTED = "promoted"
    DEMOTED = "demoted"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class WarmEntry:
    # MAG's warm-tier analogue of CacheHit. Unlike CacheHit.kv_cache (an
    # opaque tensor handle only infrastructure understands), `content` is a
    # real string here: MAG's warm tier is a semantic fact, not a KV
    # cache, so there's no opaque payload to hide -- the domain layer can
    # see and use the real text (State-Aware RAG's ranking boost and
    # query enrichment both need to read it).
    content_hash: str
    content: str


class Paradigm(Enum):
    CAG = "cag"
    MAG = "mag"
    RAG = "rag"


# Cheapest first -- the order OVERVIEW.md's fallback cascade tries tiers in.
# The router breaks score ties in this order and context assembly renders
# sections in it, so all three agree on one ordering.
PARADIGM_ORDER: tuple[Paradigm, ...] = (Paradigm.CAG, Paradigm.MAG, Paradigm.RAG)


class RoutingMode(Enum):
    CASCADE = "cascade"
    PARALLEL = "parallel"


@dataclass(frozen=True)
class RoutingDecision:
    paradigms: frozenset[Paradigm]
    mode: RoutingMode
    scores: dict[Paradigm, float]


class TierOutcome(Enum):
    HIT = "hit"
    PARTIAL = "partial"
    MISS = "miss"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class ContextItem:
    paradigm: Paradigm
    content: str
    score: float
    source_id: uuid.UUID | None = None


@dataclass(frozen=True)
class TierRequest:
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    session_id: uuid.UUID
    query: str
    query_embedding: list[float]


_TIER_REPORTABLE_OUTCOMES = frozenset({TierOutcome.HIT, TierOutcome.PARTIAL, TierOutcome.MISS})


@dataclass(frozen=True)
class TierResult:
    outcome: TierOutcome
    items: list[ContextItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        # TIMEOUT and ERROR describe what happened TO a tier, which only the
        # cascade observing it can know -- a tier able to report its own
        # timeout would not have timed out.
        if self.outcome not in _TIER_REPORTABLE_OUTCOMES:
            raise ValueError(
                f"a tier cannot report {self.outcome.value}; only the cascade records it"
            )
        if self.outcome is TierOutcome.MISS and self.items:
            raise ValueError("a MISS carries no items")


@dataclass(frozen=True)
class TierAttempt:
    paradigm: Paradigm
    outcome: TierOutcome
    elapsed_ms: float


@dataclass(frozen=True)
class CascadeResult:
    items: list[ContextItem]
    attempts: list[TierAttempt]
    satisfied: frozenset[Paradigm]
    degraded: bool

    @property
    def contributing(self) -> frozenset[Paradigm]:
        return frozenset(item.paradigm for item in self.items)


@dataclass(frozen=True)
class BudgetShares:
    # OVERVIEW.md's default 128K split ("Slicing the context window").
    cag: float = 0.40
    mag: float = 0.25
    rag: float = 0.20
    query: float = 0.10
    reserve: float = 0.05

    def __post_init__(self) -> None:
        values = (self.cag, self.mag, self.rag, self.query, self.reserve)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("budget shares must be finite and non-negative")
        # 1e-12 absorbs float noise in the sum (the defaults sum to exactly
        # 1.0) while keeping any real overshoot from flooring the slices past
        # the window and driving reserve negative at very large totals.
        # rel_tol=0.0 explicitly: isclose's default rel_tol of 1e-9 would
        # still let a 5e-10 overshoot through, whatever abs_tol says.
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"budget shares must sum to 1.0, got {sum(values)}")

    def for_paradigm(self, paradigm: Paradigm) -> float:
        return {Paradigm.CAG: self.cag, Paradigm.MAG: self.mag, Paradigm.RAG: self.rag}[paradigm]


@dataclass(frozen=True)
class BudgetAllocation:
    cag: int
    mag: int
    rag: int
    query: int
    reserve: int

    @property
    def total(self) -> int:
        return self.cag + self.mag + self.rag + self.query + self.reserve

    def for_paradigm(self, paradigm: Paradigm) -> int:
        return {Paradigm.CAG: self.cag, Paradigm.MAG: self.mag, Paradigm.RAG: self.rag}[paradigm]


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
    """The freshness router's disclosed defaults (spec decisions 3, 6, and 7)."""

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
            # Otherwise one version on the window's edge could satisfy both rules.
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
    # The hash of a change whose effects have started but whose save hasn't landed.
    # While it is set, refresh and review leave the source alone, and the next ingestion
    # re-applies the change even if the feed has reverted to the stored content.
    pending_hash: str | None = None


@dataclass(frozen=True)
class SourceVersion:
    """A changed version, recorded alongside the source that now points at it.

    content is None for a MAG-routed source: its text lives in that user's MAG fact, and
    the registry keeps no second copy of personal data.
    """

    content: str | None


@dataclass(frozen=True)
class IngestionResult:
    source_id: uuid.UUID
    route: IngestionRoute
    changed: bool


class JobState(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class JobStatus:
    state: JobState
    result: IngestionResult | None
    error: str | None


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
