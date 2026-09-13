import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
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
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
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
