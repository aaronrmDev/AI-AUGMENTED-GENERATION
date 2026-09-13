# Orchestration Meta-Layer Batch A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and measure the Paradigm Router, Context Budget Allocator, and Latency-Adaptive Fallback Cascade, composed into one `UnifiedAnswerQuestion` use case.

**Architecture:** Pure routing and allocation logic lives in `src/orchestration/domain/`. The cascade, the tier adapters over existing CAG/MAG/RAG use cases, context assembly, and the unified use case live in `src/orchestration/application/`. The three query classifiers and the Postgres budget recorder live in `src/orchestration/infrastructure/`. Evaluation runners measure the three claims the spec names.

**Tech Stack:** Python 3.14 venv (ruff target py311, mypy strict on `src/`), pytest with `asyncio_mode = "auto"`, SQLAlchemy async + asyncpg, TestContainers (pgvector/pgvector:pg16, qdrant/qdrant:v1.16.2), sentence-transformers MiniLM, transformers `distilgpt2`, tiktoken `cl100k_base`, Ollama `qwen3.5`.

**Spec:** `docs/superpowers/specs/2026-09-13-orchestration-meta-layer-design.md`

## Global Constraints

- Work only inside `.worktrees/feature/orchestration-meta-layer/`; every path below is relative to that directory.
- Run Python as `../../../.venv/Scripts/python.exe` (the main checkout's venv; it imports this worktree's `src/`).
- Unit tests: `../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider`. Baseline before this batch: 777 passed.
- Lint: `../../../.venv/Scripts/python.exe -m ruff check src tests evaluation`; types: `../../../.venv/Scripts/python.exe -m mypy src`.
- Line length 100. Every tier and port is tenant-scoped; MAG access is also user-scoped.
- Unit tests never assert wall-clock latency; timeout tests use fake delays far beyond the budget (0.5s against 0.05s).
- Tier hit/partial thresholds in `src/` have no defaults; callers pass measured values. (The router's 0.5 select threshold and 0.15 margin are the spec's stated defaults.)
- `logger = logging.getLogger(__name__)` is the logging convention.
- Commit messages follow `.gitmessage` (Conventional Commits) and end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## File map

| File | Responsibility |
|---|---|
| `src/orchestration/domain/entities.py` (modify) | `Paradigm`, `PARADIGM_ORDER`, `RoutingMode`, `RoutingDecision`, `TierOutcome`, `ContextItem`, `TierRequest`, `TierResult`, `TierAttempt`, `CascadeResult`, `BudgetShares`, `BudgetAllocation` |
| `src/orchestration/domain/similarity.py` | `cosine_similarity` |
| `src/orchestration/domain/errors.py` | `SessionNotFound`, `QueryExceedsBudget` |
| `src/orchestration/domain/paradigm_router.py` | `decide` |
| `src/orchestration/domain/budget_allocator.py` | `allocate`, `DEFAULT_SHARES` |
| `src/orchestration/domain/ports.py` (modify) | `QueryClassifier`, `CascadeTier`, `SessionBudgetRecorder` |
| `src/orchestration/application/assemble_context.py` | `assemble_context`, `AssembledContext` |
| `src/orchestration/application/latency_cascade.py` | `LatencyCascade`, `TierTimeouts` |
| `src/orchestration/application/cache_warmed_retrieve.py` (modify) | public `best_warmed_match` |
| `src/orchestration/application/cascade_tiers.py` | `CagTier`, `MagTier`, `RagTier` |
| `src/orchestration/application/unified_answer_question.py` | `UnifiedAnswerQuestion`, `UnifiedAnswer`, `StageTimings` |
| `src/orchestration/infrastructure/lexical_query_classifier.py` | `LexicalQueryClassifier` |
| `src/orchestration/infrastructure/routing_exemplars.py` | `RoutingExemplar`, `DEFAULT_ROUTING_EXEMPLARS` |
| `src/orchestration/infrastructure/prototype_query_classifier.py` | `PrototypeQueryClassifier` |
| `src/orchestration/infrastructure/llm_query_classifier.py` | `LlmQueryClassifier` |
| `src/orchestration/infrastructure/postgres_session_budget_recorder.py` | `PostgresSessionBudgetRecorder`, `budget_record` |
| `evaluation/domain/routing_metrics.py` | `RoutingObservation`, `RoutingMetrics`, `summarize` |
| `evaluation/scenarios/orchestration-meta-layer/` | query generation, router runner, cascade runner, queries |
| `evaluation/reports/orchestration-meta-layer*.md` | generated tables and the narrative report |

---

### Task 1: Domain entities, similarity, errors, and the Paradigm Router

**Files:**
- Modify: `src/orchestration/domain/entities.py`
- Create: `src/orchestration/domain/similarity.py`, `src/orchestration/domain/errors.py`, `src/orchestration/domain/paradigm_router.py`
- Test: `tests/unit/test_orchestration_entities.py`, `tests/unit/test_similarity.py`, `tests/unit/test_paradigm_router.py`

**Interfaces:**
- Produces: everything in the file map's first four rows. `decide(scores: Mapping[Paradigm, float], select_threshold: float = 0.5, uncertainty_margin: float = 0.15) -> RoutingDecision`. `BudgetShares.for_paradigm(p) -> float`, `BudgetAllocation.for_paradigm(p) -> int`, `BudgetAllocation.total`. `CascadeResult.contributing -> frozenset[Paradigm]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_similarity.py`:

```python
import pytest

from src.orchestration.domain.similarity import cosine_similarity


def test_identical_vectors_score_one():
    assert cosine_similarity([0.6, 0.8], [0.6, 0.8]) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_a_zero_vector_scores_zero_instead_of_dividing_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_vectors_of_different_lengths_are_rejected():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0])
```

`tests/unit/test_orchestration_entities.py`:

```python
import uuid

import pytest

from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    BudgetAllocation,
    BudgetShares,
    CascadeResult,
    ContextItem,
    Paradigm,
    TierOutcome,
    TierResult,
)


def test_paradigm_order_is_cheapest_first():
    assert PARADIGM_ORDER == (Paradigm.CAG, Paradigm.MAG, Paradigm.RAG)


@pytest.mark.parametrize("outcome", [TierOutcome.TIMEOUT, TierOutcome.ERROR])
def test_a_tier_cannot_report_an_outcome_only_the_cascade_can_observe(outcome):
    with pytest.raises(ValueError):
        TierResult(outcome)


def test_a_miss_carries_no_items():
    with pytest.raises(ValueError):
        TierResult(TierOutcome.MISS, [ContextItem(Paradigm.CAG, "x", 0.1)])


def test_a_hit_carries_its_items():
    item = ContextItem(Paradigm.RAG, "x", 0.9, uuid.uuid4())
    assert TierResult(TierOutcome.HIT, [item]).items == [item]


def test_contributing_is_the_set_of_paradigms_that_returned_items():
    result = CascadeResult(
        items=[ContextItem(Paradigm.CAG, "a", 0.5), ContextItem(Paradigm.RAG, "b", 0.9)],
        attempts=[],
        satisfied=frozenset({Paradigm.RAG}),
        degraded=False,
    )
    assert result.contributing == frozenset({Paradigm.CAG, Paradigm.RAG})


def test_default_shares_are_the_source_split():
    shares = BudgetShares()
    assert (shares.cag, shares.mag, shares.rag, shares.query, shares.reserve) == (
        0.40, 0.25, 0.20, 0.10, 0.05,
    )
    assert shares.for_paradigm(Paradigm.MAG) == 0.25


def test_shares_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(ValueError):
        BudgetShares(cag=0.5, mag=0.5, rag=0.5, query=0.0, reserve=0.0)


def test_negative_shares_are_rejected():
    with pytest.raises(ValueError):
        BudgetShares(cag=-0.1, mag=0.35, rag=0.5, query=0.2, reserve=0.05)


def test_allocation_total_and_per_paradigm_lookup():
    allocation = BudgetAllocation(cag=4, mag=3, rag=2, query=1, reserve=1)
    assert allocation.total == 11
    assert allocation.for_paradigm(Paradigm.RAG) == 2
```

`tests/unit/test_paradigm_router.py`:

```python
import math

import pytest

from src.orchestration.domain.entities import Paradigm, RoutingMode
from src.orchestration.domain.paradigm_router import decide

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


def _scores(cag: float, mag: float, rag: float) -> dict[Paradigm, float]:
    return {CAG: cag, MAG: mag, RAG: rag}


@pytest.mark.parametrize(
    ("query", "scores", "expected"),
    [
        ("What's our refund policy?", _scores(0.9, 0.1, 0.2), {CAG}),
        ("What changed in the policy today?", _scores(0.2, 0.0, 0.9), {RAG}),
        ("Continue where we left off yesterday", _scores(0.0, 0.95, 0.1), {MAG}),
        ("Compare today's sales with last month", _scores(0.1, 0.8, 0.85), {MAG, RAG}),
        ("Explain this code file", _scores(0.85, 0.0, 0.25), {CAG}),
        ("What did I ask you to remember?", _scores(0.0, 1.0, 0.0), {MAG}),
    ],
)
def test_the_six_concept_one_queries_route_as_the_source_table_says(query, scores, expected):
    decision = decide(scores)
    assert decision.paradigms == frozenset(expected), query
    assert decision.mode is RoutingMode.CASCADE


def test_a_selected_score_inside_the_uncertainty_band_runs_parallel():
    decision = decide(_scores(0.9, 0.55, 0.1))
    assert decision.paradigms == frozenset({CAG, MAG})
    assert decision.mode is RoutingMode.PARALLEL


def test_a_borderline_score_just_below_the_threshold_joins_a_parallel_route():
    decision = decide(_scores(0.9, 0.1, 0.42))
    assert decision.paradigms == frozenset({CAG, RAG})
    assert decision.mode is RoutingMode.PARALLEL


def test_when_nothing_clears_the_threshold_the_highest_score_is_still_selected():
    decision = decide(_scores(0.1, 0.3, 0.2))
    assert decision.paradigms == frozenset({MAG})
    assert decision.mode is RoutingMode.CASCADE


def test_an_all_zero_tie_breaks_cheapest_first():
    assert decide(_scores(0.0, 0.0, 0.0)).paradigms == frozenset({CAG})


def test_a_tie_between_mag_and_rag_breaks_toward_mag():
    assert decide(_scores(0.0, 0.2, 0.2)).paradigms == frozenset({MAG})


def test_a_zero_margin_never_goes_parallel():
    decision = decide(_scores(0.5, 0.5, 0.5), uncertainty_margin=0.0)
    assert decision.paradigms == frozenset({CAG, MAG, RAG})
    assert decision.mode is RoutingMode.CASCADE


def test_the_scores_are_kept_on_the_decision():
    scores = _scores(0.9, 0.1, 0.2)
    assert decide(scores).scores == scores


def test_scores_missing_a_paradigm_are_rejected():
    with pytest.raises(ValueError):
        decide({CAG: 0.5, MAG: 0.5})


@pytest.mark.parametrize("bad", [1.2, -0.1, math.nan])
def test_scores_outside_the_unit_interval_are_rejected(bad):
    with pytest.raises(ValueError):
        decide(_scores(bad, 0.0, 0.0))


def test_a_negative_margin_is_rejected():
    with pytest.raises(ValueError):
        decide(_scores(0.9, 0.1, 0.1), uncertainty_margin=-0.1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_similarity.py tests/unit/test_orchestration_entities.py tests/unit/test_paradigm_router.py -q -p no:cacheprovider`
Expected: collection errors (`ModuleNotFoundError` / `ImportError`).

- [ ] **Step 3: Implement**

Append to `src/orchestration/domain/entities.py` (add `import math` and `field` to the existing imports):

```python
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
```

`src/orchestration/domain/similarity.py`:

```python
import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
```

`src/orchestration/domain/errors.py`:

```python
import uuid


class SessionNotFound(Exception):
    def __init__(self, session_id: uuid.UUID) -> None:
        super().__init__(f"no session {session_id} is visible under the current tenant")
        self.session_id = session_id


class QueryExceedsBudget(Exception):
    def __init__(self, query_tokens: int, query_slice: int) -> None:
        super().__init__(
            f"the question is {query_tokens} tokens; the Query slice allows {query_slice}"
        )
        self.query_tokens = query_tokens
        self.query_slice = query_slice
```

`src/orchestration/domain/paradigm_router.py`:

```python
import math
from collections.abc import Mapping

from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    Paradigm,
    RoutingDecision,
    RoutingMode,
)

DEFAULT_SELECT_THRESHOLD = 0.5
DEFAULT_UNCERTAINTY_MARGIN = 0.15


def decide(
    scores: Mapping[Paradigm, float],
    select_threshold: float = DEFAULT_SELECT_THRESHOLD,
    uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> RoutingDecision:
    """Concept 1's routing decision over independent per-paradigm scores.

    Multi-label on purpose: the source's own table routes "Compare today's
    sales with last month" to RAG and MAG together. A decision is never
    empty, and it is confident only when no score sits within
    `uncertainty_margin` of the threshold; otherwise the paradigms inside
    that band join the route and it runs PARALLEL ("if classifier is
    uncertain, run parallel and merge").
    """
    if set(scores) != set(PARADIGM_ORDER):
        raise ValueError("scores must contain exactly one entry per paradigm")
    for paradigm, score in scores.items():
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{paradigm.value} score {score} is outside [0, 1]")
    if uncertainty_margin < 0.0:
        raise ValueError("uncertainty_margin must be non-negative")

    selected = {p for p in PARADIGM_ORDER if scores[p] >= select_threshold}
    if not selected:
        # max() keeps the first of equal maxima, so iterating PARADIGM_ORDER
        # breaks ties cheapest-first.
        selected = {max(PARADIGM_ORDER, key=lambda p: scores[p])}
    uncertain = {
        p for p in PARADIGM_ORDER if abs(scores[p] - select_threshold) < uncertainty_margin
    }

    if uncertain:
        return RoutingDecision(frozenset(selected | uncertain), RoutingMode.PARALLEL, dict(scores))
    return RoutingDecision(frozenset(selected), RoutingMode.CASCADE, dict(scores))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_similarity.py tests/unit/test_orchestration_entities.py tests/unit/test_paradigm_router.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/domain tests/unit/test_similarity.py tests/unit/test_orchestration_entities.py tests/unit/test_paradigm_router.py
git commit  # feat: add meta-layer domain entities and the Paradigm Router's decision rule
```

---

### Task 2: Context Budget Allocator

**Files:**
- Create: `src/orchestration/domain/budget_allocator.py`
- Test: `tests/unit/test_budget_allocator.py`

**Interfaces:**
- Consumes: `BudgetShares`, `BudgetAllocation`, `Paradigm`, `PARADIGM_ORDER` (Task 1).
- Produces: `DEFAULT_SHARES: BudgetShares`; `allocate(total_tokens: int, contributing: Iterable[Paradigm], shares: BudgetShares = DEFAULT_SHARES) -> BudgetAllocation`.

- [ ] **Step 1: Write the failing test**

```python
import itertools

import pytest

from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import BudgetShares, Paradigm

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


def _slices(allocation):
    return (allocation.cag, allocation.mag, allocation.rag, allocation.query, allocation.reserve)


def test_every_paradigm_contributing_gets_the_source_128k_split():
    assert _slices(allocate(128_000, {CAG, MAG, RAG})) == (51_200, 32_000, 25_600, 12_800, 6_400)


def test_an_idle_rag_slice_expands_mag():
    assert _slices(allocate(128_000, {CAG, MAG})) == (51_200, 57_600, 0, 12_800, 6_400)


def test_an_idle_mag_slice_splits_between_cag_and_rag_by_base_share():
    # 32_000 * 0.4/0.6 = 21_333.3 -> 21_333; 32_000 * 0.2/0.6 = 10_666.6 -> 10_666;
    # the one leftover token goes to reserve.
    assert _slices(allocate(128_000, {CAG, RAG})) == (72_533, 0, 36_266, 12_800, 6_401)


def test_a_cag_miss_expands_rag():
    assert _slices(allocate(128_000, {MAG, RAG})) == (0, 32_000, 76_800, 12_800, 6_400)


def test_cag_alone_absorbs_both_idle_slices():
    assert _slices(allocate(128_000, {CAG})) == (108_800, 0, 0, 12_800, 6_400)


def test_mag_alone_absorbs_both_idle_slices():
    assert _slices(allocate(128_000, {MAG})) == (0, 108_800, 0, 12_800, 6_400)


def test_rag_alone_absorbs_both_idle_slices():
    assert _slices(allocate(128_000, {RAG})) == (0, 0, 108_800, 12_800, 6_400)


def test_nothing_contributing_moves_every_paradigm_slice_into_reserve():
    assert _slices(allocate(128_000, set())) == (0, 0, 0, 12_800, 115_200)


_SUBSETS = [
    set(combo) for size in range(4) for combo in itertools.combinations((CAG, MAG, RAG), size)
]


@pytest.mark.parametrize("total", [0, 1, 7, 99, 1_000, 4_096, 128_000, 200_003])
@pytest.mark.parametrize("contributing", _SUBSETS)
def test_the_five_slices_always_sum_to_the_total_and_are_never_negative(total, contributing):
    allocation = allocate(total, contributing)
    assert allocation.total == total
    assert min(_slices(allocation)) >= 0


def test_the_order_contributing_arrives_in_does_not_change_the_allocation():
    assert allocate(128_000, [RAG, CAG]) == allocate(128_000, [CAG, RAG])


def test_custom_shares_are_honored():
    shares = BudgetShares(cag=0.5, mag=0.2, rag=0.2, query=0.05, reserve=0.05)
    assert _slices(allocate(1_000, {CAG, MAG, RAG}, shares)) == (500, 200, 200, 50, 50)


def test_a_negative_total_is_rejected():
    with pytest.raises(ValueError):
        allocate(-1, {CAG})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_budget_allocator.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'src.orchestration.domain.budget_allocator'`.

- [ ] **Step 3: Implement**

```python
import math
from collections.abc import Iterable

from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    BudgetAllocation,
    BudgetShares,
    Paradigm,
)

DEFAULT_SHARES = BudgetShares()


def _recipients(idle: Paradigm, active: frozenset[Paradigm]) -> tuple[Paradigm, ...]:
    # Each branch is one of Concept 3's reallocation rules, read literally.
    if idle is Paradigm.RAG:  # "if query is simple (no RAG needed) -> MAG slice expands"
        return next(((p,) for p in (Paradigm.MAG, Paradigm.CAG) if p in active), ())
    if idle is Paradigm.CAG:  # "if CAG cache misses -> RAG slice temporarily expands"
        return next(((p,) for p in (Paradigm.RAG, Paradigm.MAG) if p in active), ())
    # "if session is new (no MAG state) -> CAG or RAG slice expands": the
    # source doesn't choose, so both share it in proportion to base share.
    return tuple(p for p in (Paradigm.CAG, Paradigm.RAG) if p in active)


def allocate(
    total_tokens: int, contributing: Iterable[Paradigm], shares: BudgetShares = DEFAULT_SHARES
) -> BudgetAllocation:
    """Slice the context window for one turn.

    Every idle paradigm donates its whole BASE slice exactly once, so no
    donation is ever re-donated and the result doesn't depend on the order
    rules are applied in. Anything with no eligible recipient, plus every
    flooring remainder, lands in reserve, so the slices always sum to
    total_tokens. The source's fourth rule ("MAG state grows too large ->
    compression or eviction") never expands a slice; assemble_context
    enforces it by dropping lowest-scoring items.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    active = frozenset(contributing)
    base = {p: math.floor(total_tokens * shares.for_paradigm(p)) for p in PARADIGM_ORDER}
    query = math.floor(total_tokens * shares.query)
    reserve = total_tokens - sum(base.values()) - query
    slices = dict(base)

    for idle in PARADIGM_ORDER:
        if idle in active:
            continue
        donation = base[idle]
        slices[idle] -= donation
        recipients = _recipients(idle, active)
        if not recipients:
            reserve += donation
            continue
        if len(recipients) == 1:
            slices[recipients[0]] += donation
            continue
        weight_total = sum(shares.for_paradigm(p) for p in recipients)
        if weight_total <= 0.0:
            reserve += donation
            continue
        given = 0
        for recipient in recipients:
            portion = math.floor(donation * shares.for_paradigm(recipient) / weight_total)
            slices[recipient] += portion
            given += portion
        reserve += donation - given

    return BudgetAllocation(
        cag=slices[Paradigm.CAG],
        mag=slices[Paradigm.MAG],
        rag=slices[Paradigm.RAG],
        query=query,
        reserve=reserve,
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_budget_allocator.py -q -p no:cacheprovider`
Expected: all pass (the parametrized sum test contributes 64 cases).

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/domain/budget_allocator.py tests/unit/test_budget_allocator.py
git commit  # feat: add the Context Budget Allocator's per-turn slicing and reallocation rules
```

---

### Task 3: Ports, test fakes, and context assembly

**Files:**
- Modify: `src/orchestration/domain/ports.py`, `tests/unit/orchestration_fakes.py`
- Create: `src/orchestration/application/assemble_context.py`
- Test: `tests/unit/test_assemble_context.py`

**Interfaces:**
- Consumes: Task 1 entities.
- Produces:
  - `QueryClassifier.score(query: str, query_embedding: list[float]) -> dict[Paradigm, float]` (async)
  - `CascadeTier.paradigm -> Paradigm` (property), `CascadeTier.attempt(request: TierRequest) -> TierResult` (async)
  - `SessionBudgetRecorder.record(tenant_id, session_id, allocation: BudgetAllocation, contributing: frozenset[Paradigm]) -> None` (async)
  - `AssembledContext(text: str, included: list[ContextItem], dropped: dict[Paradigm, int], tokens_used: dict[Paradigm, int])`
  - `assemble_context(items: list[ContextItem], allocation: BudgetAllocation) -> AssembledContext`
  - Fakes: `FakeCascadeTier(paradigm, result=None, delay_seconds=0.0, error=None)` with `.requests` and `.cancelled`; `FakeQueryClassifier(scores)` with `.calls`; `FakeSessionBudgetRecorder()` with `.records`.

- [ ] **Step 1: Add the ports**

Append to `src/orchestration/domain/ports.py` and extend its entities import with `BudgetAllocation, Paradigm, TierRequest, TierResult`:

```python
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


class SessionBudgetRecorder(ABC):
    @abstractmethod
    async def record(
        self,
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        allocation: BudgetAllocation,
        contributing: frozenset[Paradigm],
    ) -> None: ...
```

- [ ] **Step 2: Add the fakes**

Append to `tests/unit/orchestration_fakes.py` (add `import asyncio`; extend imports with `BudgetAllocation, Paradigm, TierOutcome, TierRequest, TierResult` from entities and `CascadeTier, QueryClassifier, SessionBudgetRecorder` from ports):

```python
class FakeCascadeTier(CascadeTier):
    def __init__(
        self,
        paradigm: Paradigm,
        result: TierResult | None = None,
        delay_seconds: float = 0.0,
        error: BaseException | None = None,
    ) -> None:
        self._paradigm = paradigm
        self._result = result if result is not None else TierResult(TierOutcome.MISS)
        self._delay = delay_seconds
        self._error = error
        self.requests: list[TierRequest] = []
        self.cancelled = False

    @property
    def paradigm(self) -> Paradigm:
        return self._paradigm

    async def attempt(self, request: TierRequest) -> TierResult:
        self.requests.append(request)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self._error is not None:
            raise self._error
        return self._result


class FakeQueryClassifier(QueryClassifier):
    def __init__(self, scores: dict[Paradigm, float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[float]]] = []

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        self.calls.append((query, query_embedding))
        return dict(self._scores)


class FakeSessionBudgetRecorder(SessionBudgetRecorder):
    def __init__(self) -> None:
        self.records: list[
            tuple[uuid.UUID, uuid.UUID, BudgetAllocation, frozenset[Paradigm]]
        ] = []

    async def record(
        self,
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        allocation: BudgetAllocation,
        contributing: frozenset[Paradigm],
    ) -> None:
        self.records.append((tenant_id, session_id, allocation, contributing))
```

- [ ] **Step 3: Write the failing assembly test `tests/unit/test_assemble_context.py`**

```python
from src.orchestration.application.assemble_context import assemble_context
from src.orchestration.domain.entities import BudgetAllocation, ContextItem, Paradigm
from src.shared.tokenization import count_tokens

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG
_LARGE = "alpha " * 60


def _budget(cag=1_000, mag=1_000, rag=1_000):
    return BudgetAllocation(cag=cag, mag=mag, rag=rag, query=0, reserve=0)


def test_sections_render_cheapest_first_with_paradigm_labels_regardless_of_input_order():
    items = [
        ContextItem(RAG, "retrieved text", 0.9),
        ContextItem(CAG, "cached text", 0.8),
        ContextItem(MAG, "remembered text", 0.7),
    ]
    text = assemble_context(items, _budget()).text
    assert text.index("(CAG)") < text.index("cached text") < text.index("(MAG)")
    assert text.index("remembered text") < text.index("(RAG)") < text.index("retrieved text")


def test_items_pack_highest_score_first_and_an_oversized_item_does_not_block_a_smaller_one():
    big = ContextItem(RAG, _LARGE, 0.9)
    small = ContextItem(RAG, "small fact", 0.5)
    assembled = assemble_context([small, big], _budget(rag=10))
    assert assembled.included == [small]
    assert assembled.dropped[RAG] == 1
    assert assembled.tokens_used[RAG] == count_tokens("small fact")


def test_higher_scoring_items_win_the_budget_when_both_cannot_fit():
    first = ContextItem(MAG, "one two three", 0.9)
    second = ContextItem(MAG, "four five six", 0.4)
    budget = count_tokens("one two three")
    assembled = assemble_context([second, first], _budget(mag=budget))
    assert assembled.included == [first]
    assert assembled.dropped[MAG] == 1


def test_a_paradigm_with_a_zero_slice_drops_everything_it_returned():
    assembled = assemble_context([ContextItem(CAG, "cached", 0.9)], _budget(cag=0))
    assert assembled.included == []
    assert assembled.dropped[CAG] == 1
    assert "(CAG)" not in assembled.text


def test_no_items_assemble_to_an_empty_context():
    assembled = assemble_context([], _budget())
    assert assembled.text == ""
    assert assembled.dropped == {CAG: 0, MAG: 0, RAG: 0}
```

- [ ] **Step 4: Run it to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_assemble_context.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError`.

- [ ] **Step 5: Implement `src/orchestration/application/assemble_context.py`**

```python
from dataclasses import dataclass

from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    BudgetAllocation,
    ContextItem,
    Paradigm,
)
from src.shared.tokenization import count_tokens

_SECTION_TITLES = {
    Paradigm.CAG: "Pre-loaded reference knowledge (CAG)",
    Paradigm.MAG: "What is known about this user (MAG)",
    Paradigm.RAG: "Retrieved documents (RAG)",
}


@dataclass(frozen=True)
class AssembledContext:
    text: str
    included: list[ContextItem]
    dropped: dict[Paradigm, int]
    tokens_used: dict[Paradigm, int]


def assemble_context(items: list[ContextItem], allocation: BudgetAllocation) -> AssembledContext:
    """Pack each paradigm's items into its own slice, best score first.

    The walk skips an item that doesn't fit and keeps going, the same
    shape TokenBudgetAllocation and CompressingRetriever already use, so
    one oversized item never shuts out a smaller one. Dropped counts are
    how the source's "state grows too large -> eviction" rule becomes
    visible per turn. Section headers are not charged to any slice; the
    reserve absorbs those few tokens.
    """
    sections: list[str] = []
    included: list[ContextItem] = []
    dropped = dict.fromkeys(PARADIGM_ORDER, 0)
    tokens_used = dict.fromkeys(PARADIGM_ORDER, 0)
    for paradigm in PARADIGM_ORDER:
        budget = allocation.for_paradigm(paradigm)
        ranked = sorted(
            (item for item in items if item.paradigm is paradigm),
            key=lambda item: item.score,
            reverse=True,
        )
        kept: list[ContextItem] = []
        for item in ranked:
            cost = count_tokens(item.content)
            if tokens_used[paradigm] + cost > budget:
                dropped[paradigm] += 1
                continue
            tokens_used[paradigm] += cost
            kept.append(item)
        if kept:
            body = "\n\n".join(item.content for item in kept)
            sections.append(f"## {_SECTION_TITLES[paradigm]}\n{body}")
            included.extend(kept)
    return AssembledContext("\n\n".join(sections), included, dropped, tokens_used)
```

- [ ] **Step 6: Run the unit suite and commit**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider`
Expected: all pass.

```bash
git add src/orchestration/domain/ports.py src/orchestration/application/assemble_context.py tests/unit/orchestration_fakes.py tests/unit/test_assemble_context.py
git commit  # feat: add meta-layer ports and budgeted context assembly
```

---

### Task 4: Latency-Adaptive Fallback Cascade

**Files:**
- Create: `src/orchestration/application/latency_cascade.py`
- Test: `tests/unit/test_latency_cascade.py`

**Interfaces:**
- Consumes: Task 1 entities, `CascadeTier` and `AccessFrequencyTracker` ports, `FakeCascadeTier` (Task 3), `InMemoryAccessFrequencyTracker` (existing).
- Produces:
  - `TierTimeouts(cag: float = 0.010, mag: float = 0.050, rag: float = 2.0)` with `for_paradigm(p) -> float`
  - `RagFindingsSink = Callable[[TierRequest, list[ContextItem]], Awaitable[None]]`
  - `LatencyCascade(tiers: Sequence[CascadeTier], timeouts: TierTimeouts | None = None, access_tracker: AccessFrequencyTracker | None = None, on_rag_findings: RagFindingsSink | None = None, clock: Callable[[], datetime] = _utc_now)`
  - `async LatencyCascade.run(request: TierRequest, decision: RoutingDecision | None = None) -> CascadeResult`
  - `async LatencyCascade.drain() -> None`

- [ ] **Step 1: Write the failing test `tests/unit/test_latency_cascade.py`**

```python
import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.domain.entities import (
    ContextItem,
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierOutcome,
    TierRequest,
    TierResult,
)
from src.orchestration.infrastructure.in_memory_access_tracker import (
    InMemoryAccessFrequencyTracker,
)
from tests.unit.orchestration_fakes import FakeCascadeTier

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG
_GENEROUS = TierTimeouts(cag=1.0, mag=1.0, rag=1.0)
_TIGHT = TierTimeouts(cag=0.05, mag=0.05, rag=0.05)
_SLOW = 0.5
_NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _request() -> TierRequest:
    return TierRequest(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        query="q",
        query_embedding=[1.0, 0.0],
    )


def _result(outcome: TierOutcome, paradigm: Paradigm, content: str = "x", source_id=None):
    return TierResult(outcome, [ContextItem(paradigm, content, 0.9, source_id)])


def _route(*paradigms: Paradigm, mode: RoutingMode = RoutingMode.CASCADE) -> RoutingDecision:
    return RoutingDecision(frozenset(paradigms), mode, {})


def _outcomes(result):
    return [(attempt.paradigm, attempt.outcome) for attempt in result.attempts]


async def test_a_cag_only_route_that_hits_returns_before_mag_or_rag_run():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG, "cached"))
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, mag, rag], _GENEROUS).run(_request(), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.HIT)]
    assert mag.requests == [] and rag.requests == []
    assert [item.content for item in result.items] == ["cached"]
    assert result.satisfied == frozenset({CAG})
    assert result.degraded is False


async def test_the_unrouted_cascade_stops_at_the_first_hit_in_cheapest_first_order():
    cag = FakeCascadeTier(CAG)
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, mag, rag], _GENEROUS).run(_request())
    assert _outcomes(result) == [(CAG, TierOutcome.MISS), (MAG, TierOutcome.HIT)]
    assert rag.requests == []


async def test_routing_to_rag_keeps_a_stale_cag_hit_out_of_the_answer():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG, "thirty days"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "forty-five days"))
    cascade = LatencyCascade([cag, rag], _GENEROUS)

    unrouted = await cascade.run(_request())
    assert [item.content for item in unrouted.items] == ["thirty days"]

    cag.requests.clear()
    routed = await cascade.run(_request(), _route(RAG))
    assert [item.content for item in routed.items] == ["forty-five days"]
    assert cag.requests == []


async def test_a_cag_miss_falls_through_to_rag_without_trying_an_unrouted_mag():
    cag = FakeCascadeTier(CAG)
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, mag, rag], _GENEROUS).run(_request(), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.MISS), (RAG, TierOutcome.HIT)]
    assert mag.requests == []


async def test_a_rag_only_route_never_touches_cag_or_mag():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG))
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, mag, rag], _GENEROUS).run(_request(), _route(RAG))
    assert _outcomes(result) == [(RAG, TierOutcome.HIT)]
    assert cag.requests == [] and mag.requests == []


async def test_a_partial_cag_match_is_kept_and_supplemented_by_rag():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.PARTIAL, CAG, "partial"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "full"))
    result = await LatencyCascade([cag, rag], _GENEROUS).run(_request(), _route(CAG))
    assert [item.content for item in result.items] == ["partial", "full"]
    assert result.satisfied == frozenset({RAG})


async def test_a_mag_and_rag_route_runs_both_even_when_mag_hits():
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([mag, rag], _GENEROUS).run(_request(), _route(MAG, RAG))
    assert _outcomes(result) == [(MAG, TierOutcome.HIT), (RAG, TierOutcome.HIT)]
    assert result.satisfied == frozenset({MAG, RAG})


async def test_a_routed_paradigm_with_no_configured_tier_falls_back_to_rag():
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([rag], _GENEROUS).run(_request(), _route(MAG))
    assert _outcomes(result) == [(RAG, TierOutcome.HIT)]


async def test_a_tier_that_exceeds_its_timeout_is_recorded_skipped_and_cancelled():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG), delay_seconds=_SLOW)
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, rag], _TIGHT).run(_request(), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.TIMEOUT), (RAG, TierOutcome.HIT)]
    assert result.degraded is True
    await asyncio.sleep(0.01)
    assert cag.cancelled is True


async def test_a_tier_that_raises_is_recorded_as_an_error_and_skipped():
    cag = FakeCascadeTier(CAG, error=RuntimeError("cache offline"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, rag], _GENEROUS).run(_request(), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.ERROR), (RAG, TierOutcome.HIT)]
    assert result.degraded is True


async def test_a_cancellation_raised_inside_a_tier_is_never_swallowed():
    cag = FakeCascadeTier(CAG, error=asyncio.CancelledError())
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    with pytest.raises(asyncio.CancelledError):
        await LatencyCascade([cag, rag], _GENEROUS).run(_request(), _route(CAG))


async def test_cancelling_the_caller_also_cancels_the_running_tier():
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG), delay_seconds=5.0)
    cascade = LatencyCascade([rag], TierTimeouts(cag=1.0, mag=1.0, rag=10.0))
    task = asyncio.create_task(cascade.run(_request(), _route(RAG)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.01)
    assert rag.cancelled is True


async def test_parallel_mode_runs_every_eligible_tier_and_merges_their_items():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG, "c"))
    mag = FakeCascadeTier(MAG, _result(TierOutcome.PARTIAL, MAG, "m"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "r"))
    decision = _route(CAG, MAG, mode=RoutingMode.PARALLEL)
    result = await LatencyCascade([cag, mag, rag], _GENEROUS).run(_request(), decision)
    assert [attempt.paradigm for attempt in result.attempts] == [CAG, MAG, RAG]
    assert sorted(item.content for item in result.items) == ["c", "m", "r"]
    assert result.satisfied == frozenset({CAG, RAG})


async def test_when_rag_times_out_earlier_items_come_back_as_a_degraded_best_effort():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.PARTIAL, CAG, "partial"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG), delay_seconds=_SLOW)
    result = await LatencyCascade([cag, rag], _TIGHT).run(_request(), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.PARTIAL), (RAG, TierOutcome.TIMEOUT)]
    assert [item.content for item in result.items] == ["partial"]
    assert result.degraded is True


async def test_a_timed_out_rag_attempt_finishes_in_the_background_into_the_findings_sink():
    document_id = uuid.uuid4()
    rag = FakeCascadeTier(
        RAG, _result(TierOutcome.HIT, RAG, "late", document_id), delay_seconds=0.2
    )
    received: list[list[ContextItem]] = []

    async def sink(request: TierRequest, items: list[ContextItem]) -> None:
        received.append(items)

    tracker = InMemoryAccessFrequencyTracker()
    request = _request()
    cascade = LatencyCascade(
        [rag], _TIGHT, access_tracker=tracker, on_rag_findings=sink, clock=lambda: _NOW
    )
    result = await cascade.run(request, _route(RAG))
    assert _outcomes(result) == [(RAG, TierOutcome.TIMEOUT)]
    assert received == []

    await cascade.drain()
    assert [[item.content for item in items] for items in received] == [["late"]]
    assert tracker.access_count(request.tenant_id, document_id, timedelta(hours=1), _NOW) == 1


async def test_without_a_findings_sink_a_timed_out_rag_attempt_is_cancelled():
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG), delay_seconds=_SLOW)
    await LatencyCascade([rag], _TIGHT).run(_request(), _route(RAG))
    await asyncio.sleep(0.01)
    assert rag.cancelled is True


async def test_a_rag_hit_records_one_access_per_returned_document():
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    rag = FakeCascadeTier(
        RAG,
        TierResult(
            TierOutcome.HIT,
            [
                ContextItem(RAG, "a1", 0.9, doc_a),
                ContextItem(RAG, "a2", 0.8, doc_a),
                ContextItem(RAG, "b", 0.7, doc_b),
            ],
        ),
    )
    tracker = InMemoryAccessFrequencyTracker()
    request = _request()
    cascade = LatencyCascade([rag], _GENEROUS, access_tracker=tracker, clock=lambda: _NOW)
    await cascade.run(request, _route(RAG))
    window = timedelta(hours=1)
    assert tracker.access_count(request.tenant_id, doc_a, window, _NOW) == 1
    assert tracker.access_count(request.tenant_id, doc_b, window, _NOW) == 1


def test_a_cascade_without_a_rag_tier_is_rejected():
    with pytest.raises(ValueError):
        LatencyCascade([FakeCascadeTier(CAG)])


def test_two_tiers_for_one_paradigm_are_rejected():
    with pytest.raises(ValueError):
        LatencyCascade([FakeCascadeTier(RAG), FakeCascadeTier(RAG)])


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_timeouts_are_rejected(bad):
    with pytest.raises(ValueError):
        TierTimeouts(cag=bad)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_latency_cascade.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/orchestration/application/latency_cascade.py`**

```python
import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    CascadeResult,
    ContextItem,
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierAttempt,
    TierOutcome,
    TierRequest,
    TierResult,
)
from src.orchestration.domain.ports import AccessFrequencyTracker, CascadeTier

logger = logging.getLogger(__name__)

RagFindingsSink = Callable[[TierRequest, list[ContextItem]], Awaitable[None]]


@dataclass(frozen=True)
class TierTimeouts:
    # Seconds. Defaults are Concept 5's own timeout guards.
    cag: float = 0.010
    mag: float = 0.050
    rag: float = 2.0

    def __post_init__(self) -> None:
        if min(self.cag, self.mag, self.rag) <= 0.0:
            raise ValueError("tier timeouts must be positive")

    def for_paradigm(self, paradigm: Paradigm) -> float:
        return {Paradigm.CAG: self.cag, Paradigm.MAG: self.mag, Paradigm.RAG: self.rag}[paradigm]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


class _Collector:
    def __init__(self) -> None:
        self.items: list[ContextItem] = []
        self.attempts: list[TierAttempt] = []
        self.satisfied: set[Paradigm] = set()
        self.degraded = False

    def add(self, attempt: TierAttempt, result: TierResult | None) -> None:
        self.attempts.append(attempt)
        if result is None:
            self.degraded = True
            return
        self.items.extend(result.items)
        if result.outcome is TierOutcome.HIT:
            self.satisfied.add(attempt.paradigm)

    def result(self) -> CascadeResult:
        return CascadeResult(
            list(self.items), list(self.attempts), frozenset(self.satisfied), self.degraded
        )


class LatencyCascade:
    """Concept 5's Latency-Adaptive Fallback Cascade, steered by a routing decision.

    - decision=None is the source's cascade as drawn: CAG, MAG, RAG, first
      HIT wins. It is the ablation baseline the router has to beat.
    - A CASCADE decision tries routed paradigms cheapest-first plus RAG as
      the universal last resort, stopping once every routed paradigm has a
      HIT. A paradigm the router left out is never attempted -- that is what
      keeps a freshness query away from a stale frozen cache.
    - A PARALLEL decision (router unsure) runs every eligible tier at once.

    A timeout or exception degrades the answer instead of failing it; a
    CancelledError always propagates. A RAG attempt that times out keeps
    running in the background when on_rag_findings is set ("if RAG is slow
    -> return best-effort + async update"). That task lives in this process
    only -- src/workers/ doesn't exist yet -- so drain() before shutdown.
    """

    def __init__(
        self,
        tiers: Sequence[CascadeTier],
        timeouts: TierTimeouts | None = None,
        access_tracker: AccessFrequencyTracker | None = None,
        on_rag_findings: RagFindingsSink | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        by_paradigm = {tier.paradigm: tier for tier in tiers}
        if len(by_paradigm) != len(tiers):
            raise ValueError("configure at most one tier per paradigm")
        if Paradigm.RAG not in by_paradigm:
            raise ValueError("a RAG tier is required: it is the cascade's last resort")
        self._tiers = by_paradigm
        self._timeouts = timeouts if timeouts is not None else TierTimeouts()
        self._access_tracker = access_tracker
        self._on_rag_findings = on_rag_findings
        self._clock = clock
        # Strong references: the event loop only keeps weak ones, so an
        # unreferenced background task can be garbage-collected mid-flight.
        self._background: set[asyncio.Task[None]] = set()

    async def run(
        self, request: TierRequest, decision: RoutingDecision | None = None
    ) -> CascadeResult:
        if decision is None:
            return await self._run_unrouted(request)
        eligible = [
            p
            for p in PARADIGM_ORDER
            if p in self._tiers and (p in decision.paradigms or p is Paradigm.RAG)
        ]
        if decision.mode is RoutingMode.PARALLEL:
            return await self._run_parallel(request, eligible)
        return await self._run_routed(request, decision.paradigms, eligible)

    async def drain(self) -> None:
        while self._background:
            await asyncio.gather(*self._background, return_exceptions=True)

    async def _run_unrouted(self, request: TierRequest) -> CascadeResult:
        collector = _Collector()
        for paradigm in PARADIGM_ORDER:
            if paradigm not in self._tiers:
                continue
            attempt, result = await self._attempt(paradigm, request)
            collector.add(attempt, result)
            if result is not None and result.outcome is TierOutcome.HIT:
                break
        return collector.result()

    async def _run_routed(
        self, request: TierRequest, routed: frozenset[Paradigm], eligible: list[Paradigm]
    ) -> CascadeResult:
        collector = _Collector()
        for paradigm in eligible:
            if routed <= collector.satisfied:
                break
            attempt, result = await self._attempt(paradigm, request)
            collector.add(attempt, result)
        return collector.result()

    async def _run_parallel(
        self, request: TierRequest, eligible: list[Paradigm]
    ) -> CascadeResult:
        collector = _Collector()
        outcomes = await asyncio.gather(*(self._attempt(p, request) for p in eligible))
        for attempt, result in outcomes:
            collector.add(attempt, result)
        return collector.result()

    async def _attempt(
        self, paradigm: Paradigm, request: TierRequest
    ) -> tuple[TierAttempt, TierResult | None]:
        started = time.perf_counter()
        task = asyncio.create_task(self._tiers[paradigm].attempt(request))
        try:
            # shield: a timeout must not cancel the task outright, because a
            # slow RAG attempt may still be worth finishing in the background.
            result = await asyncio.wait_for(
                asyncio.shield(task), self._timeouts.for_paradigm(paradigm)
            )
        except TimeoutError:
            if paradigm is Paradigm.RAG and self._on_rag_findings is not None:
                self._finish_in_background(task, request, self._on_rag_findings)
            else:
                task.cancel()
            return TierAttempt(paradigm, TierOutcome.TIMEOUT, _elapsed_ms(started)), None
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception:
            logger.warning(
                "cascade tier %s raised; continuing without it", paradigm.value, exc_info=True
            )
            return TierAttempt(paradigm, TierOutcome.ERROR, _elapsed_ms(started)), None
        if paradigm is Paradigm.RAG and result.outcome is TierOutcome.HIT:
            self._record_rag_access(request, result.items)
        return TierAttempt(paradigm, result.outcome, _elapsed_ms(started)), result

    def _finish_in_background(
        self, task: asyncio.Task[TierResult], request: TierRequest, sink: RagFindingsSink
    ) -> None:
        async def finish() -> None:
            try:
                result = await task
            except Exception:
                logger.warning("background RAG attempt failed after timing out", exc_info=True)
                return
            if result.outcome is not TierOutcome.HIT:
                return
            self._record_rag_access(request, result.items)
            try:
                await sink(request, result.items)
            except Exception:
                logger.warning("on_rag_findings raised for a background result", exc_info=True)

        background = asyncio.create_task(finish())
        self._background.add(background)
        background.add_done_callback(self._background.discard)

    def _record_rag_access(self, request: TierRequest, items: list[ContextItem]) -> None:
        # Pattern 1's "if result was frequent -> flag for CAG pre-loading":
        # feeds the tracker WarmCache/TieringPolicy already read. One access
        # per document, however many of its chunks came back.
        if self._access_tracker is None:
            return
        now = self._clock()
        source_ids = [item.source_id for item in items]
        document_ids: dict[uuid.UUID, None] = dict.fromkeys(
            source_id for source_id in source_ids if source_id is not None
        )
        for document_id in document_ids:
            self._access_tracker.record_access(request.tenant_id, document_id, now)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_latency_cascade.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/application/latency_cascade.py tests/unit/test_latency_cascade.py
git commit  # feat: add the Latency-Adaptive Fallback Cascade with routed, unrouted, and parallel modes
```

---

### Task 5: Cascade tiers over the existing CAG, MAG, and RAG use cases

**Files:**
- Modify: `src/orchestration/application/cache_warmed_retrieve.py`, `tests/unit/test_cache_warmed_retrieve.py`
- Create: `src/orchestration/application/cascade_tiers.py`
- Test: `tests/unit/test_cascade_tiers.py`

**Interfaces:**
- Consumes: `cosine_similarity` (Task 1), `CascadeTier` (Task 3), existing `FindSemanticFacts`, `Retriever`, `FakeSemanticMemoryRepository`, `FakeRetriever`, `FakeFrozenCache`, `FakeBagOfWordsEmbeddingModel`.
- Produces:
  - `CacheWarmedRetrieve.best_warmed_match(tenant_id: uuid.UUID, query_embedding: list[float]) -> tuple[SearchResult, float] | None`
  - `CagTier(cache_warmed_retrieve, *, hit_threshold: float, partial_threshold: float)`
  - `MagTier(find_semantic_facts, *, hit_threshold: float, partial_threshold: float, top_k: int = 5)`
  - `RagTier(retriever, *, top_k: int = 5)`

- [ ] **Step 1: Write the failing tests for the extraction**

Append to `tests/unit/test_cache_warmed_retrieve.py` (add `import pytest` and `from src.orchestration.domain.similarity import cosine_similarity`):

```python
class _CountingEmbedder(FakeBagOfWordsEmbeddingModel):
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return super().embed(text)


def _warmed() -> tuple[CacheWarmedRetrieve, FakeFrozenCache, uuid.UUID]:
    retriever, cache, _ = _build()
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED_CONTENT)
    retriever.note_warmed(_TENANT, document_id, _WARMED_CONTENT)
    return retriever, cache, document_id


def test_best_warmed_match_reports_the_closest_document_with_no_threshold_applied():
    retriever, _, document_id = _warmed()
    embedder = FakeBagOfWordsEmbeddingModel()
    query_embedding = embedder.embed(_UNRELATED_QUERY)

    match = retriever.best_warmed_match(_TENANT, query_embedding)

    assert match is not None
    result, score = match
    assert result.document_id == document_id
    assert result.content == _WARMED_CONTENT
    assert score == pytest.approx(
        cosine_similarity(query_embedding, embedder.embed(_WARMED_CONTENT))
    )


def test_best_warmed_match_is_scoped_to_the_tenant():
    retriever, _, _ = _warmed()
    embedding = FakeBagOfWordsEmbeddingModel().embed(_WARMED_CONTENT)
    assert retriever.best_warmed_match(uuid.uuid4(), embedding) is None


def test_best_warmed_match_ignores_a_document_evicted_from_the_frozen_cache():
    retriever, cache, document_id = _warmed()
    cache.evict(_TENANT, document_id)
    embedding = FakeBagOfWordsEmbeddingModel().embed(_WARMED_CONTENT)
    assert retriever.best_warmed_match(_TENANT, embedding) is None


async def test_execute_does_not_embed_the_query_when_nothing_is_warmed_for_the_tenant():
    embedder = _CountingEmbedder()
    retriever = CacheWarmedRetrieve(embedder, FakeFrozenCache(), FakeRetriever(), _THRESHOLD)
    await retriever.execute(_TENANT, _MATCHING_QUERY, top_k=5)
    assert embedder.calls == 0
```

- [ ] **Step 2: Run to verify the new tests fail and the old ones pass**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_cache_warmed_retrieve.py -q -p no:cacheprovider`
Expected: the three `best_warmed_match` tests fail with `AttributeError`; the others pass.

- [ ] **Step 3: Rewrite `src/orchestration/application/cache_warmed_retrieve.py`**

Keep the class docstring as it is, and append one paragraph to it:

```text
    best_warmed_match is public so the orchestration cascade's CagTier can
    reuse the same confirmed matching with its own hit/partial thresholds
    and a query embedding computed once upstream, instead of a second copy
    of this logic.
```

Replace the module's imports, the `_cosine_similarity` helper, `execute`, and `_best_warmed_match` with:

```python
import uuid

from src.orchestration.domain.ports import FrozenCache
from src.orchestration.domain.similarity import cosine_similarity
from src.orchestration.domain.sync_mixer import content_hash
from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import EmbeddingModel, Retriever

_Key = tuple[uuid.UUID, uuid.UUID]  # (tenant_id, document_id)
```

```python
    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        # Embedding the query only when this tenant has something warmed
        # keeps a cold tenant's queries from paying for a lookup that
        # cannot succeed.
        if self._has_warmed(tenant_id):
            match = self.best_warmed_match(tenant_id, self._embedder.embed(query))
            if match is not None and match[1] >= self._threshold:
                self._hits += 1
                return [match[0]]

        self._misses += 1
        return await self._fallback.execute(tenant_id, query, top_k)

    def best_warmed_match(
        self, tenant_id: uuid.UUID, query_embedding: list[float]
    ) -> tuple[SearchResult, float] | None:
        """The closest warmed document for this tenant and its similarity,
        confirmed against FrozenCache's real content_hash, with no threshold
        applied -- the caller decides what score counts."""
        best: tuple[uuid.UUID, float] | None = None
        for (tid, document_id), embedding in self._warmed_embeddings.items():
            if tid != tenant_id:
                continue
            score = cosine_similarity(query_embedding, embedding)
            if best is None or score > best[1]:
                best = (document_id, score)
        if best is None:
            return None

        document_id, score = best
        local_content = self._warmed_content[(tenant_id, document_id)]
        cached_hit = self._frozen_cache.lookup(tenant_id, document_id)
        if cached_hit is None or cached_hit.content_hash != content_hash(local_content):
            return None
        result = SearchResult(
            document_id=document_id, chunk_id=document_id, content=local_content, score=score
        )
        return result, score

    def _has_warmed(self, tenant_id: uuid.UUID) -> bool:
        return any(tid == tenant_id for tid, _ in self._warmed_embeddings)
```

`__init__`, `note_warmed`, and `stats` stay unchanged.

- [ ] **Step 4: Run the retrieve tests**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_cache_warmed_retrieve.py -q -p no:cacheprovider`
Expected: all pass, including every pre-existing test.

- [ ] **Step 5: Write the failing tier tests `tests/unit/test_cascade_tiers.py`**

```python
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
from src.mag.domain.entities import SemanticMemory
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.cascade_tiers import CagTier, MagTier, RagTier
from src.orchestration.domain.entities import Paradigm, TierOutcome, TierRequest
from src.orchestration.domain.similarity import cosine_similarity
from src.rag.domain.entities import SearchResult
from tests.unit.mag_fakes import FakeSemanticMemoryRepository
from tests.unit.orchestration_fakes import FakeBagOfWordsEmbeddingModel, FakeFrozenCache
from tests.unit.rag_fakes import FakeRetriever

_TENANT = uuid.uuid4()
_USER = uuid.uuid4()
_WARMED = "return policy allows customers to return unopened items within thirty days"
_PARTIAL_QUERY = "what is the return policy for unopened items"
_EMBEDDER = FakeBagOfWordsEmbeddingModel()


def _request(query="q", embedding=None, tenant_id=_TENANT, user_id=_USER) -> TierRequest:
    return TierRequest(
        tenant_id, user_id, uuid.uuid4(), query, embedding if embedding is not None else [1.0, 0.0]
    )


def _warmed_retriever():
    cache = FakeFrozenCache()
    retriever = CacheWarmedRetrieve(_EMBEDDER, cache, FakeRetriever(), similarity_threshold=0.99)
    document_id = uuid.uuid4()
    cache.preload(_TENANT, document_id, _WARMED)
    retriever.note_warmed(_TENANT, document_id, _WARMED)
    return retriever, cache, document_id


def _partial_score() -> float:
    score = cosine_similarity(_EMBEDDER.embed(_PARTIAL_QUERY), _EMBEDDER.embed(_WARMED))
    assert 0.1 < score < 0.95, "fake-embedder precondition"
    return score


async def test_cag_tier_reports_a_hit_at_or_above_the_hit_threshold():
    retriever, _, document_id = _warmed_retriever()
    tier = CagTier(retriever, hit_threshold=0.9, partial_threshold=0.5)
    result = await tier.attempt(_request(_WARMED, _EMBEDDER.embed(_WARMED)))
    assert result.outcome is TierOutcome.HIT
    [item] = result.items
    assert (item.paradigm, item.content, item.source_id) == (Paradigm.CAG, _WARMED, document_id)
    assert item.score == pytest.approx(1.0)


async def test_cag_tier_reports_a_partial_between_the_two_thresholds():
    retriever, _, _ = _warmed_retriever()
    score = _partial_score()
    tier = CagTier(retriever, hit_threshold=score + 0.01, partial_threshold=score - 0.01)
    result = await tier.attempt(_request(_PARTIAL_QUERY, _EMBEDDER.embed(_PARTIAL_QUERY)))
    assert result.outcome is TierOutcome.PARTIAL
    assert [item.content for item in result.items] == [_WARMED]


async def test_cag_tier_reports_a_miss_below_the_partial_threshold():
    retriever, _, _ = _warmed_retriever()
    score = _partial_score()
    tier = CagTier(retriever, hit_threshold=2.0, partial_threshold=score + 0.05)
    result = await tier.attempt(_request(_PARTIAL_QUERY, _EMBEDDER.embed(_PARTIAL_QUERY)))
    assert result.outcome is TierOutcome.MISS
    assert result.items == []


async def test_cag_tier_misses_for_another_tenant():
    retriever, _, _ = _warmed_retriever()
    tier = CagTier(retriever, hit_threshold=0.9, partial_threshold=0.5)
    request = _request(_WARMED, _EMBEDDER.embed(_WARMED), tenant_id=uuid.uuid4())
    assert (await tier.attempt(request)).outcome is TierOutcome.MISS


async def test_cag_tier_misses_once_the_document_is_evicted():
    retriever, cache, document_id = _warmed_retriever()
    cache.evict(_TENANT, document_id)
    tier = CagTier(retriever, hit_threshold=0.9, partial_threshold=0.5)
    assert (await tier.attempt(_request(_WARMED, _EMBEDDER.embed(_WARMED)))).outcome is (
        TierOutcome.MISS
    )


def _fact(key: str, embedding: list[float], valid_until: datetime | None = None):
    return SemanticMemory(
        id=uuid.uuid4(),
        user_id=_USER,
        fact_key=key,
        fact_value=f"{key} value",
        embedding=embedding,
        valid_until=valid_until,
    )


def _mag_tier(facts, top_k: int = 5) -> MagTier:
    repository = FakeSemanticMemoryRepository()
    repository.set_search_results(facts)
    return MagTier(
        FindSemanticFacts(repository), hit_threshold=0.9, partial_threshold=0.5, top_k=top_k
    )


async def test_mag_tier_hits_when_the_best_fact_clears_the_hit_threshold_and_drops_weak_facts():
    strong, weak = _fact("strong", [1.0, 0.0]), _fact("weak", [0.0, 1.0])
    result = await _mag_tier([strong, weak]).attempt(_request(embedding=[1.0, 0.0]))
    assert result.outcome is TierOutcome.HIT
    assert [(i.paradigm, i.content, i.source_id) for i in result.items] == [
        (Paradigm.MAG, "strong: strong value", strong.id)
    ]


async def test_mag_tier_reports_partial_when_relevant_facts_fall_short_of_a_hit():
    a, b = _fact("a", [1.0, 0.0]), _fact("b", [0.0, 1.0])
    result = await _mag_tier([a, b]).attempt(_request(embedding=[0.8, 0.6]))
    assert result.outcome is TierOutcome.PARTIAL
    assert [item.content for item in result.items] == ["a: a value", "b: b value"]


async def test_mag_tier_misses_when_no_fact_clears_the_partial_threshold():
    result = await _mag_tier([_fact("a", [1.0, 0.0])]).attempt(_request(embedding=[-1.0, 0.0]))
    assert result.outcome is TierOutcome.MISS


async def test_mag_tier_never_sees_an_invalidated_fact():
    stale = _fact("stale", [1.0, 0.0], valid_until=datetime.now(UTC) - timedelta(days=1))
    result = await _mag_tier([stale]).attempt(_request(embedding=[1.0, 0.0]))
    assert result.outcome is TierOutcome.MISS


async def test_mag_tier_scopes_its_search_to_the_requesting_user_and_tenant():
    class _SpyRepository(FakeSemanticMemoryRepository):
        def __init__(self) -> None:
            super().__init__()
            self.search_calls: list[tuple[uuid.UUID, uuid.UUID, int]] = []

        async def search_by_similarity(self, query_embedding, user_id, tenant_id, top_k):
            self.search_calls.append((user_id, tenant_id, top_k))
            return await super().search_by_similarity(query_embedding, user_id, tenant_id, top_k)

    spy = _SpyRepository()
    tier = MagTier(FindSemanticFacts(spy), hit_threshold=0.9, partial_threshold=0.5, top_k=3)
    other_tenant, other_user = uuid.uuid4(), uuid.uuid4()
    await tier.attempt(_request(tenant_id=other_tenant, user_id=other_user))
    assert spy.search_calls == [(other_user, other_tenant, 3)]


async def test_rag_tier_hits_with_every_retrieved_result():
    document_id = uuid.uuid4()
    results = [
        SearchResult(document_id=document_id, chunk_id=uuid.uuid4(), content="fresh", score=0.7)
    ]
    retriever = FakeRetriever(results)
    result = await RagTier(retriever, top_k=4).attempt(_request("what changed today"))
    assert result.outcome is TierOutcome.HIT
    assert [(i.paradigm, i.content, i.score, i.source_id) for i in result.items] == [
        (Paradigm.RAG, "fresh", 0.7, document_id)
    ]
    assert retriever.calls == [(_TENANT, "what changed today", 4)]


async def test_rag_tier_misses_when_nothing_is_retrieved():
    assert (await RagTier(FakeRetriever()).attempt(_request())).outcome is TierOutcome.MISS


def test_a_partial_threshold_above_the_hit_threshold_is_rejected():
    retriever, _, _ = _warmed_retriever()
    with pytest.raises(ValueError):
        CagTier(retriever, hit_threshold=0.5, partial_threshold=0.6)
    with pytest.raises(ValueError):
        MagTier(
            FindSemanticFacts(FakeSemanticMemoryRepository()),
            hit_threshold=0.5,
            partial_threshold=0.6,
        )


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_top_k_is_rejected(bad):
    with pytest.raises(ValueError):
        RagTier(FakeRetriever(), top_k=bad)
    with pytest.raises(ValueError):
        MagTier(
            FindSemanticFacts(FakeSemanticMemoryRepository()),
            hit_threshold=0.9,
            partial_threshold=0.5,
            top_k=bad,
        )
```

- [ ] **Step 6: Run to verify they fail**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_cascade_tiers.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'src.orchestration.application.cascade_tiers'`.

- [ ] **Step 7: Implement `src/orchestration/application/cascade_tiers.py`**

```python
import asyncio

from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.domain.entities import (
    ContextItem,
    Paradigm,
    TierOutcome,
    TierRequest,
    TierResult,
)
from src.orchestration.domain.ports import CascadeTier
from src.rag.domain.ports import Retriever


def _check_thresholds(hit_threshold: float, partial_threshold: float) -> None:
    if partial_threshold > hit_threshold:
        raise ValueError("partial_threshold cannot exceed hit_threshold")


def _check_top_k(top_k: int) -> None:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")


class CagTier(CascadeTier):
    """The cascade's first tier: a confirmed match against CAG's warmed set."""

    def __init__(
        self,
        cache_warmed_retrieve: CacheWarmedRetrieve,
        *,
        hit_threshold: float,
        partial_threshold: float,
    ) -> None:
        _check_thresholds(hit_threshold, partial_threshold)
        self._retrieve = cache_warmed_retrieve
        self._hit = hit_threshold
        self._partial = partial_threshold

    @property
    def paradigm(self) -> Paradigm:
        return Paradigm.CAG

    async def attempt(self, request: TierRequest) -> TierResult:
        # Matching is CPU work (a cosine pass over the warmed set plus a
        # FrozenCache lookup). Inline, it would hold the event loop and the
        # cascade's 10ms timeout could never fire.
        match = await asyncio.to_thread(
            self._retrieve.best_warmed_match, request.tenant_id, request.query_embedding
        )
        if match is None:
            return TierResult(TierOutcome.MISS)
        result, score = match
        if score < self._partial:
            return TierResult(TierOutcome.MISS)
        item = ContextItem(Paradigm.CAG, result.content, score, result.document_id)
        outcome = TierOutcome.HIT if score >= self._hit else TierOutcome.PARTIAL
        return TierResult(outcome, [item])


class MagTier(CascadeTier):
    """The cascade's second tier: this user's semantic facts.

    Invalidated and archived facts never arrive here --
    PostgresSemanticMemoryRepository.search_by_similarity filters them --
    which is Concept 5's "if MAG has stale state -> invalidate".
    """

    def __init__(
        self,
        find_semantic_facts: FindSemanticFacts,
        *,
        hit_threshold: float,
        partial_threshold: float,
        top_k: int = 5,
    ) -> None:
        _check_thresholds(hit_threshold, partial_threshold)
        _check_top_k(top_k)
        self._find = find_semantic_facts
        self._hit = hit_threshold
        self._partial = partial_threshold
        self._top_k = top_k

    @property
    def paradigm(self) -> Paradigm:
        return Paradigm.MAG

    async def attempt(self, request: TierRequest) -> TierResult:
        facts = await self._find.by_similarity(
            request.query_embedding, request.user_id, request.tenant_id, self._top_k
        )
        relevant = [scored for scored in facts if scored.score >= self._partial]
        if not relevant:
            return TierResult(TierOutcome.MISS)
        items = [
            ContextItem(
                Paradigm.MAG,
                f"{scored.fact.fact_key}: {scored.fact.fact_value}",
                scored.score,
                scored.fact.id,
            )
            for scored in relevant
        ]
        best = max(scored.score for scored in relevant)
        return TierResult(TierOutcome.HIT if best >= self._hit else TierOutcome.PARTIAL, items)


class RagTier(CascadeTier):
    """The cascade's last resort: any RAG Retriever. RAG has no partial hit."""

    def __init__(self, retriever: Retriever, *, top_k: int = 5) -> None:
        _check_top_k(top_k)
        self._retriever = retriever
        self._top_k = top_k

    @property
    def paradigm(self) -> Paradigm:
        return Paradigm.RAG

    async def attempt(self, request: TierRequest) -> TierResult:
        results = await self._retriever.execute(request.tenant_id, request.query, self._top_k)
        if not results:
            return TierResult(TierOutcome.MISS)
        return TierResult(
            TierOutcome.HIT,
            [ContextItem(Paradigm.RAG, r.content, r.score, r.document_id) for r in results],
        )
```

- [ ] **Step 8: Run the unit suite and commit**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider`
Expected: all pass.

```bash
git add src/orchestration/application/cache_warmed_retrieve.py src/orchestration/application/cascade_tiers.py tests/unit/test_cache_warmed_retrieve.py tests/unit/test_cascade_tiers.py
git commit  # feat: add CAG, MAG, and RAG cascade tiers over the existing use cases
```

---

### Task 6: The three query classifiers

**Files:**
- Create: `src/orchestration/infrastructure/lexical_query_classifier.py`, `src/orchestration/infrastructure/routing_exemplars.py`, `src/orchestration/infrastructure/prototype_query_classifier.py`, `src/orchestration/infrastructure/llm_query_classifier.py`
- Test: `tests/unit/test_lexical_query_classifier.py`, `tests/unit/test_prototype_query_classifier.py`, `tests/unit/test_llm_query_classifier.py`

**Interfaces:**
- Consumes: `QueryClassifier` (Task 3), `cosine_similarity` (Task 1), `EmbeddingModel` and `ChatModel` ports, `FakeChatModel`.
- Produces:
  - `LexicalQueryClassifier(cues: dict[Paradigm, tuple[str, ...]] | None = None)`, `DEFAULT_CUES`
  - `RoutingExemplar(query: str, paradigms: frozenset[Paradigm])`, `DEFAULT_ROUTING_EXEMPLARS: tuple[RoutingExemplar, ...]`
  - `PrototypeQueryClassifier(embedding_model, exemplars: Sequence[RoutingExemplar] = DEFAULT_ROUTING_EXEMPLARS, k: int = 5)`
  - `LlmQueryClassifier(chat_model)` with `parse_failures: int`

- [ ] **Step 1: Write the failing lexical test**

```python
import pytest

from src.orchestration.domain.entities import Paradigm
from src.orchestration.infrastructure.lexical_query_classifier import LexicalQueryClassifier

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


async def test_a_query_with_no_cues_scores_zero_everywhere():
    scores = await LexicalQueryClassifier().score("what is two plus two", [])
    assert scores == {CAG: 0.0, MAG: 0.0, RAG: 0.0}


async def test_each_matching_cue_raises_its_paradigm_score():
    scores = await LexicalQueryClassifier().score("What changed in the policy today?", [])
    assert scores[RAG] == pytest.approx(0.91)  # "changed", "today"
    assert scores[CAG] == pytest.approx(0.7)  # "policy"
    assert scores[MAG] == 0.0


async def test_cues_match_whole_words_case_insensitively():
    classifier = LexicalQueryClassifier()
    assert (await classifier.score("the NEWSPAPER archive", []))[RAG] == 0.0
    assert (await classifier.score("any NEWS about it", []))[RAG] == pytest.approx(0.7)


async def test_custom_cues_replace_the_defaults():
    classifier = LexicalQueryClassifier({CAG: ("zebra",), MAG: (), RAG: ()})
    scores = await classifier.score("zebra today", [])
    assert scores == {CAG: pytest.approx(0.7), MAG: 0.0, RAG: 0.0}
```

- [ ] **Step 2: Implement `lexical_query_classifier.py`**

```python
import re

from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm
from src.orchestration.domain.ports import QueryClassifier

# Cue phrases drawn from how Concept 1 describes each routing dimension
# (unified_rag_cag_mag_architecture.md): data freshness and live data point
# to RAG, prior-conversation dependency points to MAG, static and
# pre-loaded reference knowledge points to CAG.
DEFAULT_CUES: dict[Paradigm, tuple[str, ...]] = {
    Paradigm.CAG: (
        "policy", "policies", "manual", "guide", "handbook", "documentation",
        "procedure", "how do i", "how to", "explain", "reference", "code file", "faq",
    ),
    Paradigm.MAG: (
        "remember", "remind me", "we discussed", "we talked", "left off", "continue",
        "earlier", "last time", "previously", "i told you", "i asked", "i mentioned",
        "my preference", "you said", "our conversation",
    ),
    Paradigm.RAG: (
        "today", "latest", "current", "currently", "right now", "this week",
        "this month", "changed", "recent", "recently", "new", "news", "update",
        "updated", "live", "price", "prices", "stock", "compare",
    ),
}

# One cue scores 0.7, two 0.91, three 0.973: each further cue closes 70% of
# the remaining distance to certainty.
_CUE_WEIGHT = 0.7


class LexicalQueryClassifier(QueryClassifier):
    """The cheapest classifier and the floor the other two must beat."""

    def __init__(self, cues: dict[Paradigm, tuple[str, ...]] | None = None) -> None:
        source = cues if cues is not None else DEFAULT_CUES
        self._patterns = {
            paradigm: [
                re.compile(rf"\b{re.escape(cue)}\b", re.IGNORECASE)
                for cue in source.get(paradigm, ())
            ]
            for paradigm in PARADIGM_ORDER
        }

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        scores: dict[Paradigm, float] = {}
        for paradigm, patterns in self._patterns.items():
            matches = sum(1 for pattern in patterns if pattern.search(query))
            scores[paradigm] = 1.0 - (1.0 - _CUE_WEIGHT) ** matches
        return scores
```

- [ ] **Step 3: Write the failing prototype test**

```python
import pytest

from src.orchestration.domain.entities import Paradigm
from src.orchestration.infrastructure.prototype_query_classifier import (
    PrototypeQueryClassifier,
)
from src.orchestration.infrastructure.routing_exemplars import (
    DEFAULT_ROUTING_EXEMPLARS,
    RoutingExemplar,
)
from src.rag.domain.ports import EmbeddingModel

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG
_CONCEPT_ONE_QUERIES = (
    "What's our refund policy?",
    "What changed in the policy today?",
    "Continue where we left off yesterday",
    "Compare today's sales with last month",
    "Explain this code file",
    "What did I ask you to remember?",
)


class _TableEmbedder(EmbeddingModel):
    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return self._table[text]


_TABLE = {
    "cag example": [1.0, 0.0, 0.0],
    "rag example": [0.0, 1.0, 0.0],
    "mag example": [0.0, 0.0, 1.0],
    "rag mag example": [0.0, 0.7071, 0.7071],
}
_EXEMPLARS = (
    RoutingExemplar("cag example", frozenset({CAG})),
    RoutingExemplar("rag example", frozenset({RAG})),
    RoutingExemplar("mag example", frozenset({MAG})),
    RoutingExemplar("rag mag example", frozenset({RAG, MAG})),
)


async def test_the_single_nearest_exemplar_decides_when_k_is_one():
    classifier = PrototypeQueryClassifier(_TableEmbedder(_TABLE), _EXEMPLARS, k=1)
    assert await classifier.score("q", [1.0, 0.0, 0.0]) == {CAG: 1.0, MAG: 0.0, RAG: 0.0}


async def test_neighbors_vote_by_similarity_and_multi_label_exemplars_vote_for_each_label():
    classifier = PrototypeQueryClassifier(_TableEmbedder(_TABLE), _EXEMPLARS, k=2)
    scores = await classifier.score("q", [0.0, 1.0, 0.0])
    assert scores[RAG] == pytest.approx(1.0)
    assert scores[MAG] == pytest.approx(0.7071 / 1.7071, abs=1e-4)
    assert scores[CAG] == 0.0


async def test_neighbors_with_no_positive_similarity_give_all_zero_scores():
    classifier = PrototypeQueryClassifier(_TableEmbedder(_TABLE), _EXEMPLARS, k=1)
    assert await classifier.score("q", [-1.0, 0.0, 0.0]) == {CAG: 0.0, MAG: 0.0, RAG: 0.0}


async def test_exemplars_are_embedded_once_at_construction_not_per_query():
    embedder = _TableEmbedder(_TABLE)
    classifier = PrototypeQueryClassifier(embedder, _EXEMPLARS, k=2)
    await classifier.score("q", [1.0, 0.0, 0.0])
    await classifier.score("q", [0.0, 1.0, 0.0])
    assert embedder.calls == len(_EXEMPLARS)


def test_an_empty_exemplar_set_or_a_non_positive_k_is_rejected():
    with pytest.raises(ValueError):
        PrototypeQueryClassifier(_TableEmbedder(_TABLE), ())
    with pytest.raises(ValueError):
        PrototypeQueryClassifier(_TableEmbedder(_TABLE), _EXEMPLARS, k=0)


def test_the_default_exemplars_are_labeled_and_never_contain_a_concept_one_query():
    assert all(exemplar.paradigms for exemplar in DEFAULT_ROUTING_EXEMPLARS)
    queries = {exemplar.query.casefold() for exemplar in DEFAULT_ROUTING_EXEMPLARS}
    assert queries.isdisjoint(query.casefold() for query in _CONCEPT_ONE_QUERIES)
```

- [ ] **Step 4: Implement `routing_exemplars.py` and `prototype_query_classifier.py`**

`src/orchestration/infrastructure/routing_exemplars.py`:

```python
from dataclasses import dataclass

from src.orchestration.domain.entities import Paradigm


@dataclass(frozen=True)
class RoutingExemplar:
    query: str
    paradigms: frozenset[Paradigm]


_CAG = frozenset({Paradigm.CAG})
_MAG = frozenset({Paradigm.MAG})
_RAG = frozenset({Paradigm.RAG})
_RAG_MAG = frozenset({Paradigm.RAG, Paradigm.MAG})
_CAG_RAG = frozenset({Paradigm.CAG, Paradigm.RAG})

# Labeled by the same reading of Concept 1 and Concept 9 as the lexical
# cues. Kept disjoint from Concept 1's own six example queries, which the
# evaluation set uses verbatim; a unit test enforces that.
DEFAULT_ROUTING_EXEMPLARS: tuple[RoutingExemplar, ...] = (
    RoutingExemplar("What does the employee handbook say about vacation days?", _CAG),
    RoutingExemplar("How do I reset my password according to the help guide?", _CAG),
    RoutingExemplar("What are the warranty terms in the product manual?", _CAG),
    RoutingExemplar("Walk me through the onboarding procedure for new contractors.", _CAG),
    RoutingExemplar("What is the standard shipping policy for international orders?", _CAG),
    RoutingExemplar("Summarize the architecture section of our internal documentation.", _CAG),
    RoutingExemplar("What does this function in the billing module do?", _CAG),
    RoutingExemplar("What are the size guide measurements for a medium shirt?", _CAG),
    RoutingExemplar("What is the company's stock price right now?", _RAG),
    RoutingExemplar("Is there any news about our competitor from this morning?", _RAG),
    RoutingExemplar("What is the current inventory level for the blue running shoes?", _RAG),
    RoutingExemplar("Which orders were delayed by the storm in the last hour?", _RAG),
    RoutingExemplar("What did the latest pricing update change for enterprise plans?", _RAG),
    RoutingExemplar("Is the payment service having an outage at the moment?", _RAG),
    RoutingExemplar("What are today's exchange rates from euros to dollars?", _RAG),
    RoutingExemplar("Which commits were merged into the release branch in the past hour?", _RAG),
    RoutingExemplar("What name did I tell you to call me?", _MAG),
    RoutingExemplar("Pick up the budget discussion from our last session.", _MAG),
    RoutingExemplar("Which of the options I shortlisted did I like best?", _MAG),
    RoutingExemplar("Remind me what dietary restrictions I mentioned.", _MAG),
    RoutingExemplar("What was the last thing we were working on together?", _MAG),
    RoutingExemplar("Write it in the tone I asked you to use before.", _MAG),
    RoutingExemplar("Did I already give you my project's deadline?", _MAG),
    RoutingExemplar("Go back to the draft email you helped me with.", _MAG),
    RoutingExemplar(
        "Given the portfolio I described to you, how did those stocks move today?", _RAG_MAG
    ),
    RoutingExemplar(
        "Are there flights leaving today for the destination I told you about?", _RAG_MAG
    ),
    RoutingExemplar(
        "Based on my usual order, is anything I buy out of stock right now?", _RAG_MAG
    ),
    RoutingExemplar(
        "How does this week's weather forecast affect the hiking trip we planned?", _RAG_MAG
    ),
    RoutingExemplar("Check my saved shopping list against current grocery prices.", _RAG_MAG),
    RoutingExemplar(
        "Has the bug I reported to you been fixed in the latest release notes?", _RAG_MAG
    ),
    RoutingExemplar(
        "What does the catalog say about this laptop, and is it in stock today?", _CAG_RAG
    ),
    RoutingExemplar(
        "What is the listed price of the premium plan, and has it changed recently?", _CAG_RAG
    ),
    RoutingExemplar(
        "How does the documented orders API differ from what was deployed this week?", _CAG_RAG
    ),
    RoutingExemplar(
        "What does the size guide recommend, and are those sizes available right now?", _CAG_RAG
    ),
)
```

`src/orchestration/infrastructure/prototype_query_classifier.py`:

```python
from collections.abc import Sequence

from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm
from src.orchestration.domain.ports import QueryClassifier
from src.orchestration.domain.similarity import cosine_similarity
from src.orchestration.infrastructure.routing_exemplars import (
    DEFAULT_ROUTING_EXEMPLARS,
    RoutingExemplar,
)
from src.rag.domain.ports import EmbeddingModel


class PrototypeQueryClassifier(QueryClassifier):
    """Similarity-weighted vote over the k nearest labeled exemplars.

    Each paradigm's score is the share of neighbor similarity carried by
    exemplars labeled with it. An exemplar with two labels votes for both,
    which is how a RAG + MAG route can come out with both scores high.
    Exemplars are embedded once, here; queries arrive already embedded.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        exemplars: Sequence[RoutingExemplar] = DEFAULT_ROUTING_EXEMPLARS,
        k: int = 5,
    ) -> None:
        if not exemplars:
            raise ValueError("at least one exemplar is required")
        if k < 1:
            raise ValueError("k must be at least 1")
        self._k = k
        self._exemplars = [
            (exemplar, embedding_model.embed(exemplar.query)) for exemplar in exemplars
        ]

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        nearest = sorted(
            (
                (cosine_similarity(query_embedding, embedding), exemplar)
                for exemplar, embedding in self._exemplars
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )[: self._k]
        weighted = [(max(0.0, similarity), exemplar) for similarity, exemplar in nearest]
        total = sum(weight for weight, _ in weighted)
        if total == 0.0:
            return dict.fromkeys(PARADIGM_ORDER, 0.0)
        return {
            paradigm: sum(weight for weight, ex in weighted if paradigm in ex.paradigms) / total
            for paradigm in PARADIGM_ORDER
        }
```

- [ ] **Step 5: Write the failing LLM classifier test**

```python
import pytest

from src.orchestration.domain.entities import Paradigm
from src.orchestration.infrastructure.llm_query_classifier import LlmQueryClassifier
from tests.unit.rag_fakes import FakeChatModel

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG
_FALLBACK = {CAG: 0.5, MAG: 0.5, RAG: 0.5}


async def test_a_clean_json_response_becomes_the_scores():
    chat_model = FakeChatModel('{"cag": 0.1, "mag": 0.0, "rag": 0.9}')
    classifier = LlmQueryClassifier(chat_model)
    assert await classifier.score("What changed today?", []) == {CAG: 0.1, MAG: 0.0, RAG: 0.9}
    assert "What changed today?" in chat_model.last_prompt
    assert classifier.parse_failures == 0


async def test_prose_around_the_json_object_is_tolerated():
    reply = 'Sure. {"cag": 1, "mag": 0.25, "rag": 0} Hope that helps.'
    scores = await LlmQueryClassifier(FakeChatModel(reply)).score("q", [])
    assert scores == {CAG: 1.0, MAG: 0.25, RAG: 0.0}


@pytest.mark.parametrize(
    "reply",
    [
        "no json here",
        '{"cag": 0.1, "mag": 0.2}',
        '{"cag": 1.5, "mag": 0.0, "rag": 0.0}',
        '{"cag": true, "mag": 0.0, "rag": 0.0}',
        '{"cag": "high", "mag": 0.0, "rag": 0.0}',
        "[0.1, 0.2, 0.3]",
        '{"cag": 0.1, "mag": 0.2, "rag": 0.3',
    ],
)
async def test_an_unusable_reply_routes_everything_into_the_uncertainty_band(reply):
    classifier = LlmQueryClassifier(FakeChatModel(reply))
    assert await classifier.score("q", []) == _FALLBACK
    assert classifier.parse_failures == 1


async def test_a_query_containing_braces_does_not_break_the_prompt():
    chat_model = FakeChatModel('{"cag": 0.0, "mag": 0.0, "rag": 1.0}')
    await LlmQueryClassifier(chat_model).score("parse {this} json", [])
    assert "parse {this} json" in chat_model.last_prompt
```

- [ ] **Step 6: Implement `llm_query_classifier.py`**

```python
import json
import math

from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm
from src.orchestration.domain.ports import QueryClassifier
from src.rag.domain.ports import ChatModel

_PROMPT_TEMPLATE = (
    "You route a user's question to the knowledge sources able to answer it. "
    "For each source, give your confidence from 0 to 1 that answering the "
    "question needs it. A question can need more than one source.\n"
    "- cag: stable reference knowledge that rarely changes and is asked about "
    "often, such as policies, manuals, guides, or a codebase.\n"
    "- mag: this user's own conversation history, preferences, or things they "
    "asked to have remembered.\n"
    "- rag: fresh or changing external information, such as today's data, "
    "recent changes, live figures, or news.\n"
    'Respond with ONLY a JSON object of the form {{"cag": 0.0, "mag": 0.0, '
    '"rag": 0.0}} and nothing else.\n\n'
    "Question: {query}"
)
# Exactly the default decision threshold, so every paradigm lands inside the
# uncertainty band and the route runs PARALLEL across all tiers: an
# unreadable classification becomes the safest route, not a guess.
_UNPARSEABLE_SCORE = 0.5


def _parse_scores(raw: str) -> dict[Paradigm, float] | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    scores: dict[Paradigm, float] = {}
    for paradigm in PARADIGM_ORDER:
        value = data.get(paradigm.value)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        scores[paradigm] = float(value)
    return scores


class LlmQueryClassifier(QueryClassifier):
    def __init__(self, chat_model: ChatModel) -> None:
        self._chat_model = chat_model
        # Counted, not hidden: the router report states how often the real
        # model's reply was unusable, the same failure #149 found in the judge.
        self.parse_failures = 0

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        raw = await self._chat_model.complete(_PROMPT_TEMPLATE.format(query=query))
        scores = _parse_scores(raw)
        if scores is None:
            self.parse_failures += 1
            return dict.fromkeys(PARADIGM_ORDER, _UNPARSEABLE_SCORE)
        return scores
```

- [ ] **Step 7: Run all three test files, then the unit suite, and commit**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_lexical_query_classifier.py tests/unit/test_prototype_query_classifier.py tests/unit/test_llm_query_classifier.py -q -p no:cacheprovider`
Expected: all pass. Then run the full unit suite; expected: all pass.

```bash
git add src/orchestration/infrastructure/lexical_query_classifier.py src/orchestration/infrastructure/routing_exemplars.py src/orchestration/infrastructure/prototype_query_classifier.py src/orchestration/infrastructure/llm_query_classifier.py tests/unit/test_lexical_query_classifier.py tests/unit/test_prototype_query_classifier.py tests/unit/test_llm_query_classifier.py
git commit  # feat: add lexical, prototype, and LLM query classifiers for the Paradigm Router
```

---

### Task 7: Session budget recorder

**Files:**
- Create: `src/orchestration/infrastructure/postgres_session_budget_recorder.py`
- Test: `tests/unit/test_session_budget_record.py`, `tests/integration/test_postgres_session_budget_recorder.py`

**Interfaces:**
- Consumes: `SessionBudgetRecorder` (Task 3), `allocate` (Task 2), `SessionNotFound` (Task 1), existing `set_tenant_context` and the `db_session` integration fixture.
- Produces: `budget_record(allocation, contributing, recorded_at) -> dict[str, object]`; `PostgresSessionBudgetRecorder(session: AsyncSession, clock: Callable[[], datetime] = _utc_now)`.

- [ ] **Step 1: Write the failing unit test for the record shape**

```python
from datetime import UTC, datetime

from src.orchestration.domain.entities import BudgetAllocation, Paradigm
from src.orchestration.infrastructure.postgres_session_budget_recorder import budget_record


def test_the_record_carries_every_slice_the_contributing_set_and_a_timestamp():
    allocation = BudgetAllocation(cag=5, mag=4, rag=3, query=2, reserve=1)
    record = budget_record(
        allocation,
        frozenset({Paradigm.RAG, Paradigm.CAG}),
        datetime(2026, 9, 13, 12, 0, tzinfo=UTC),
    )
    assert record == {
        "total": 15,
        "slices": {"cag": 5, "mag": 4, "rag": 3, "query": 2, "reserve": 1},
        "contributing": ["cag", "rag"],
        "recorded_at": "2026-09-13T12:00:00+00:00",
    }
```

- [ ] **Step 2: Write the failing integration test**

```python
import json
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import Paradigm
from src.orchestration.domain.errors import SessionNotFound
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"
_CONTRIBUTING = frozenset({Paradigm.CAG, Paradigm.RAG})


async def _create_user_and_session(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    await set_tenant_context(db_session, tenant_id)
    now = datetime.now(UTC)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id, "created_at": now, "updated_at": now,
        },
    )
    session_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "t"},
    )
    await db_session.commit()
    return session_id


async def _stored_budget(db_session, tenant_id: uuid.UUID, session_id: uuid.UUID):
    await set_tenant_context(db_session, tenant_id)
    value = (
        await db_session.execute(
            text("SELECT context_budget FROM sessions WHERE id = :id"), {"id": session_id}
        )
    ).scalar_one()
    return json.loads(value) if isinstance(value, str) else value


async def test_record_writes_the_allocation_into_the_real_sessions_row(db_session):
    tenant_id = uuid.uuid4()
    session_id = await _create_user_and_session(db_session, tenant_id)
    recorder = PostgresSessionBudgetRecorder(
        db_session, clock=lambda: datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    )

    await recorder.record(tenant_id, session_id, allocate(128_000, _CONTRIBUTING), _CONTRIBUTING)

    assert await _stored_budget(db_session, tenant_id, session_id) == {
        "total": 128_000,
        "slices": {"cag": 72_533, "mag": 0, "rag": 36_266, "query": 12_800, "reserve": 6_401},
        "contributing": ["cag", "rag"],
        "recorded_at": "2026-09-13T12:00:00+00:00",
    }


async def test_recording_under_another_tenant_is_refused_by_rls_and_leaves_the_row_untouched(
    db_session,
):
    owner = uuid.uuid4()
    session_id = await _create_user_and_session(db_session, owner)
    recorder = PostgresSessionBudgetRecorder(db_session)

    with pytest.raises(SessionNotFound):
        await recorder.record(
            uuid.uuid4(), session_id, allocate(128_000, _CONTRIBUTING), _CONTRIBUTING
        )

    assert await _stored_budget(db_session, owner, session_id) is None


async def test_recording_for_a_session_that_does_not_exist_raises(db_session):
    recorder = PostgresSessionBudgetRecorder(db_session)
    with pytest.raises(SessionNotFound):
        await recorder.record(
            uuid.uuid4(), uuid.uuid4(), allocate(128_000, _CONTRIBUTING), _CONTRIBUTING
        )
```

- [ ] **Step 3: Run both to verify they fail**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_session_budget_record.py tests/integration/test_postgres_session_budget_recorder.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError` (Docker must be running for the integration file).

- [ ] **Step 4: Implement `postgres_session_budget_recorder.py`**

```python
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src.identity.infrastructure.db import set_tenant_context
from src.orchestration.domain.entities import PARADIGM_ORDER, BudgetAllocation, Paradigm
from src.orchestration.domain.errors import SessionNotFound
from src.orchestration.domain.ports import SessionBudgetRecorder


def _utc_now() -> datetime:
    return datetime.now(UTC)


def budget_record(
    allocation: BudgetAllocation, contributing: frozenset[Paradigm], recorded_at: datetime
) -> dict[str, object]:
    return {
        "total": allocation.total,
        "slices": {
            "cag": allocation.cag,
            "mag": allocation.mag,
            "rag": allocation.rag,
            "query": allocation.query,
            "reserve": allocation.reserve,
        },
        "contributing": [p.value for p in PARADIGM_ORDER if p in contributing],
        "recorded_at": recorded_at.isoformat(),
    }


class PostgresSessionBudgetRecorder(SessionBudgetRecorder):
    """Writes the latest turn's allocation into sessions.context_budget.

    The column has existed since migration 0001 and DATABASE.md already
    describes it; this is its first writer. set_tenant_context is called
    here rather than trusted to the caller, so the tenant_isolation RLS
    policy always has a tenant to enforce -- and an UPDATE that RLS hides
    (another tenant's session) looks exactly like a missing session, which
    is why both raise SessionNotFound instead of passing silently.
    Flushes but does not commit, matching the MAG repositories.
    """

    def __init__(self, session: AsyncSession, clock: Callable[[], datetime] = _utc_now) -> None:
        self._session = session
        self._clock = clock

    async def record(
        self,
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        allocation: BudgetAllocation,
        contributing: frozenset[Paradigm],
    ) -> None:
        await set_tenant_context(self._session, tenant_id)
        result = await self._session.execute(
            text("UPDATE sessions SET context_budget = CAST(:budget AS jsonb) WHERE id = :id"),
            {
                "budget": json.dumps(budget_record(allocation, contributing, self._clock())),
                "id": session_id,
            },
        )
        if cast(CursorResult[Any], result).rowcount != 1:
            raise SessionNotFound(session_id)
        await self._session.flush()
```

- [ ] **Step 5: Run to verify they pass, then commit**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_session_budget_record.py tests/integration/test_postgres_session_budget_recorder.py -q -p no:cacheprovider`
Expected: all pass.

```bash
git add src/orchestration/infrastructure/postgres_session_budget_recorder.py tests/unit/test_session_budget_record.py tests/integration/test_postgres_session_budget_recorder.py
git commit  # feat: record each turn's context budget in sessions.context_budget under RLS
```

---

### Task 8: The unified use case

**Files:**
- Create: `src/orchestration/application/unified_answer_question.py`
- Test: `tests/unit/test_unified_answer_question.py`

**Interfaces:**
- Consumes: `decide` (Task 1), `allocate`/`DEFAULT_SHARES` (Task 2), `assemble_context` (Task 3), `LatencyCascade` (Task 4), `QueryClassifier`/`SessionBudgetRecorder` ports, `QueryExceedsBudget`, `count_tokens`, RAG `EmbeddingModel`/`ChatModel`.
- Produces:
  - `StageTimings(embed_ms, route_ms, cascade_ms, assemble_ms, generate_ms)` (floats)
  - `UnifiedAnswer(answer: str, sources: list[ContextItem], decision: RoutingDecision | None, attempts: list[TierAttempt], allocation: BudgetAllocation, dropped: dict[Paradigm, int], degraded: bool, timings: StageTimings)`
  - `UnifiedAnswerQuestion(embedding_model, classifier: QueryClassifier | None, cascade: LatencyCascade, chat_model, *, total_context_tokens: int = 128_000, shares: BudgetShares = DEFAULT_SHARES, select_threshold: float = 0.5, uncertainty_margin: float = 0.15, budget_recorder: SessionBudgetRecorder | None = None)`
  - `async execute(tenant_id, user_id, session_id, question: str) -> UnifiedAnswer`

- [ ] **Step 1: Write the failing test**

```python
import uuid

import pytest

from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import ContextItem, Paradigm, TierOutcome, TierResult
from src.orchestration.domain.errors import QueryExceedsBudget
from tests.unit.orchestration_fakes import (
    FakeCascadeTier,
    FakeQueryClassifier,
    FakeSessionBudgetRecorder,
)
from tests.unit.rag_fakes import FakeChatModel, FakeEmbeddingModel

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG
_GENEROUS = TierTimeouts(cag=1.0, mag=1.0, rag=1.0)
_RAG_ONLY = {CAG: 0.0, MAG: 0.0, RAG: 1.0}
_QUESTION = "what changed today?"


def _hit(paradigm: Paradigm, content: str) -> TierResult:
    return TierResult(TierOutcome.HIT, [ContextItem(paradigm, content, 0.9, uuid.uuid4())])


def _ids() -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    return uuid.uuid4(), uuid.uuid4(), uuid.uuid4()


async def test_a_routed_question_flows_through_every_stage_and_records_its_budget():
    classifier = FakeQueryClassifier(_RAG_ONLY)
    cag = FakeCascadeTier(CAG, _hit(CAG, "stale"))
    rag = FakeCascadeTier(RAG, _hit(RAG, "fresh doc"))
    chat_model = FakeChatModel("the answer")
    recorder = FakeSessionBudgetRecorder()
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(),
        classifier,
        LatencyCascade([cag, rag], _GENEROUS),
        chat_model,
        budget_recorder=recorder,
    )
    tenant_id, user_id, session_id = _ids()

    result = await use_case.execute(tenant_id, user_id, session_id, _QUESTION)

    assert result.answer == "the answer"
    assert result.decision is not None
    assert result.decision.paradigms == frozenset({RAG})
    assert [attempt.paradigm for attempt in result.attempts] == [RAG]
    assert cag.requests == []
    assert result.allocation == allocate(128_000, {RAG})
    assert "fresh doc" in chat_model.last_context
    assert "(RAG)" in chat_model.last_context
    assert "stale" not in chat_model.last_context
    assert [source.content for source in result.sources] == ["fresh doc"]
    assert recorder.records == [(tenant_id, session_id, result.allocation, frozenset({RAG}))]
    assert result.degraded is False
    timings = result.timings
    assert min(
        timings.embed_ms, timings.route_ms, timings.cascade_ms,
        timings.assemble_ms, timings.generate_ms,
    ) >= 0.0


async def test_the_classifier_and_the_tiers_share_one_upstream_embedding_and_scope():
    embedder = FakeEmbeddingModel()
    classifier = FakeQueryClassifier(_RAG_ONLY)
    rag = FakeCascadeTier(RAG, _hit(RAG, "doc"))
    use_case = UnifiedAnswerQuestion(
        embedder, classifier, LatencyCascade([rag], _GENEROUS), FakeChatModel()
    )
    tenant_id, user_id, session_id = _ids()

    await use_case.execute(tenant_id, user_id, session_id, _QUESTION)

    assert classifier.calls == [(_QUESTION, embedder.embed(_QUESTION))]
    [request] = rag.requests
    assert request.query_embedding == embedder.embed(_QUESTION)
    assert (request.tenant_id, request.user_id, request.session_id, request.query) == (
        tenant_id, user_id, session_id, _QUESTION,
    )


async def test_without_a_classifier_the_source_cascade_answers_unrouted():
    cag = FakeCascadeTier(CAG, _hit(CAG, "cached"))
    rag = FakeCascadeTier(RAG, _hit(RAG, "fresh"))
    chat_model = FakeChatModel()
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(), None, LatencyCascade([cag, rag], _GENEROUS), chat_model
    )

    result = await use_case.execute(*_ids(), _QUESTION)

    assert result.decision is None
    assert [attempt.paradigm for attempt in result.attempts] == [CAG]
    assert "cached" in chat_model.last_context


async def test_a_question_longer_than_the_query_slice_is_rejected_before_any_work():
    classifier = FakeQueryClassifier(_RAG_ONLY)
    chat_model = FakeChatModel()
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(),
        classifier,
        LatencyCascade([FakeCascadeTier(RAG)], _GENEROUS),
        chat_model,
        total_context_tokens=100,
    )

    with pytest.raises(QueryExceedsBudget) as raised:
        await use_case.execute(*_ids(), "word " * 50)

    assert raised.value.query_slice == 10
    assert classifier.calls == []
    assert chat_model.last_question is None


async def test_items_that_do_not_fit_their_slice_are_reported_as_dropped():
    rag = FakeCascadeTier(RAG, _hit(RAG, "alpha " * 400))
    chat_model = FakeChatModel()
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(),
        FakeQueryClassifier(_RAG_ONLY),
        LatencyCascade([rag], _GENEROUS),
        chat_model,
        total_context_tokens=200,
    )

    result = await use_case.execute(*_ids(), "q")

    assert result.dropped[RAG] == 1
    assert result.sources == []
    assert chat_model.last_context == ""


def test_a_non_positive_context_window_is_rejected():
    with pytest.raises(ValueError):
        UnifiedAnswerQuestion(
            FakeEmbeddingModel(),
            None,
            LatencyCascade([FakeCascadeTier(RAG)]),
            FakeChatModel(),
            total_context_tokens=0,
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_unified_answer_question.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/orchestration/application/unified_answer_question.py`**

```python
import math
import time
import uuid
from dataclasses import dataclass

from src.orchestration.application.assemble_context import assemble_context
from src.orchestration.application.latency_cascade import LatencyCascade
from src.orchestration.domain.budget_allocator import DEFAULT_SHARES, allocate
from src.orchestration.domain.entities import (
    BudgetAllocation,
    BudgetShares,
    ContextItem,
    Paradigm,
    RoutingDecision,
    TierAttempt,
    TierRequest,
)
from src.orchestration.domain.errors import QueryExceedsBudget
from src.orchestration.domain.paradigm_router import (
    DEFAULT_SELECT_THRESHOLD,
    DEFAULT_UNCERTAINTY_MARGIN,
    decide,
)
from src.orchestration.domain.ports import QueryClassifier, SessionBudgetRecorder
from src.rag.domain.ports import ChatModel, EmbeddingModel
from src.shared.tokenization import count_tokens

DEFAULT_CONTEXT_TOKENS = 128_000


@dataclass(frozen=True)
class StageTimings:
    embed_ms: float
    route_ms: float
    cascade_ms: float
    assemble_ms: float
    generate_ms: float


@dataclass(frozen=True)
class UnifiedAnswer:
    answer: str
    sources: list[ContextItem]
    decision: RoutingDecision | None
    attempts: list[TierAttempt]
    allocation: BudgetAllocation
    dropped: dict[Paradigm, int]
    degraded: bool
    timings: StageTimings


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000


class UnifiedAnswerQuestion:
    """§3.4 Pattern 1, "The Smart Router": route, cascade, budget, answer.

    With classifier=None the cascade runs unrouted, exactly as Concept 5
    draws it -- the ablation baseline the router is measured against.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        classifier: QueryClassifier | None,
        cascade: LatencyCascade,
        chat_model: ChatModel,
        *,
        total_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        shares: BudgetShares = DEFAULT_SHARES,
        select_threshold: float = DEFAULT_SELECT_THRESHOLD,
        uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
        budget_recorder: SessionBudgetRecorder | None = None,
    ) -> None:
        if total_context_tokens <= 0:
            raise ValueError("total_context_tokens must be positive")
        self._embedder = embedding_model
        self._classifier = classifier
        self._cascade = cascade
        self._chat_model = chat_model
        self._total = total_context_tokens
        self._shares = shares
        self._select_threshold = select_threshold
        self._uncertainty_margin = uncertainty_margin
        self._budget_recorder = budget_recorder

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> UnifiedAnswer:
        # Same floor allocate() applies, so this check and the allocation
        # below can never disagree about the Query slice's size.
        query_slice = math.floor(self._total * self._shares.query)
        query_tokens = count_tokens(question)
        if query_tokens > query_slice:
            raise QueryExceedsBudget(query_tokens, query_slice)

        started = time.perf_counter()
        embedding = self._embedder.embed(question)
        embedded = time.perf_counter()

        decision: RoutingDecision | None = None
        if self._classifier is not None:
            scores = await self._classifier.score(question, embedding)
            decision = decide(scores, self._select_threshold, self._uncertainty_margin)
        routed = time.perf_counter()

        request = TierRequest(tenant_id, user_id, session_id, question, embedding)
        cascade_result = await self._cascade.run(request, decision)
        cascaded = time.perf_counter()

        allocation = allocate(self._total, cascade_result.contributing, self._shares)
        assembled = assemble_context(cascade_result.items, allocation)
        assembled_at = time.perf_counter()

        answer = await self._chat_model.generate(question=question, context=assembled.text)
        generated = time.perf_counter()

        if self._budget_recorder is not None:
            await self._budget_recorder.record(
                tenant_id, session_id, allocation, cascade_result.contributing
            )

        return UnifiedAnswer(
            answer=answer,
            sources=assembled.included,
            decision=decision,
            attempts=cascade_result.attempts,
            allocation=allocation,
            dropped=assembled.dropped,
            degraded=cascade_result.degraded,
            timings=StageTimings(
                embed_ms=_ms(started, embedded),
                route_ms=_ms(embedded, routed),
                cascade_ms=_ms(routed, cascaded),
                assemble_ms=_ms(cascaded, assembled_at),
                generate_ms=_ms(assembled_at, generated),
            ),
        )
```

- [ ] **Step 4: Run it, then the full unit suite, ruff, and mypy; commit**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider`
Expected: all pass.
Run: `../../../.venv/Scripts/python.exe -m ruff check src tests` and `../../../.venv/Scripts/python.exe -m mypy src`
Expected: no errors.

```bash
git add src/orchestration/application/unified_answer_question.py tests/unit/test_unified_answer_question.py
git commit  # feat: compose router, cascade, allocator, and generation into UnifiedAnswerQuestion
```

---

### Task 9: Integration tests against real infrastructure

**Files:**
- Create: `tests/integration/orchestration_env.py` (shared builder, not a test module)
- Create: `tests/integration/test_prototype_query_classifier.py`, `tests/integration/test_orchestration_meta_layer.py`, `tests/integration/test_orchestration_against_ollama.py`

**Interfaces:**
- Consumes: every earlier task; fixtures `db_session`, `qdrant_url`, `embedding_model`, `distilgpt2_tokenizer`, `distilgpt2_model` from `tests/integration/conftest.py`.
- Produces: `build_env(...) -> OrchestrationEnv`, `create_user_and_session`, `FixedScoresClassifier`, `ContextEchoChatModel`, the corpus constants, and the four measured thresholds `CAG_HIT`, `CAG_PARTIAL`, `MAG_HIT`, `MAG_PARTIAL` (reused by Task 11's runner).

- [ ] **Step 1: Measure the real similarities the thresholds must separate**

Run from the worktree root:

```bash
PYTHONPATH=. ../../../.venv/Scripts/python.exe - <<'PY'
from src.orchestration.domain.similarity import cosine_similarity
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder

POLICY_V1 = ("Our return policy allows customers to return unopened items within "
             "thirty days of purchase for a full refund.")
FACT = "The user prefers expedited two-day shipping on every order."
e = SentenceTransformersEmbedder()
pairs = {
    "MUST HIT  cag freshness q vs v1": ("What changed in the return policy today?", POLICY_V1),
    "MUST HIT  cag policy q vs v1": ("What is the return policy for unopened items?", POLICY_V1),
    "MUST MISS cag shipping q vs v1": ("How long does standard shipping take?", POLICY_V1),
    "MUST HIT  mag preference q vs fact": ("What shipping speed did I say I prefer?", FACT),
    "MUST MISS mag policy q vs fact": ("What is the return policy for unopened items?", FACT),
}
for label, (query, doc) in pairs.items():
    print(f"{label}: {cosine_similarity(e.embed(query), e.embed(doc)):.4f}")
PY
```

Set each hit threshold to the midpoint between its lowest MUST HIT score and its highest MUST MISS score, rounded down to two decimals, and each partial threshold to the MUST MISS score minus 0.05, floored at 0.0. If a MUST HIT score does not exceed its MUST MISS score, stop: the corpus sentences cannot separate the cases, and they must be reworded before any test is written.

- [ ] **Step 2: Create `tests/integration/orchestration_env.py`**

Fill the four threshold constants with Step 1's numbers and replace each constant's comment with the measured scores it separates.

```python
"""Shared real-infrastructure environment for the orchestration meta-layer's
integration tests: Postgres with RLS for MAG, Qdrant for RAG, real MiniLM
embeddings, and a real distilgpt2 HFFrozenCache for CAG that holds a
SUPERSEDED copy of the return policy while Qdrant holds the current one.
Not collected by pytest (no test_ prefix)."""
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.cascade_tiers import CagTier, MagTier, RagTier
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.domain.entities import Paradigm, TierRequest
from src.orchestration.domain.ports import CascadeTier, QueryClassifier
from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import Chunk
from src.rag.domain.ports import ChatModel, EmbeddingModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"

POLICY_V1 = (
    "Our return policy allows customers to return unopened items within thirty days "
    "of purchase for a full refund."
)
POLICY_V2 = (
    "Our return policy allows customers to return unopened items within forty-five days "
    "of purchase for a full refund. The window changed from thirty days today."
)
SHIPPING = (
    "Standard shipping takes five to seven business days. Expedited shipping arrives "
    "within two business days."
)
FACT_KEY = "preferred_shipping_speed"
FACT_VALUE = "The user prefers expedited two-day shipping on every order."

FRESHNESS_QUERY = "What changed in the return policy today?"
POLICY_QUERY = "What is the return policy for unopened items?"
PREFERENCE_QUERY = "What shipping speed did I say I prefer?"

CAG_HIT = 0.0  # Step 1: midpoint of measured MUST HIT / MUST MISS CAG scores
CAG_PARTIAL = 0.0  # Step 1: measured CAG MUST MISS score minus 0.05
MAG_HIT = 0.0  # Step 1: midpoint of measured MUST HIT / MUST MISS MAG scores
MAG_PARTIAL = 0.0  # Step 1: measured MAG MUST MISS score minus 0.05

# Generous on purpose: these tests check routing correctness against real
# stores. Latency against Concept 5's budgets is the evaluation runner's
# job, where first-call warm-up can be measured separately.
CORRECTNESS_TIMEOUTS = TierTimeouts(cag=5.0, mag=5.0, rag=10.0)


class FixedScoresClassifier(QueryClassifier):
    def __init__(self, scores: dict[Paradigm, float]) -> None:
        self._scores = scores

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        return dict(self._scores)


class ContextEchoChatModel(ChatModel):
    async def generate(self, question: str, context: str) -> str:
        return context

    async def complete(self, prompt: str) -> str:
        return prompt


@dataclass
class OrchestrationEnv:
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    session_id: uuid.UUID
    policy_document_id: uuid.UUID
    search: SearchDocuments
    warmed: CacheWarmedRetrieve
    repository: PostgresSemanticMemoryRepository

    def tiers(self, repository: PostgresSemanticMemoryRepository | None = None) -> list[CascadeTier]:
        return [
            CagTier(self.warmed, hit_threshold=CAG_HIT, partial_threshold=CAG_PARTIAL),
            MagTier(
                FindSemanticFacts(repository or self.repository),
                hit_threshold=MAG_HIT,
                partial_threshold=MAG_PARTIAL,
            ),
            RagTier(self.search, top_k=2),
        ]

    def cascade(self, timeouts: TierTimeouts = CORRECTNESS_TIMEOUTS) -> LatencyCascade:
        return LatencyCascade(self.tiers(), timeouts)

    def request(
        self, query: str, embedding_model: EmbeddingModel, user_id: uuid.UUID | None = None
    ) -> TierRequest:
        return TierRequest(
            self.tenant_id,
            user_id or self.user_id,
            self.session_id,
            query,
            embedding_model.embed(query),
        )


async def create_user_and_session(
    db_session: AsyncSession, tenant_id: uuid.UUID
) -> tuple[uuid.UUID, uuid.UUID]:
    await set_tenant_context(db_session, tenant_id)
    now = datetime.now(UTC)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id, "created_at": now, "updated_at": now,
        },
    )
    session_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "t"},
    )
    await db_session.commit()
    return user_id, session_id


async def build_env(
    db_session: AsyncSession,
    qdrant_url: str,
    embedding_model: EmbeddingModel,
    distilgpt2_tokenizer: Any,
    distilgpt2_model: Any,
) -> OrchestrationEnv:
    tenant_id = uuid.uuid4()
    user_id, session_id = await create_user_and_session(db_session, tenant_id)

    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    policy_id = uuid.uuid4()
    for document_id, content in ((policy_id, POLICY_V2), (uuid.uuid4(), SHIPPING)):
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=document_id,
            content=content,
            embedding=embedding_model.embed(content),
        )
        await vector_store.upsert(chunk, tenant_id)
    search = SearchDocuments(embedding_model, vector_store)

    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    cache.preload(tenant_id, policy_id, POLICY_V1)
    warmed = CacheWarmedRetrieve(embedding_model, cache, search, similarity_threshold=CAG_HIT)
    warmed.note_warmed(tenant_id, policy_id, POLICY_V1)

    # set_tenant_context is transaction-local, and PostgresSemanticMemoryRepository
    # (like every MAG repository) relies on its caller having set it -- so the
    # MAG tier's reads in these tests run inside this same transaction.
    await set_tenant_context(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    await repository.save(
        SemanticMemory(
            id=uuid.uuid4(),
            user_id=user_id,
            fact_key=FACT_KEY,
            fact_value=FACT_VALUE,
            embedding=embedding_model.embed(FACT_VALUE),
        ),
        tenant_id,
    )
    return OrchestrationEnv(
        tenant_id, user_id, session_id, policy_id, search, warmed, repository
    )
```

- [ ] **Step 3: Write `tests/integration/test_orchestration_meta_layer.py`**

```python
"""Real validation of the orchestration meta-layer's cascade against real
Postgres, Qdrant, MiniLM, and a distilgpt2 frozen cache. No LLM: generation
is ContextEchoChatModel, so this file needs only Docker."""
import asyncio
import json
import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.application.cascade_tiers import MagTier, RagTier
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
from src.orchestration.domain.entities import (
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierOutcome,
)
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)
from tests.integration.orchestration_env import (
    FACT_KEY,
    FACT_VALUE,
    FRESHNESS_QUERY,
    MAG_HIT,
    MAG_PARTIAL,
    POLICY_QUERY,
    PREFERENCE_QUERY,
    ContextEchoChatModel,
    FixedScoresClassifier,
    build_env,
)

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


def _route(*paradigms: Paradigm) -> RoutingDecision:
    return RoutingDecision(frozenset(paradigms), RoutingMode.CASCADE, {})


def _outcomes(result):
    return [(attempt.paradigm, attempt.outcome) for attempt in result.attempts]


async def test_routing_to_rag_keeps_a_superseded_cag_document_out_of_a_freshness_answer(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    cascade = env.cascade()

    unrouted = await cascade.run(env.request(FRESHNESS_QUERY, embedding_model))
    print(f"unrouted: {_outcomes(unrouted)} -> {[i.content[:60] for i in unrouted.items]}")
    assert _outcomes(unrouted) == [(CAG, TierOutcome.HIT)]
    assert any("thirty days" in item.content for item in unrouted.items)
    assert not any("forty-five" in item.content for item in unrouted.items)

    routed = await cascade.run(env.request(FRESHNESS_QUERY, embedding_model), _route(RAG))
    print(f"routed: {_outcomes(routed)} -> {[i.content[:60] for i in routed.items]}")
    assert [attempt.paradigm for attempt in routed.attempts] == [RAG]
    assert any("forty-five days" in item.content for item in routed.items)


async def test_a_real_cag_hit_short_circuits_before_mag_and_rag(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    result = await env.cascade().run(env.request(POLICY_QUERY, embedding_model), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.HIT)]
    assert result.items[0].source_id == env.policy_document_id


async def test_a_real_mag_fact_answers_a_question_about_the_users_own_preference(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    result = await env.cascade().run(
        env.request(PREFERENCE_QUERY, embedding_model), _route(MAG)
    )
    assert _outcomes(result) == [(MAG, TierOutcome.HIT)]
    assert result.items[0].content == f"{FACT_KEY}: {FACT_VALUE}"


async def test_another_users_fact_never_reaches_the_mag_tier(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    stranger = uuid.uuid4()
    result = await env.cascade().run(
        env.request(PREFERENCE_QUERY, embedding_model, user_id=stranger), _route(MAG)
    )
    assert _outcomes(result) == [(MAG, TierOutcome.MISS), (RAG, TierOutcome.HIT)]
    assert not any(FACT_VALUE in item.content for item in result.items)


async def test_the_unified_pipeline_answers_from_rag_and_records_the_real_budget(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    use_case = UnifiedAnswerQuestion(
        embedding_model,
        FixedScoresClassifier({CAG: 0.0, MAG: 0.0, RAG: 1.0}),
        env.cascade(),
        ContextEchoChatModel(),
        budget_recorder=PostgresSessionBudgetRecorder(db_session),
    )

    result = await use_case.execute(env.tenant_id, env.user_id, env.session_id, FRESHNESS_QUERY)

    assert "forty-five days" in result.answer
    assert "thirty days of purchase" not in result.answer
    await set_tenant_context(db_session, env.tenant_id)
    stored = (
        await db_session.execute(
            text("SELECT context_budget FROM sessions WHERE id = :id"), {"id": env.session_id}
        )
    ).scalar_one()
    stored = json.loads(stored) if isinstance(stored, str) else stored
    assert stored["contributing"] == ["rag"]
    assert stored["slices"]["rag"] == result.allocation.rag == 108_800


class _SlowSemanticMemoryRepository(PostgresSemanticMemoryRepository):
    # A real server-side delay on the real session, so the MAG timeout
    # cancels an in-flight database round-trip rather than a task that
    # never started.
    async def search_by_similarity(self, query_embedding, user_id, tenant_id, top_k):
        await self._session.execute(text("SELECT pg_sleep(0.5)"))
        return await super().search_by_similarity(query_embedding, user_id, tenant_id, top_k)


async def test_a_mag_query_cancelled_mid_flight_leaves_the_session_usable_after_rollback(
    db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    await db_session.commit()  # keep the seeded fact across the rollback below
    await set_tenant_context(db_session, env.tenant_id)
    slow = MagTier(
        FindSemanticFacts(_SlowSemanticMemoryRepository(db_session)),
        hit_threshold=MAG_HIT,
        partial_threshold=MAG_PARTIAL,
    )
    tight = LatencyCascade([slow, RagTier(env.search)], TierTimeouts(cag=5.0, mag=0.05, rag=10.0))

    first = await tight.run(env.request(PREFERENCE_QUERY, embedding_model), _route(MAG))
    assert _outcomes(first)[0] == (MAG, TierOutcome.TIMEOUT)

    await asyncio.sleep(0.6)
    await db_session.rollback()
    await set_tenant_context(db_session, env.tenant_id)
    second = await env.cascade().run(env.request(PREFERENCE_QUERY, embedding_model), _route(MAG))
    assert _outcomes(second) == [(MAG, TierOutcome.HIT)]
```

- [ ] **Step 4: Write `tests/integration/test_prototype_query_classifier.py`**

```python
import math

from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm
from src.orchestration.domain.paradigm_router import decide
from src.orchestration.infrastructure.prototype_query_classifier import (
    PrototypeQueryClassifier,
)
from src.orchestration.infrastructure.routing_exemplars import DEFAULT_ROUTING_EXEMPLARS


async def test_real_embeddings_produce_well_formed_scores_for_every_exemplar(embedding_model):
    classifier = PrototypeQueryClassifier(embedding_model)
    for exemplar in DEFAULT_ROUTING_EXEMPLARS:
        scores = await classifier.score(exemplar.query, embedding_model.embed(exemplar.query))
        assert set(scores) == set(PARADIGM_ORDER)
        assert all(math.isfinite(s) and 0.0 <= s <= 1.0 for s in scores.values())


async def test_a_lightly_reworded_exemplar_routes_to_include_its_own_label(embedding_model):
    # Accuracy on held-out queries is measured in the router report, not
    # gated here -- gating on it would invite tuning exemplars to pass.
    classifier = PrototypeQueryClassifier(embedding_model, k=3)
    reworded = [
        ("Please tell me what the employee handbook says about vacation days.", Paradigm.CAG),
        ("Remind me, which dietary restrictions did I mention?", Paradigm.MAG),
        ("Right now, what is the inventory level for the blue running shoes?", Paradigm.RAG),
    ]
    for query, expected in reworded:
        decision = decide(await classifier.score(query, embedding_model.embed(query)))
        print(f"{query!r} -> {sorted(p.value for p in decision.paradigms)} {decision.scores}")
        assert expected in decision.paradigms
```

- [ ] **Step 5: Write `tests/integration/test_orchestration_against_ollama.py`**

```python
"""Live checks against the local Ollama qwen3.5 model. Skips, with the reason,
wherever Ollama or the model is unavailable -- the same shape as the
vLLM-dependent CAG tests."""
import ollama
import pytest

from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm
from src.orchestration.infrastructure.llm_query_classifier import LlmQueryClassifier
from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from tests.integration.orchestration_env import (
    FRESHNESS_QUERY,
    FixedScoresClassifier,
    build_env,
)

_MODEL_ID = "qwen3.5"


@pytest.fixture(scope="module")
def qwen_available() -> None:
    try:
        names = [model.model or "" for model in ollama.Client().list().models]
    except Exception as exc:  # any connection failure means "not available here"
        pytest.skip(f"Ollama is not reachable at its default host: {exc}")
    if not any(name.split(":")[0] == _MODEL_ID for name in names):
        pytest.skip(f"{_MODEL_ID} is not pulled into the local Ollama")


async def test_the_real_model_returns_parseable_routing_scores(qwen_available):
    classifier = LlmQueryClassifier(OllamaChatModel(ollama.AsyncClient(), _MODEL_ID))
    for query in (
        "What's our refund policy?",
        "What changed in the policy today?",
        "What did I ask you to remember?",
    ):
        scores = await classifier.score(query, [])
        print(f"{query!r} -> {scores}")
        assert set(scores) == set(PARADIGM_ORDER)
    assert classifier.parse_failures == 0


async def test_the_unified_pipeline_answers_a_freshness_question_from_the_current_document(
    qwen_available, db_session, qdrant_url, embedding_model, distilgpt2_tokenizer,
    distilgpt2_model,
):
    env = await build_env(
        db_session, qdrant_url, embedding_model, distilgpt2_tokenizer, distilgpt2_model
    )
    use_case = UnifiedAnswerQuestion(
        embedding_model,
        FixedScoresClassifier({Paradigm.CAG: 0.0, Paradigm.MAG: 0.0, Paradigm.RAG: 1.0}),
        env.cascade(),
        OllamaChatModel(ollama.AsyncClient(), _MODEL_ID),
    )
    result = await use_case.execute(env.tenant_id, env.user_id, env.session_id, FRESHNESS_QUERY)
    print(f"answer: {result.answer!r}")
    assert "45" in result.answer or "forty-five" in result.answer.lower()
```

- [ ] **Step 6: Run the four files and handle what they reveal**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/integration/test_prototype_query_classifier.py tests/integration/test_orchestration_meta_layer.py tests/integration/test_orchestration_against_ollama.py tests/integration/test_postgres_session_budget_recorder.py -v -s -p no:cacheprovider`
Expected: all pass (Docker running; Ollama tests skip with a reason if Ollama is down).

If `test_a_mag_query_cancelled_mid_flight_leaves_the_session_usable_after_rollback` fails, that is a real finding, not a test to loosen: record the exact failure in `MagTier`'s docstring and the report, change the composition guidance to one `AsyncSession` per request, and make the test assert that a fresh session from a new engine on `app_database_url` serves the next request.

- [ ] **Step 7: Commit**

```bash
git add tests/integration/orchestration_env.py tests/integration/test_prototype_query_classifier.py tests/integration/test_orchestration_meta_layer.py tests/integration/test_orchestration_against_ollama.py
git commit  # test: validate routing, tiers, and the unified pipeline against real stores and Ollama
```

---

### Task 10: Router comparison — metrics, held-out query set, runner

**Files:**
- Create: `evaluation/domain/statistics.py`, `evaluation/domain/routing_metrics.py`, `evaluation/infrastructure/routing_report.py`
- Create: `evaluation/scenarios/orchestration-meta-layer/generate_routing_queries.py`, `evaluation/scenarios/orchestration-meta-layer/run_router_comparison.py` (no `__init__.py`: scenario directories are hyphenated and never imported as packages, matching `rag-crag/`)
- Generated then hand-reviewed: `evaluation/scenarios/orchestration-meta-layer/routing_queries.yaml`
- Generated: `evaluation/reports/orchestration-meta-layer-router.md`
- Test: `tests/unit/test_routing_metrics.py`, `tests/unit/test_routing_report.py`

**Interfaces:**
- Consumes: `decide` (Task 1), the three classifiers (Task 6), `SentenceTransformersEmbedder`, `OllamaChatModel`.
- Produces:
  - `RoutingObservation(query: str, expected: frozenset[Paradigm], decision: RoutingDecision, latency_ms: float)`
  - `PrecisionRecall(precision: float, recall: float)` (`nan` when undefined)
  - `RoutingMetrics(count, exact_match_rate, coverage_rate, confident_rate, per_paradigm: dict[Paradigm, PrecisionRecall], latency_p50_ms, latency_p95_ms)`
  - `summarize(observations: list[RoutingObservation]) -> RoutingMetrics`
  - `render_router_comparison(metrics: dict[str, RoutingMetrics], observations: dict[str, list[RoutingObservation]], notes: str) -> str`

- [ ] **Step 1: Write the failing metrics test**

```python
import math

import pytest

from evaluation.domain.routing_metrics import RoutingObservation, summarize
from src.orchestration.domain.entities import Paradigm, RoutingDecision, RoutingMode

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


def _obs(expected, routed, mode=RoutingMode.CASCADE, latency_ms=1.0):
    return RoutingObservation(
        "q", frozenset(expected), RoutingDecision(frozenset(routed), mode, {}), latency_ms
    )


def test_summarize_reports_exact_match_coverage_confidence_precision_recall_and_latency():
    metrics = summarize(
        [
            _obs({CAG}, {CAG}, latency_ms=1.0),
            _obs({RAG}, {CAG, RAG}, RoutingMode.PARALLEL, latency_ms=2.0),
            _obs({MAG, RAG}, {MAG}, latency_ms=3.0),
            _obs({MAG}, {RAG}, latency_ms=4.0),
        ]
    )
    assert metrics.count == 4
    assert metrics.exact_match_rate == 0.25
    assert metrics.coverage_rate == 0.5
    assert metrics.confident_rate == 0.75
    assert (metrics.per_paradigm[CAG].precision, metrics.per_paradigm[CAG].recall) == (0.5, 1.0)
    assert (metrics.per_paradigm[MAG].precision, metrics.per_paradigm[MAG].recall) == (1.0, 0.5)
    assert (metrics.per_paradigm[RAG].precision, metrics.per_paradigm[RAG].recall) == (0.5, 0.5)
    assert (metrics.latency_p50_ms, metrics.latency_p95_ms) == (3.0, 4.0)


def test_undefined_precision_and_recall_are_nan_rather_than_a_misleading_zero():
    metrics = summarize([_obs({CAG}, {RAG})])
    assert math.isnan(metrics.per_paradigm[CAG].precision)
    assert metrics.per_paradigm[CAG].recall == 0.0
    assert math.isnan(metrics.per_paradigm[MAG].precision)
    assert math.isnan(metrics.per_paradigm[MAG].recall)


def test_an_empty_observation_list_is_rejected():
    with pytest.raises(ValueError):
        summarize([])
```

- [ ] **Step 2: Implement `evaluation/domain/statistics.py` and `evaluation/domain/routing_metrics.py`**

`evaluation/domain/statistics.py` (shared with Task 11's cascade metrics):

```python
def nearest_rank_percentile(sorted_values: list[float], pct: float) -> float:
    # The same nearest-rank formula RunComparison uses, so latency figures
    # across this project's reports mean the same thing.
    if not sorted_values:
        raise ValueError("at least one value is required")
    index = min(len(sorted_values) - 1, round(pct * (len(sorted_values) - 1)))
    return sorted_values[index]
```

Add to `tests/unit/test_routing_metrics.py`:

```python
from evaluation.domain.statistics import nearest_rank_percentile


def test_nearest_rank_percentile_matches_run_comparisons_formula():
    assert nearest_rank_percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 3.0
    assert nearest_rank_percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    with pytest.raises(ValueError):
        nearest_rank_percentile([], 0.5)
```

`evaluation/domain/routing_metrics.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass

from evaluation.domain.statistics import nearest_rank_percentile
from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    Paradigm,
    RoutingDecision,
    RoutingMode,
)


@dataclass(frozen=True)
class RoutingObservation:
    query: str
    expected: frozenset[Paradigm]
    decision: RoutingDecision
    latency_ms: float


@dataclass(frozen=True)
class PrecisionRecall:
    # nan, not 0.0, when undefined: a classifier that never selects a
    # paradigm has no precision for it, and reporting 0% would read as
    # "always wrong" instead of "never tried".
    precision: float
    recall: float


@dataclass(frozen=True)
class RoutingMetrics:
    count: int
    exact_match_rate: float
    # expected is a subset of the routed set: a PARALLEL widening that
    # still includes every needed paradigm answers correctly, just slower.
    coverage_rate: float
    confident_rate: float
    per_paradigm: dict[Paradigm, PrecisionRecall]
    latency_p50_ms: float
    latency_p95_ms: float


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def summarize(observations: list[RoutingObservation]) -> RoutingMetrics:
    if not observations:
        raise ValueError("at least one observation is required")
    count = len(observations)
    per_paradigm: dict[Paradigm, PrecisionRecall] = {}
    for paradigm in PARADIGM_ORDER:
        true_pos = sum(
            1 for o in observations if paradigm in o.decision.paradigms and paradigm in o.expected
        )
        false_pos = sum(
            1
            for o in observations
            if paradigm in o.decision.paradigms and paradigm not in o.expected
        )
        false_neg = sum(
            1
            for o in observations
            if paradigm not in o.decision.paradigms and paradigm in o.expected
        )
        per_paradigm[paradigm] = PrecisionRecall(
            _ratio(true_pos, true_pos + false_pos), _ratio(true_pos, true_pos + false_neg)
        )
    latencies = sorted(o.latency_ms for o in observations)
    return RoutingMetrics(
        count=count,
        exact_match_rate=sum(1 for o in observations if o.decision.paradigms == o.expected)
        / count,
        coverage_rate=sum(1 for o in observations if o.expected <= o.decision.paradigms) / count,
        confident_rate=sum(1 for o in observations if o.decision.mode is RoutingMode.CASCADE)
        / count,
        per_paradigm=per_paradigm,
        latency_p50_ms=nearest_rank_percentile(latencies, 0.50),
        latency_p95_ms=nearest_rank_percentile(latencies, 0.95),
    )
```

- [ ] **Step 3: Write the failing report test**

```python
from evaluation.domain.routing_metrics import RoutingObservation, summarize
from evaluation.infrastructure.routing_report import render_router_comparison
from src.orchestration.domain.entities import Paradigm, RoutingDecision, RoutingMode

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


def _obs(query, expected, routed, mode=RoutingMode.CASCADE):
    return RoutingObservation(
        query, frozenset(expected), RoutingDecision(frozenset(routed), mode, {}), 1.5
    )


def test_the_report_tabulates_every_classifier_and_lists_its_misroutes():
    lexical = [_obs("refund policy?", {CAG}, {CAG}), _obs("changed today?", {RAG}, {CAG, RAG})]
    llm = [_obs("refund policy?", {CAG}, {CAG})]
    report = render_router_comparison(
        {"lexical": summarize(lexical), "llm": summarize(llm)},
        {"lexical": lexical, "llm": llm},
        notes="measured on this machine",
    )
    assert "measured on this machine" in report
    assert "| lexical | 2 | 50% | 100% | 100% |" in report
    assert "n/a" in report  # llm never selected MAG or RAG
    assert "- 'changed today?': expected RAG, routed CAG+RAG (cascade)" in report
    assert "## Misrouted by llm (0)" in report
    assert "- none" in report
```

- [ ] **Step 4: Implement `evaluation/infrastructure/routing_report.py`**

```python
from __future__ import annotations

import math

from evaluation.domain.routing_metrics import RoutingMetrics, RoutingObservation
from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.0%}"


def _labels(paradigms: frozenset[Paradigm]) -> str:
    return "+".join(p.value.upper() for p in PARADIGM_ORDER if p in paradigms)


def render_router_comparison(
    metrics: dict[str, RoutingMetrics],
    observations: dict[str, list[RoutingObservation]],
    notes: str,
) -> str:
    lines = [
        "# Orchestration Meta-Layer — Router Comparison",
        "",
        notes,
        "",
        "| Classifier | Queries | Exact route | Covers needed | Confident "
        "| CAG P / R | MAG P / R | RAG P / R | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        pairs = " | ".join(
            f"{_pct(m.per_paradigm[p].precision)} / {_pct(m.per_paradigm[p].recall)}"
            for p in PARADIGM_ORDER
        )
        lines.append(
            f"| {name} | {m.count} | {_pct(m.exact_match_rate)} | {_pct(m.coverage_rate)} "
            f"| {_pct(m.confident_rate)} | {pairs} "
            f"| {m.latency_p50_ms:.2f} | {m.latency_p95_ms:.2f} |"
        )
    for name, observed in observations.items():
        misrouted = [o for o in observed if o.decision.paradigms != o.expected]
        lines += ["", f"## Misrouted by {name} ({len(misrouted)})", ""]
        lines += [
            f"- {o.query!r}: expected {_labels(o.expected)}, "
            f"routed {_labels(o.decision.paradigms)} ({o.decision.mode.value})"
            for o in misrouted
        ] or ["- none"]
    return "\n".join(lines) + "\n"
```

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_routing_metrics.py tests/unit/test_routing_report.py -q -p no:cacheprovider`
Expected: all pass. Commit:

```bash
git add evaluation/domain/statistics.py evaluation/domain/routing_metrics.py evaluation/infrastructure/routing_report.py tests/unit/test_routing_metrics.py tests/unit/test_routing_report.py
git commit  # feat: add routing accuracy metrics and the router comparison report renderer
```

- [ ] **Step 5: Create the query generator**

`evaluation/scenarios/orchestration-meta-layer/generate_routing_queries.py`:

```python
"""Generates the router comparison's held-out query set with the local qwen3.5
model. Run once, only after the three classifiers, their cue lists, and their
exemplars are committed, so the evaluation phrasings are not written by the
author of the cues. Review every label by hand afterward; record any
correction in that entry's review_note instead of editing silently.

Usage (from the repository root):
    PYTHONPATH=. python evaluation/scenarios/orchestration-meta-layer/generate_routing_queries.py
"""
import asyncio
import re
from pathlib import Path

import ollama
import yaml

_OUT = Path(__file__).parent / "routing_queries.yaml"
_MODEL_ID = "qwen3.5"
_PER_ROUTE = 8
_ROUTES = {
    "cag": (
        "stable reference knowledge that rarely changes, such as a company policy, a "
        "product manual, a how-to guide, or an explanation of existing code"
    ),
    "rag": (
        "fresh or changing external information, such as today's figures, live status, "
        "recent news, or something that changed recently"
    ),
    "mag": (
        "only the user's own earlier conversation with the assistant, their stated "
        "preferences, or something they asked the assistant to remember"
    ),
    "mag,rag": (
        "BOTH the user's own earlier conversation or preferences AND fresh or changing "
        "external information, in the same question"
    ),
    "cag,rag": (
        "BOTH stable reference documentation AND whether that documented information has "
        "changed or is available right now, in the same question"
    ),
}
_CONCEPT_ONE = [
    ("What's our refund policy?", ["cag"]),
    ("What changed in the policy today?", ["rag"]),
    ("Continue where we left off yesterday", ["mag"]),
    ("Compare today's sales with last month", ["mag", "rag"]),
    ("Explain this code file", ["cag"]),
    ("What did I ask you to remember?", ["mag"]),
]
_PROMPT = (
    "Write {n} different requests that a user might send to an AI assistant used by an "
    "online store's customers and staff. Every request must need {description}. Vary the "
    "topic and the wording. Output one request per line, with no numbering and nothing else."
)
_LEADING_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


async def _generate(client: ollama.AsyncClient, description: str) -> list[str]:
    response = await client.chat(
        model=_MODEL_ID,
        messages=[{"role": "user", "content": _PROMPT.format(n=_PER_ROUTE, description=description)}],
    )
    lines = [_LEADING_MARKER.sub("", line).strip() for line in response.message.content.splitlines()]
    return [line for line in lines if len(line) > 10][:_PER_ROUTE]


async def _run() -> None:
    client = ollama.AsyncClient()
    entries: list[dict[str, object]] = []
    for labels, description in _ROUTES.items():
        generated = await _generate(client, description)
        print(f"{labels}: {len(generated)} generated")
        entries.extend(
            {"query": query, "expected": labels.split(","), "source": _MODEL_ID}
            for query in generated
        )
    entries.extend(
        {"query": query, "expected": expected, "source": "concept-1-table"}
        for query, expected in _CONCEPT_ONE
    )
    document = {"name": "orchestration-meta-layer-routing", "queries": entries}
    _OUT.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {len(entries)} queries to {_OUT}")


if __name__ == "__main__":
    asyncio.run(_run())
```

- [ ] **Step 6: Generate, review every label, and commit the set**

Run: `PYTHONPATH=. ../../../.venv/Scripts/python.exe evaluation/scenarios/orchestration-meta-layer/generate_routing_queries.py`
Expected: five `<labels>: 8 generated` lines (fewer is acceptable and reported) and `wrote N queries`.

Read every generated entry. Where a query does not actually need exactly its `expected` paradigms, change `expected` and add `review_note: "<why the generated label was wrong>"` to that entry. Count the corrections for the report. Commit the generator and the reviewed YAML:

```bash
git add evaluation/scenarios/orchestration-meta-layer/generate_routing_queries.py evaluation/scenarios/orchestration-meta-layer/routing_queries.yaml
git commit  # test: generate and hand-review the held-out routing query set
```

- [ ] **Step 7: Create the router runner**

`evaluation/scenarios/orchestration-meta-layer/run_router_comparison.py`:

```python
"""Router comparison: lexical cues vs. MiniLM nearest-prototype vs. qwen3.5, on
the held-out routing set. Not pytest-collected. Needs Ollama with qwen3.5.

Usage (from the repository root):
    PYTHONPATH=. python evaluation/scenarios/orchestration-meta-layer/run_router_comparison.py
"""
import asyncio
import time
from pathlib import Path
from typing import Any

import ollama
import yaml

from evaluation.domain.routing_metrics import RoutingObservation, summarize
from evaluation.infrastructure.routing_report import render_router_comparison
from src.orchestration.domain.entities import Paradigm
from src.orchestration.domain.paradigm_router import decide
from src.orchestration.domain.ports import QueryClassifier
from src.orchestration.infrastructure.lexical_query_classifier import LexicalQueryClassifier
from src.orchestration.infrastructure.llm_query_classifier import LlmQueryClassifier
from src.orchestration.infrastructure.prototype_query_classifier import (
    PrototypeQueryClassifier,
)
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder

_QUERIES = Path(__file__).parent / "routing_queries.yaml"
_REPORT = Path("evaluation/reports/orchestration-meta-layer-router.md")


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


async def _observe(
    classifier: QueryClassifier,
    queries: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
) -> list[RoutingObservation]:
    observations = []
    for entry in queries:
        query = str(entry["query"])
        expected = frozenset(Paradigm(str(value)) for value in entry["expected"])
        started = time.perf_counter()
        decision = decide(await classifier.score(query, embeddings[query]))
        observations.append(RoutingObservation(query, expected, decision, _ms(started)))
    return observations


async def _run() -> None:
    queries = yaml.safe_load(_QUERIES.read_text(encoding="utf-8"))["queries"]
    embedder = SentenceTransformersEmbedder()
    embedder.embed("warm up")
    embed_ms: list[float] = []
    embeddings: dict[str, list[float]] = {}
    for entry in queries:
        started = time.perf_counter()
        embeddings[entry["query"]] = embedder.embed(entry["query"])
        embed_ms.append(_ms(started))
    embed_ms.sort()

    llm = LlmQueryClassifier(OllamaChatModel(ollama.AsyncClient(), "qwen3.5"))
    classifiers: dict[str, QueryClassifier] = {
        "lexical": LexicalQueryClassifier(),
        "prototype (MiniLM, k=5)": PrototypeQueryClassifier(embedder),
        "llm (qwen3.5)": llm,
    }
    for classifier in classifiers.values():  # untimed warm-up: model load is not routing cost
        await classifier.score("warm up", embeddings[queries[0]["query"]])
    llm.parse_failures = 0

    observations = {name: await _observe(c, queries, embeddings) for name, c in classifiers.items()}
    metrics = {name: summarize(observed) for name, observed in observations.items()}
    reviewed = sum(1 for entry in queries if entry.get("review_note"))
    notes = (
        f"{len(queries)} queries ({sum(1 for e in queries if e['source'] != 'concept-1-table')} "
        f"generated by qwen3.5 after the classifiers were committed, {reviewed} of their labels "
        f"corrected by hand review, plus Concept 1's six example queries verbatim). Latency is "
        f"classify + decide only; query embedding is paid once per question by "
        f"UnifiedAnswerQuestion whichever classifier runs, and measured separately here: "
        f"p50 {embed_ms[len(embed_ms) // 2]:.2f}ms, max {embed_ms[-1]:.2f}ms (MiniLM, CPU). "
        f"LLM classifier parse failures: {llm.parse_failures} of {len(queries)}."
    )
    report = render_router_comparison(metrics, observations, notes)
    _REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    asyncio.run(_run())
```

- [ ] **Step 8: Run it and commit the generated report**

Run: `PYTHONPATH=. ../../../.venv/Scripts/python.exe evaluation/scenarios/orchestration-meta-layer/run_router_comparison.py`
Expected: the table prints and `evaluation/reports/orchestration-meta-layer-router.md` is written. Run `../../../.venv/Scripts/python.exe -m ruff check evaluation tests src`; expected: clean.

```bash
git add evaluation/scenarios/orchestration-meta-layer/run_router_comparison.py evaluation/reports/orchestration-meta-layer-router.md
git commit  # test: measure the three router classifiers' accuracy and latency
```

---

### Task 11: Cascade measurements, the router ablation, and the self-versus-self comparison

**Files:**
- Create: `evaluation/domain/cascade_metrics.py`, `evaluation/infrastructure/cascade_report.py`
- Create: `evaluation/scenarios/orchestration-meta-layer/queries.yaml`, `evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py`
- Generated: `evaluation/reports/orchestration-meta-layer-cascade.md`, `evaluation/reports/orchestration-meta-layer-comparison.md`
- Create: `evaluation/reports/orchestration-meta-layer.md` (narrative)
- Test: `tests/unit/test_cascade_metrics.py`, `tests/unit/test_cascade_report.py`

**Interfaces:**
- Consumes: `nearest_rank_percentile` (Task 10), every `src/orchestration` piece, `RunComparison`, `render`, `OllamaJudge`, `load_scenario`, and `CAG_HIT`/`CAG_PARTIAL`/`MAG_HIT`/`MAG_PARTIAL`/`create_user_and_session` from `tests/integration/orchestration_env.py`.
- Produces:
  - `TierLatencySummary(paradigm, attempts, outcomes: dict[TierOutcome, int], p50_ms, p95_ms, budget_ms, within_budget_rate)`
  - `summarize_tier_latency(attempts: list[TierAttempt], budgets_ms: dict[Paradigm, float]) -> list[TierLatencySummary]`
  - `StaleAnswerTally(arm, runs, stale, fresh, mixed)` with `stale_rate`; `tally_staleness(arm, contexts, stale_marker, fresh_marker) -> StaleAnswerTally`
  - `AllocatorTally(window_tokens, turns, dynamic_dropped, static_dropped, dynamic_tokens, static_tokens)`
  - `render_cascade_measurements(tiers, tallies, allocator: AllocatorTally, notes: str) -> str`

- [ ] **Step 1: Write the failing metrics and report tests**

`tests/unit/test_cascade_metrics.py`:

```python
from evaluation.domain.cascade_metrics import summarize_tier_latency, tally_staleness
from src.orchestration.domain.entities import Paradigm, TierAttempt, TierOutcome

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG
_BUDGETS = {CAG: 10.0, MAG: 50.0, RAG: 2000.0}


def test_tier_latency_is_summarized_per_attempted_tier_against_its_own_budget():
    summaries = summarize_tier_latency(
        [
            TierAttempt(CAG, TierOutcome.HIT, 2.0),
            TierAttempt(CAG, TierOutcome.TIMEOUT, 30.0),
            TierAttempt(RAG, TierOutcome.HIT, 100.0),
        ],
        _BUDGETS,
    )
    assert [s.paradigm for s in summaries] == [CAG, RAG]
    cag, rag = summaries
    assert cag.attempts == 2
    assert cag.outcomes == {TierOutcome.HIT: 1, TierOutcome.TIMEOUT: 1}
    assert (cag.p50_ms, cag.p95_ms, cag.budget_ms) == (2.0, 30.0, 10.0)
    assert cag.within_budget_rate == 0.5
    assert rag.within_budget_rate == 1.0


def test_staleness_separates_stale_only_current_only_and_mixed_contexts():
    tally = tally_staleness(
        "router off",
        ["within thirty days", "within forty-five days", "within thirty days / forty-five days", ""],
        stale_marker="within thirty days",
        fresh_marker="forty-five days",
    )
    assert (tally.runs, tally.stale, tally.fresh, tally.mixed) == (4, 1, 1, 1)
    assert tally.stale_rate == 0.25
```

`tests/unit/test_cascade_report.py`:

```python
from evaluation.domain.cascade_metrics import (
    AllocatorTally,
    StaleAnswerTally,
    TierLatencySummary,
)
from evaluation.infrastructure.cascade_report import render_cascade_measurements
from src.orchestration.domain.entities import Paradigm, TierOutcome


def test_the_report_renders_tier_latency_staleness_and_allocator_sections():
    report = render_cascade_measurements(
        [TierLatencySummary(Paradigm.CAG, 4, {TierOutcome.HIT: 3, TierOutcome.MISS: 1},
                            1.25, 3.5, 10.0, 1.0)],
        [StaleAnswerTally("router off", 6, 6, 0, 0)],
        AllocatorTally(1_000, 10, dynamic_dropped=1, static_dropped=7,
                       dynamic_tokens=900, static_tokens=400),
        notes="measured here",
    )
    assert "measured here" in report
    assert "| CAG | 4 | hit 3, miss 1 | 1.25 | 3.50 | 10 | 100% |" in report
    assert "| router off | 6 | 6 | 0 | 0 | 100% |" in report
    assert "| 1000 | 10 | 1 | 7 | 900 | 400 |" in report
```

- [ ] **Step 2: Run them to verify they fail**

Run: `../../../.venv/Scripts/python.exe -m pytest tests/unit/test_cascade_metrics.py tests/unit/test_cascade_report.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `evaluation/domain/cascade_metrics.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from evaluation.domain.statistics import nearest_rank_percentile
from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    Paradigm,
    TierAttempt,
    TierOutcome,
)


@dataclass(frozen=True)
class TierLatencySummary:
    paradigm: Paradigm
    attempts: int
    outcomes: dict[TierOutcome, int]
    p50_ms: float
    p95_ms: float
    budget_ms: float
    within_budget_rate: float


def summarize_tier_latency(
    attempts: list[TierAttempt], budgets_ms: dict[Paradigm, float]
) -> list[TierLatencySummary]:
    summaries: list[TierLatencySummary] = []
    for paradigm in PARADIGM_ORDER:
        mine = [attempt for attempt in attempts if attempt.paradigm is paradigm]
        if not mine:
            continue
        elapsed = sorted(attempt.elapsed_ms for attempt in mine)
        counts = {o: sum(1 for a in mine if a.outcome is o) for o in TierOutcome}
        budget = budgets_ms[paradigm]
        summaries.append(
            TierLatencySummary(
                paradigm=paradigm,
                attempts=len(mine),
                outcomes={outcome: n for outcome, n in counts.items() if n},
                p50_ms=nearest_rank_percentile(elapsed, 0.50),
                p95_ms=nearest_rank_percentile(elapsed, 0.95),
                budget_ms=budget,
                within_budget_rate=sum(1 for a in mine if a.elapsed_ms <= budget) / len(mine),
            )
        )
    return summaries


@dataclass(frozen=True)
class StaleAnswerTally:
    arm: str
    runs: int
    stale: int  # only the superseded text reached the model
    fresh: int  # only the current text reached the model
    mixed: int  # both did, leaving the model to pick one

    @property
    def stale_rate(self) -> float:
        return self.stale / self.runs if self.runs else 0.0


def tally_staleness(
    arm: str, contexts: list[str], stale_marker: str, fresh_marker: str
) -> StaleAnswerTally:
    has_stale = [stale_marker in context for context in contexts]
    has_fresh = [fresh_marker in context for context in contexts]
    return StaleAnswerTally(
        arm=arm,
        runs=len(contexts),
        stale=sum(1 for s, f in zip(has_stale, has_fresh, strict=True) if s and not f),
        fresh=sum(1 for s, f in zip(has_stale, has_fresh, strict=True) if f and not s),
        mixed=sum(1 for s, f in zip(has_stale, has_fresh, strict=True) if s and f),
    )


@dataclass(frozen=True)
class AllocatorTally:
    window_tokens: int
    turns: int
    dynamic_dropped: int
    static_dropped: int
    dynamic_tokens: int
    static_tokens: int
```

- [ ] **Step 4: Implement `evaluation/infrastructure/cascade_report.py`**

```python
from __future__ import annotations

from evaluation.domain.cascade_metrics import (
    AllocatorTally,
    StaleAnswerTally,
    TierLatencySummary,
)


def render_cascade_measurements(
    tiers: list[TierLatencySummary],
    tallies: list[StaleAnswerTally],
    allocator: AllocatorTally,
    notes: str,
) -> str:
    lines = [
        "# Orchestration Meta-Layer — Cascade Measurements",
        "",
        notes,
        "",
        "## Tier latency against Concept 5's budgets",
        "",
        "| Tier | Attempts | Outcomes | p50 ms | p95 ms | Budget ms | Within budget |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in tiers:
        outcomes = ", ".join(f"{outcome.value} {n}" for outcome, n in s.outcomes.items())
        lines.append(
            f"| {s.paradigm.value.upper()} | {s.attempts} | {outcomes} | {s.p50_ms:.2f} "
            f"| {s.p95_ms:.2f} | {s.budget_ms:.0f} | {s.within_budget_rate:.0%} |"
        )
    lines += [
        "",
        "## Superseded frozen-cache text reaching the model, router off vs. on",
        "",
        "| Arm | Runs | Stale only | Current only | Both | Stale rate |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {t.arm} | {t.runs} | {t.stale} | {t.fresh} | {t.mixed} | {t.stale_rate:.0%} |"
        for t in tallies
    ]
    lines += [
        "",
        "## Dynamic reallocation vs. static base slices",
        "",
        "| Window tokens | Turns | Dropped (dynamic) | Dropped (static) "
        "| Tokens used (dynamic) | Tokens used (static) |",
        "|---|---|---|---|---|---|",
        f"| {allocator.window_tokens} | {allocator.turns} | {allocator.dynamic_dropped} "
        f"| {allocator.static_dropped} | {allocator.dynamic_tokens} | {allocator.static_tokens} |",
    ]
    return "\n".join(lines) + "\n"
```

Run the two test files; expected: pass. Commit:

```bash
git add evaluation/domain/cascade_metrics.py evaluation/infrastructure/cascade_report.py tests/unit/test_cascade_metrics.py tests/unit/test_cascade_report.py
git commit  # feat: add tier-latency, staleness, and allocator measurement summaries
```

- [ ] **Step 5: Create `evaluation/scenarios/orchestration-meta-layer/queries.yaml`**

```yaml
name: orchestration-meta-layer
questions:
  - question: What is the return window for unopened items?
    success_criterion: States forty-five days (the current policy), not thirty.
    gold_passage: >-
      Our return policy allows customers to return unopened items within forty-five days
      of purchase for a full refund. The window changed from thirty days today.
  - question: What changed in the return policy today?
    success_criterion: States the window changed to forty-five days.
    gold_passage: >-
      Our return policy allows customers to return unopened items within forty-five days
      of purchase for a full refund. The window changed from thirty days today.
  - question: How long does standard shipping take?
    success_criterion: States five to seven business days.
    gold_passage: >-
      Standard shipping takes five to seven business days. Expedited shipping arrives
      within two business days.
  - question: How do I set up an account?
    success_criterion: Mentions the twelve-character password or the verification email.
    gold_passage: >-
      To set up an account, provide your email address and choose a password of at least
      twelve characters. A verification email is sent immediately after registration.
  - question: How long is the product warranty?
    success_criterion: States a two-year limited warranty.
    gold_passage: Every product carries a two-year limited warranty covering manufacturing defects.
  - question: Are the blue running shoes in stock today?
    success_criterion: States they are out of stock in sizes 9 and 10.
    gold_passage: >-
      As of this morning, the blue running shoes are out of stock in sizes 9 and 10, with
      a restock expected on Friday.
  - question: Is there a sale on backpacks today?
    success_criterion: States 20% off backpacks today.
    gold_passage: A flash sale today takes 20% off every backpack until midnight.
  - question: What shipping speed did I say I prefer?
    success_criterion: States expedited two-day shipping.
    gold_passage: The user prefers expedited two-day shipping on every order.
  - question: What shoe size do I wear?
    success_criterion: States size 10.
    gold_passage: The user wears size 10 running shoes.
  - question: Are the running shoes in my size in stock right now?
    success_criterion: States that size 10 is out of stock.
    gold_passage: >-
      The user wears size 10 running shoes. As of this morning, the blue running shoes are
      out of stock in sizes 9 and 10, with a restock expected on Friday.
```

- [ ] **Step 6: Create `evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py`**

```python
"""Cascade measurements for the orchestration meta-layer, in three parts:
tier latency against Concept 5's 10/50/2000ms budgets, the router-off vs.
router-on stale-text ablation, and a self-versus-self RunComparison
(RAG-only AnswerQuestion vs. UnifiedAnswerQuestion) on qwen3.5.

Starts its own Postgres and Qdrant containers, so it needs Docker and Ollama
but no compose stack. The four tier thresholds are imported unchanged from
tests/integration/orchestration_env.py, where they were measured on a
smaller corpus; this run tests whether they carry over rather than re-tuning
them. Not pytest-collected.

Usage (from the repository root):
    PYTHONPATH=. python evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py
"""
import asyncio
import dataclasses
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import ollama
from sqlalchemy.engine import make_url
from testcontainers.postgres import PostgresContainer
from testcontainers.qdrant import QdrantContainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from alembic import command
from alembic.config import Config
from evaluation.application.run_comparison import RunComparison
from evaluation.domain.cascade_metrics import (
    AllocatorTally,
    StaleAnswerTally,
    summarize_tier_latency,
    tally_staleness,
)
from evaluation.domain.entities import Answer
from evaluation.infrastructure.cascade_report import render_cascade_measurements
from evaluation.infrastructure.markdown_report import render
from evaluation.infrastructure.ollama_judge import OllamaJudge
from evaluation.scenarios.loader import load_scenario
from src.identity.infrastructure.db import get_engine, get_sessionmaker, set_tenant_context
from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.application.assemble_context import assemble_context
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.cascade_tiers import CagTier, MagTier, RagTier
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    Paradigm,
    TierAttempt,
    TierRequest,
)
from src.orchestration.domain.paradigm_router import decide
from src.orchestration.domain.ports import QueryClassifier
from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache
from src.orchestration.infrastructure.lexical_query_classifier import LexicalQueryClassifier
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)
from src.orchestration.infrastructure.prototype_query_classifier import (
    PrototypeQueryClassifier,
)
from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import Chunk
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from tests.integration.orchestration_env import (
    CAG_HIT,
    CAG_PARTIAL,
    MAG_HIT,
    MAG_PARTIAL,
    create_user_and_session,
)

_SCENARIO_DIR = Path(__file__).parent
_CASCADE_REPORT = Path("evaluation/reports/orchestration-meta-layer-cascade.md")
_COMPARISON_REPORT = Path("evaluation/reports/orchestration-meta-layer-comparison.md")
_MODEL_ID = "qwen3.5"
_APP_DB_PASSWORD = "evaluation-only-app-user-password"
_LATENCY_REPEATS = 5
_SMALL_WINDOW = 1_000
_BUDGETS_MS = {Paradigm.CAG: 10.0, Paradigm.MAG: 50.0, Paradigm.RAG: 2000.0}
_GENEROUS = TierTimeouts(cag=5.0, mag=5.0, rag=10.0)

_SHIPPING = (
    "Standard shipping takes five to seven business days. Expedited shipping arrives "
    "within two business days."
)
_ACCOUNT = (
    "To set up an account, provide your email address and choose a password of at least "
    "twelve characters. A verification email is sent immediately after registration."
)
_WARRANTY = "Every product carries a two-year limited warranty covering manufacturing defects."
# (text Qdrant holds now, text frozen into CAG or None when the document is RAG-only)
_DOCUMENTS: list[tuple[str, str | None]] = [
    (
        "Our return policy allows customers to return unopened items within forty-five days "
        "of purchase for a full refund. The window changed from thirty days today.",
        "Our return policy allows customers to return unopened items within thirty days of "
        "purchase for a full refund.",
    ),
    (_SHIPPING, _SHIPPING),
    (_ACCOUNT, _ACCOUNT),
    (_WARRANTY, _WARRANTY),
    (
        "As of this morning, the blue running shoes are out of stock in sizes 9 and 10, with "
        "a restock expected on Friday.",
        None,
    ),
    ("A flash sale today takes 20% off every backpack until midnight.", None),
]
_FACTS = [
    ("preferred_shipping_speed", "The user prefers expedited two-day shipping on every order."),
    ("shoe_size", "The user wears size 10 running shoes."),
    ("loyalty_tier", "The user is a Gold loyalty member."),
]
# Every pipeline piece here is deterministic, so the ablation uses varied
# phrasings rather than repeats of one phrasing.
_FRESHNESS_QUERIES = [
    "What changed in the return policy today?",
    "Has the return window been updated today?",
    "What is the latest return policy?",
    "Did the refund window change recently?",
    "What is the current number of days to return an item?",
    "Is there a new return policy as of today?",
]
_TEN = re.compile(r"\b(10|ten)\b")
_CHECKS: list[Callable[[str], bool]] = [
    lambda t: "45" in t or "forty-five" in t,
    lambda t: "45" in t or "forty-five" in t,
    lambda t: ("five" in t and "seven" in t) or ("5" in t and "7" in t),
    lambda t: "verification" in t or "twelve" in t or "12" in t,
    lambda t: any(term in t for term in ("two-year", "two year", "2-year", "2 year")),
    lambda t: "out of stock" in t,
    lambda t: "20%" in t or "20 percent" in t or "twenty percent" in t,
    lambda t: "expedited" in t or "two-day" in t or "2-day" in t,
    lambda t: _TEN.search(t) is not None,
    lambda t: "out of stock" in t and _TEN.search(t) is not None,
]


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


async def _measure(app_url: str, qdrant_url: str) -> None:
    scenario = load_scenario(_SCENARIO_DIR)
    questions = [q.question for q in scenario.questions]
    checks = dict(zip(questions, _CHECKS, strict=True))
    embedder = SentenceTransformersEmbedder()
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    chat_model = OllamaChatModel(ollama.AsyncClient(), _MODEL_ID)
    engine = get_engine(app_url)

    async with get_sessionmaker(engine)() as db_session:
        tenant_id = uuid.uuid4()
        user_id, session_id = await create_user_and_session(db_session, tenant_id)
        vector_store = QdrantVectorStore(qdrant_url)
        await vector_store.ensure_collection()
        search = SearchDocuments(embedder, vector_store)
        cache = HFFrozenCache(tokenizer=tokenizer, model=model)
        warmed = CacheWarmedRetrieve(embedder, cache, search, similarity_threshold=CAG_HIT)
        for current, frozen in _DOCUMENTS:
            document_id = uuid.uuid4()
            chunk = Chunk(
                id=uuid.uuid4(), document_id=document_id, content=current,
                embedding=embedder.embed(current),
            )
            await vector_store.upsert(chunk, tenant_id)
            if frozen is not None:
                cache.preload(tenant_id, document_id, frozen)
                warmed.note_warmed(tenant_id, document_id, frozen)
        await set_tenant_context(db_session, tenant_id)
        repository = PostgresSemanticMemoryRepository(db_session)
        for key, value in _FACTS:
            fact = SemanticMemory(
                id=uuid.uuid4(), user_id=user_id, fact_key=key, fact_value=value,
                embedding=embedder.embed(value),
            )
            await repository.save(fact, tenant_id)
        await db_session.commit()

        async def begin_turn() -> None:
            # Every turn starts from a clean transaction with the tenant set:
            # MAG repositories rely on the transaction-local tenant context,
            # and a MAG timeout may have cancelled a query mid-flight.
            await db_session.rollback()
            await set_tenant_context(db_session, tenant_id)

        def cascade(timeouts: TierTimeouts) -> LatencyCascade:
            return LatencyCascade(
                [
                    CagTier(warmed, hit_threshold=CAG_HIT, partial_threshold=CAG_PARTIAL),
                    MagTier(
                        FindSemanticFacts(repository),
                        hit_threshold=MAG_HIT,
                        partial_threshold=MAG_PARTIAL,
                    ),
                    RagTier(search, top_k=3),
                ],
                timeouts,
            )

        def request(query: str, embedding: list[float]) -> TierRequest:
            return TierRequest(tenant_id, user_id, session_id, query, embedding)

        prototype = PrototypeQueryClassifier(embedder)
        lexical = LexicalQueryClassifier()

        # Part 1: tier latency under Concept 5's real budgets.
        budgeted = cascade(TierTimeouts())
        await begin_turn()
        await budgeted.run(request("warm up", embedder.embed("warm up")))  # untimed warm-up
        attempts: list[TierAttempt] = []
        embed_ms: list[float] = []
        for _ in range(_LATENCY_REPEATS):
            for question in questions:
                await begin_turn()
                started = time.perf_counter()
                embedding = embedder.embed(question)
                embed_ms.append(_ms(started))
                decision = decide(await prototype.score(question, embedding))
                attempts.extend((await budgeted.run(request(question, embedding), decision)).attempts)
        embed_ms.sort()
        tiers = summarize_tier_latency(attempts, _BUDGETS_MS)

        # Part 2: which return-policy text reaches the model, router off vs. on.
        generous = cascade(_GENEROUS)
        arms: dict[str, QueryClassifier | None] = {
            "router off (Concept 5 cascade as drawn)": None,
            "router on (lexical)": lexical,
            "router on (prototype, MiniLM)": prototype,
        }
        tallies: list[StaleAnswerTally] = []
        for arm, classifier in arms.items():
            contexts: list[str] = []
            for query in _FRESHNESS_QUERIES:
                await begin_turn()
                embedding = embedder.embed(query)
                decision = (
                    None if classifier is None
                    else decide(await classifier.score(query, embedding))
                )
                result = await generous.run(request(query, embedding), decision)
                contexts.append("\n".join(item.content for item in result.items))
            tallies.append(
                tally_staleness(arm, contexts, "within thirty days", "forty-five days")
            )

        # Part 3a: dynamic reallocation vs. static base slices in a small window.
        dynamic_dropped = static_dropped = dynamic_tokens = static_tokens = 0
        for question in questions:
            await begin_turn()
            embedding = embedder.embed(question)
            decision = decide(await prototype.score(question, embedding))
            result = await generous.run(request(question, embedding), decision)
            dynamic = assemble_context(
                result.items, allocate(_SMALL_WINDOW, result.contributing)
            )
            static = assemble_context(result.items, allocate(_SMALL_WINDOW, PARADIGM_ORDER))
            dynamic_dropped += sum(dynamic.dropped.values())
            static_dropped += sum(static.dropped.values())
            dynamic_tokens += sum(dynamic.tokens_used.values())
            static_tokens += sum(static.tokens_used.values())
        allocator = AllocatorTally(
            _SMALL_WINDOW, len(questions), dynamic_dropped, static_dropped,
            dynamic_tokens, static_tokens,
        )

        cascade_notes = (
            f"Corpus: {len(_DOCUMENTS)} documents in Qdrant ({sum(1 for _, f in _DOCUMENTS if f)} "
            f"also frozen into a distilgpt2 HFFrozenCache on CPU, the return policy in its "
            f"superseded thirty-day version), {len(_FACTS)} MAG semantic facts in Postgres. "
            f"Thresholds carried over unchanged from the integration corpus: CAG hit "
            f"{CAG_HIT}, partial {CAG_PARTIAL}; MAG hit {MAG_HIT}, partial {MAG_PARTIAL}. "
            f"Tier latency: {len(questions)} questions x {_LATENCY_REPEATS} repeats, prototype "
            f"routing, default TierTimeouts, after one untimed warm-up. Query embedding "
            f"(MiniLM, CPU) is paid before any tier runs: p50 "
            f"{embed_ms[len(embed_ms) // 2]:.2f}ms, max {embed_ms[-1]:.2f}ms."
        )
        _CASCADE_REPORT.write_text(
            render_cascade_measurements(tiers, tallies, allocator, cascade_notes),
            encoding="utf-8",
        )
        print(_CASCADE_REPORT.read_text(encoding="utf-8"))

        # Part 3b: self-versus-self comparison with real generation and judging.
        unified = UnifiedAnswerQuestion(
            embedder, prototype, generous, chat_model,
            budget_recorder=PostgresSessionBudgetRecorder(db_session),
        )
        baseline_answerer = AnswerQuestion(search_documents=search, chat_model=chat_model, top_k=3)
        routes: dict[str, str] = {}

        async def baseline(question: str) -> Answer:
            await begin_turn()
            result = await baseline_answerer.execute(tenant_id=tenant_id, question=question)
            return Answer(
                text=result.answer,
                input_tokens=chat_model.last_input_tokens,
                output_tokens=chat_model.last_output_tokens,
                context="\n\n".join(source.content for source in result.sources),
            )

        async def treatment(question: str) -> Answer:
            await begin_turn()
            result = await unified.execute(tenant_id, user_id, session_id, question)
            await db_session.commit()
            route = sorted(p.value for p in result.decision.paradigms) if result.decision else []
            routes[question] = (
                f"routed {route}, attempts "
                f"{[(a.paradigm.value, a.outcome.value) for a in result.attempts]}"
            )
            return Answer(
                text=result.answer,
                input_tokens=chat_model.last_input_tokens,
                output_tokens=chat_model.last_output_tokens,
                context="\n\n".join(source.content for source in result.sources),
            )

        comparison = await RunComparison(
            judge=OllamaJudge(client=ollama.AsyncClient(), model_id=_MODEL_ID), repeat_count=5
        ).execute(
            scenario_name=scenario.name,
            model_config=f"{_MODEL_ID}, Ollama",
            success_criterion="see evaluation/scenarios/orchestration-meta-layer/queries.yaml",
            rag=True, cag=True, mag=True,
            questions=questions,
            baseline=baseline,
            treatment=treatment,
            success_check=lambda question, answer: checks[question](answer.text.lower()),
            reference_contexts=[q.gold_passage for q in scenario.questions],
            notes=(
                "Baseline = RAG-only AnswerQuestion (top_k=3). Treatment = "
                "UnifiedAnswerQuestion: prototype routing, the CAG/MAG/RAG cascade with "
                "generous timeouts, the 128K budget allocator, and a real session budget "
                "record. CAVEAT 1: judge and generator are both qwen3.5 (self-grading "
                "risk, as in every earlier batch). CAVEAT 2: CAG is a CPU distilgpt2 proxy "
                "whose lookup, not generation speed, is what the treatment exercises. "
                "CAVEAT 3: question 1 asks for the return window without any freshness "
                "cue; a CAG hit on the superseded policy there is a Sync Mixer failure the "
                "router is not designed to catch, and is reported, not hidden."
            ),
        )
        comparison = dataclasses.replace(
            comparison,
            notes=comparison.notes + " ROUTES (last repeat): "
            + "; ".join(f"{q!r}: {r}" for q, r in routes.items()),
        )
        _COMPARISON_REPORT.write_text(render(comparison), encoding="utf-8")
        print(_COMPARISON_REPORT.read_text(encoding="utf-8"))
    await engine.dispose()


def main() -> None:
    # Migrations run before the event loop starts: alembic's env drives its own
    # async engine with asyncio.run, which cannot nest inside a running loop.
    with (
        PostgresContainer("pgvector/pgvector:pg16") as postgres,
        QdrantContainer("qdrant/qdrant:v1.16.2") as qdrant,
    ):
        database_url = postgres.get_connection_url().replace(
            "postgresql+psycopg2", "postgresql+asyncpg"
        )
        os.environ["DATABASE_URL"] = database_url
        os.environ["APP_DB_PASSWORD"] = _APP_DB_PASSWORD
        command.upgrade(Config("alembic.ini"), "head")
        app_url = (
            make_url(database_url)
            .set(username="app_user", password=_APP_DB_PASSWORD)
            .render_as_string(hide_password=False)
        )
        asyncio.run(_measure(app_url, f"http://127.0.0.1:{qdrant.get_exposed_port(6333)}"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run it and commit the generated reports**

Run: `PYTHONPATH=. ../../../.venv/Scripts/python.exe evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py`
Expected: both reports print and are written. A failure here is diagnosed with `superpowers:systematic-debugging`, never worked around by loosening a check.

```bash
git add evaluation/scenarios/orchestration-meta-layer/queries.yaml evaluation/scenarios/orchestration-meta-layer/run_cascade_comparison.py evaluation/reports/orchestration-meta-layer-cascade.md evaluation/reports/orchestration-meta-layer-comparison.md
git commit  # test: measure tier latency, the router ablation, and unified vs RAG-only answers
```

- [ ] **Step 8: Write the narrative report `evaluation/reports/orchestration-meta-layer.md`**

Follow the connected-prose standard in `docs/governance/WRITING_STANDARDS.md` and the shape of `evaluation/reports/rag-cag-synthesis.md`. One section per spec claim, each citing the generated report it draws from:

1. **Does routing keep superseded cache text out of freshness answers?** Stale rates per arm from the cascade report, plus the classifiers' accuracy and misroutes from the router report.
2. **Are the tier budgets achievable here?** Per-tier p50/p95 and within-budget rate against 10/50/2000ms, with query embedding reported as its own cost.
3. **Does classification cost less than it saves?** Router latency per classifier set against CAG's measured tier latency.
4. **What did the allocator change?** Dropped items and tokens used, dynamic vs. static, at the 1,000-token window; plainly stated as no effect at 128K on this corpus if that is what the numbers show.
5. **Unified vs. RAG-only answers.** Task success and judge scores from the comparison report, including question 1's result.
6. **What these numbers do not show.** The CPU cache proxy, the single-author label review, self-judging, and the in-process background update.

Commit:

```bash
git add evaluation/reports/orchestration-meta-layer.md
git commit  # docs: report the orchestration meta-layer's measured behavior
```

---

### Task 12: Documentation, review, and integration

**Files:**
- Modify: `docs/architecture/OVERVIEW.md`, `docs/testing/TESTING.md`, `docs/database/DATABASE.md`, `docs/architecture/CONTEXT_GRAPH.md`, `docs/superpowers/specs/2026-09-13-orchestration-meta-layer-design.md` (scope line: issue numbers), `CLAUDE.md` (only through `claude-md-sync` proposals)

- [ ] **Step 1: Full verification before touching docs**

Run, in order, and read every result:
- `../../../.venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider` — all pass; note the new count.
- `../../../.venv/Scripts/python.exe -m pytest tests/integration -q -p no:cacheprovider` — all pass except the documented skips; note counts.
- `../../../.venv/Scripts/python.exe -m ruff check src tests evaluation` — clean.
- `../../../.venv/Scripts/python.exe -m mypy src` — clean.

- [ ] **Step 2: Update the architecture, testing, and database docs**

- `docs/architecture/OVERVIEW.md`: under "Paradigm Router", "Context Budget Allocator", and "Latency-Adaptive Fallback Cascade", add what is built (module paths, the routing decision rule, the reallocation rules as implemented, the three cascade modes) and cite `evaluation/reports/orchestration-meta-layer.md` for measured numbers. Replace the cascade section's "None of these timeout numbers ... measured against a running cascade yet" sentence with the measured tier latencies against 10 / 50 / 2000ms. In "The Phase 1 module blueprint", move the router, allocator, and cascade from unbuilt to built, leaving `freshness_router.py`, workers, frontend, and Kubernetes as what remains.
- `docs/testing/TESTING.md`: refresh the unit/integration file counts, record the Ollama-dependent skips beside the vLLM ones, and replace "`scripts/benchmark.py` still doesn't exist" reasoning about cascade budgets with a pointer to the measured report (the script itself still doesn't exist; say so).
- `docs/database/DATABASE.md`: describe `sessions.context_budget`'s JSON shape (`total`, `slices`, `contributing`, `recorded_at`), that it holds the latest turn only, and that `PostgresSessionBudgetRecorder` is its first writer under RLS.

- [ ] **Step 3: Regenerate the context graph**

Invoke the `graphify` skill to regenerate `docs/architecture/CONTEXT_GRAPH.md`, so the router, allocator, and cascade become real Module/Class/Test File nodes and only the Freshness-Aware Data Router, workers, and frontend remain dashed.

- [ ] **Step 4: Sync CLAUDE.md**

Invoke `claude-md-sync`. Apply each proposed edit it grounds in this batch's git history (test counts, the built/unbuilt list), and state in the commit message that the user granted full control for this session.

- [ ] **Step 5: Commit the docs**

```bash
git add docs CLAUDE.md
git commit  # docs: record the built router, allocator, and cascade and their measured behavior
```

- [ ] **Step 6: Review**

Invoke `superpowers:requesting-code-review` over `develop..HEAD`. Fix every confirmed finding test-first, re-run Step 1, and commit each fix separately.

- [ ] **Step 7: Integrate**

Invoke `superpowers:finishing-a-development-branch` and take the standing path: merge to `develop` locally with a `merge:` commit, re-run the unit suite on the merged result, remove the worktree, and delete the branch. Close this batch's GitHub issues with a comment naming the merge commit and the report.

---

## Execution notes

Where executing this plan against the real stack changed what the plan says, the change was kept and recorded here rather than bent back to match the text above.

- **Partial thresholds (Task 9).** The plan set each partial threshold to the must-miss score minus 0.05. Against the measured MiniLM scores, that would have turned a clearly unrelated query into a PARTIAL match and put the superseded policy into its context. Each partial threshold instead sits midway between the must-miss score and the hit threshold: CAG hit 0.43 / partial 0.35, MAG hit 0.42 / partial 0.37. The measured scores are recorded in `tests/integration/orchestration_env.py`.
- **Session recovery after a cancelled MAG query (Task 9).** The plan's integration test expected `rollback()` to recover a session whose MAG query the cascade cancelled mid-flight. It does not. SQLAlchemy treats the `CancelledError` as a disconnect and terminates the asyncpg connection, after which both `rollback()` and `close()` raise `InterfaceError`. A probe of four recovery paths showed that `invalidate()` recovers cleanly. The test asserts that contract, `MagTier` documents it, and the cascade runner invalidates before every turn.
- **The budget recorder owns its unit of work (Task 7).** The same termination broke a recorder flushing through the MAG tier's session, failing the whole request after a MAG timeout. `PostgresSessionBudgetRecorder` now takes a sessionmaker and commits its own short transaction, and an integration test pins the MAG-timeout case.
- **Held-out labels (Task 10).** Hand review kept 36 of qwen3.5's 40 generated labels and relabeled four MAG-only requests as MAG+RAG, because answering them also needs current catalog or stock data. Each correction carries a `review_note`.
- **Scenario packages (Task 10).** No `__init__.py` was added under `evaluation/scenarios/orchestration-meta-layer/`, matching the other hyphenated scenario directories.
- **Lint scope.** `ruff check tests` reports pre-existing line-length violations in files this batch never touched, so lint was run on every file this batch created or changed. `ruff check src` and `mypy src` are clean across the whole tree.

### Changes from the pre-merge code review

A review of `develop..HEAD` found no critical issues and seven important ones. Each was checked against the code before being changed, and each change went in test-first.

- **The MAG tier owns its unit of work.** `MagTier` now depends on a `SemanticFactSearch` port. Its implementation, `SessionScopedSemanticFactSearch`, opens a session per search and invalidates it on failure or cancellation. The earlier "call `invalidate()` after a MAG timeout" obligation on callers is gone. The review confirmed the risk concretely: `BM25KeywordSearch`, and so hybrid RAG, reads Postgres, so a shared session broke the MAG → RAG fallback, and in PARALLEL mode two tiers would have used one `AsyncSession` concurrently.
- **Classification failure degrades instead of failing the request.** `UnifiedAnswerQuestion` classifies under `classifier_timeout`. A timeout, an exception, or a `ClassificationFailed` routes with `fallback_decision()` (every paradigm, PARALLEL), and `routing_fallback` records which case occurred. `LlmQueryClassifier` raises `ClassificationFailed` instead of returning 0.5 everywhere, a value that only lands in the uncertainty band at the default threshold.
- **No-signal scores run every paradigm in parallel.** `decide` used to pick the highest score when nothing cleared the threshold, which routed no-signal queries confidently to CAG alone. It now returns `fallback_decision()`.
- **Background RAG completion is bounded.** `background_timeout` gives each completion a deadline, and `max_background` caps concurrent completions; a timed-out attempt beyond the cap is cancelled.
- **Warmed-cache matching is thread-safe.** `CagTier` runs `best_warmed_match` on a worker thread. Warmed entries are now indexed by tenant and scanned from a `dict.copy()` snapshot, and the method returns a `SearchResult` whose `score` carries the similarity.
- **Budget writes are user-scoped.** `SessionBudgetRecorder.record` takes `user_id`, and the UPDATE matches it. An integration test pins that another user in the same tenant is refused.
- **Budget recorded before generation.** A missing session now fails before an answer is paid for.
- **Minor fixes:**
  - A failing access tracker or findings sink no longer fails a request.
  - PARALLEL mode cancels sibling tiers when one raises `CancelledError`.
  - The measured thresholds moved to `evaluation/scenarios/orchestration_meta_layer_thresholds.py`, so the runner no longer imports test code.
  - `stale_rate` is NaN over zero runs.
  - Budget-share tolerance is `abs_tol=1e-12` with `rel_tol=0.0`. `isclose`'s default relative tolerance had silently allowed a 5e-10 overshoot, and the new test caught it.
- **The integration Postgres URL.** After the review fixes, one test still failed. A MAG search cancelled by the cascade was followed by a fresh search that hung until its 5s timeout. A probe ruled out the cascade and SQLAlchemy: raw asyncpg connections to the TestContainers Postgres via `localhost` timed out after 20s even before any cancellation, while `127.0.0.1` connected in about 40ms, including immediately after a cancel. Pooled connections had been hiding it; an invalidated connection forces the next request to open a new one. The `database_url` fixture now pins `127.0.0.1`, as `redis_url`, `qdrant_url`, and `neo4j_url` already did, and the cascade runner does the same. The three affected test files went from 535s to 23s.

### What running the measurements changed

- **The CAG tier missed its budget because another tier blocked the event loop, not because of its own work.** The first cascade run put CAG inside its 10ms budget only 29% of the time (p50 10.76ms; 32 of 45 attempts timed out). Measured alone, a CAG attempt is p50 0.40ms. Two hypotheses were tested with probes against the real `LatencyCascade`, `CagTier`, and MiniLM:
  - *Rejected:* contention left over from the query embedding computed just before the cascade. A CAG attempt right after an embedding is still p50 0.68ms.
  - *Confirmed:* in PARALLEL routes, `SearchDocuments` re-embeds the question synchronously on the event loop, delaying delivery of the CAG thread's result past its timeout. That produced 15 of 60 CAG timeouts at p50 9.77ms, against none at p50 0.86ms with the embedding on a worker thread.

  The fix is `CachingEmbeddingModel`, shared by `UnifiedAnswerQuestion` and the retriever, plus a documented contract that a tier must never block the loop. Re-run on the fixed composition, CAG met its budget in 45 of 45 attempts (p50 2.00ms, no timeouts).
- **The allocator comparison needed smaller windows.** At a 1,000-token window every slice had room to spare, so dynamic and static allocation dropped nothing either way — a null result that measured nothing. The runner now sweeps 1,000, 400, 250, and 150 tokens:
  - No difference at 1,000 or 400.
  - At 250, static slices dropped 8 items where reallocation dropped none.
  - At 150, 14 dropped against 6.
- **An oracle-routed comparison arm was added.** The first prototype-routed comparison showed the unified pipeline barely beating RAG-only (80% vs. 76% task success). The router comparison had already measured the prototype classifier's MAG recall at 63%, and here it never routed the corpus's three personal questions to MAG at all. A treatment routed by per-question labels in `queries.yaml` separates what the pipeline adds from what the classifier loses.
- **The runner writes UTF-8 to the console.** The comparison report's Δ character crashed a run under Windows cp1252 after both reports were written, skipping `engine.dispose()`.
- **The shared embedding cache is pinned by an integration test.** `tests/integration/orchestration_env.py` now composes one `CachingEmbeddingModel` for the use case and the retriever, the way the runner does, and a PARALLEL-route test against real MiniLM asserts that the retriever's embed of the question is a cache hit. With the retriever on the raw model, the test fails with zero hits.
