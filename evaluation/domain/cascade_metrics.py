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
        counts = {outcome: sum(1 for a in mine if a.outcome is outcome) for outcome in TierOutcome}
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
    flags = [(stale_marker in context, fresh_marker in context) for context in contexts]
    return StaleAnswerTally(
        arm=arm,
        runs=len(contexts),
        stale=sum(1 for stale, fresh in flags if stale and not fresh),
        fresh=sum(1 for stale, fresh in flags if fresh and not stale),
        mixed=sum(1 for stale, fresh in flags if stale and fresh),
    )


@dataclass(frozen=True)
class AllocatorTally:
    window_tokens: int
    turns: int
    dynamic_dropped: int
    static_dropped: int
    dynamic_tokens: int
    static_tokens: int
