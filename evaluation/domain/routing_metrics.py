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
