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
