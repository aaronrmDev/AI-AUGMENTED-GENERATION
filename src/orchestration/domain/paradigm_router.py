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


def fallback_decision(scores: Mapping[Paradigm, float] | None = None) -> RoutingDecision:
    """Every paradigm, in PARALLEL: the single route for "the router could not
    decide" -- no score cleared the threshold, or the classifier timed out,
    raised, or reported ClassificationFailed. Running every tier and merging
    cannot drop the paradigm that held the answer; guessing one can."""
    return RoutingDecision(frozenset(PARADIGM_ORDER), RoutingMode.PARALLEL, dict(scores or {}))


def decide(
    scores: Mapping[Paradigm, float],
    select_threshold: float = DEFAULT_SELECT_THRESHOLD,
    uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> RoutingDecision:
    """Concept 1's routing decision over independent per-paradigm scores.

    Multi-label on purpose: the source's own table routes "Compare today's
    sales with last month" to RAG and MAG together. A decision is confident
    only when no score sits within `uncertainty_margin` of the threshold;
    otherwise the paradigms inside that band join the route and it runs
    PARALLEL ("if classifier is uncertain, run parallel and merge"). A
    classification in which nothing clears the threshold has no signal at
    all, which is uncertainty too, so it gets fallback_decision().
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
        # An earlier version picked the highest score here, which sent
        # no-signal queries confidently to CAG alone -- the route where a
        # weak frozen-cache match answers with nothing to contradict it.
        return fallback_decision(scores)
    uncertain = {
        p for p in PARADIGM_ORDER if abs(scores[p] - select_threshold) < uncertainty_margin
    }

    if uncertain:
        return RoutingDecision(frozenset(selected | uncertain), RoutingMode.PARALLEL, dict(scores))
    return RoutingDecision(frozenset(selected), RoutingMode.CASCADE, dict(scores))
