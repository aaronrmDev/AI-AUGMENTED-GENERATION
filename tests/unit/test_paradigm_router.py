import math

import pytest

from src.orchestration.domain.entities import Paradigm, RoutingMode
from src.orchestration.domain.paradigm_router import decide, fallback_decision

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


@pytest.mark.parametrize(
    "scores",
    [
        _scores(0.0, 0.0, 0.0),
        _scores(0.2, 0.2, 0.2),
        _scores(0.34, 0.33, 0.33),
        _scores(0.1, 0.3, 0.2),
    ],
)
def test_scores_with_no_signal_run_every_paradigm_in_parallel(scores):
    # "Nothing is relevant" is uncertainty too: guessing a single paradigm here
    # would let a weak frozen-cache match answer on its own.
    decision = decide(scores)
    assert decision.paradigms == frozenset({CAG, MAG, RAG})
    assert decision.mode is RoutingMode.PARALLEL
    assert decision.scores == scores


def test_the_fallback_decision_runs_every_paradigm_in_parallel():
    decision = fallback_decision()
    assert decision.paradigms == frozenset({CAG, MAG, RAG})
    assert decision.mode is RoutingMode.PARALLEL
    assert decision.scores == {}


def test_a_score_exactly_one_margin_from_the_threshold_is_confident():
    decision = decide(_scores(0.75, 0.0, 0.0), select_threshold=0.5, uncertainty_margin=0.25)
    assert decision.paradigms == frozenset({CAG})
    assert decision.mode is RoutingMode.CASCADE


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
