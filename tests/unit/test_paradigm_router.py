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
