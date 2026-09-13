import math

import pytest

from evaluation.domain.routing_metrics import RoutingObservation, summarize
from evaluation.domain.statistics import nearest_rank_percentile
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


def test_nearest_rank_percentile_matches_run_comparisons_formula():
    assert nearest_rank_percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 3.0
    assert nearest_rank_percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    with pytest.raises(ValueError):
        nearest_rank_percentile([], 0.5)
