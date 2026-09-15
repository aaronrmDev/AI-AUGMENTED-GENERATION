import math

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
        [
            "within thirty days",
            "within forty-five days",
            "within thirty days / forty-five days",
            "",
        ],
        stale_marker="within thirty days",
        fresh_marker="forty-five days",
    )
    assert (tally.runs, tally.stale, tally.fresh, tally.mixed) == (4, 1, 1, 1)
    assert tally.stale_rate == 0.25


def test_a_stale_rate_over_zero_runs_is_undefined_rather_than_zero():
    # Matches routing_metrics: nan means "never measured", 0.0 would read as "never stale".
    assert math.isnan(tally_staleness("empty", [], "old", "new").stale_rate)
