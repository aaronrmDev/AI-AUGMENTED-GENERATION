from evaluation.domain.cascade_metrics import (
    AllocatorTally,
    StaleAnswerTally,
    TierLatencySummary,
)
from evaluation.infrastructure.cascade_report import (
    render_cascade_measurements,
    render_retriever_measurements,
)
from src.orchestration.domain.entities import Paradigm, TierOutcome


def test_the_report_renders_tier_latency_staleness_and_allocator_sections():
    report = render_cascade_measurements(
        [
            TierLatencySummary(
                Paradigm.CAG, 4, {TierOutcome.HIT: 3, TierOutcome.MISS: 1}, 1.25, 3.5, 10.0, 1.0
            )
        ],
        [StaleAnswerTally("router off", 6, 6, 0, 0)],
        [
            AllocatorTally(
                1_000,
                10,
                dynamic_dropped=1,
                static_dropped=7,
                dynamic_tokens=900,
                static_tokens=400,
            ),
            AllocatorTally(
                250,
                10,
                dynamic_dropped=4,
                static_dropped=12,
                dynamic_tokens=200,
                static_tokens=150,
            ),
        ],
        notes="measured here",
    )
    assert "measured here" in report
    assert "| CAG | 4 | hit 3, miss 1 | 1.25 | 3.50 | 10 | 100% |" in report
    assert "| router off | 6 | 6 | 0 | 0 | 100% |" in report
    # One row per window in the sweep, so a null result at a roomy window can be
    # read against the windows where slices actually bind.
    assert "| 1000 | 10 | 1 | 7 | 900 | 400 |" in report
    assert "| 250 | 10 | 4 | 12 | 200 | 150 |" in report


def test_an_arm_with_no_runs_renders_its_stale_rate_as_not_applicable():
    report = render_cascade_measurements(
        [],
        [StaleAnswerTally("empty", 0, 0, 0, 0)],
        [AllocatorTally(1_000, 0, 0, 0, 0, 0)],
        notes="",
    )
    assert "| empty | 0 | 0 | 0 | 0 | n/a |" in report


def test_the_retriever_report_renders_every_tier_under_each_retriever():
    cag = TierLatencySummary(Paradigm.CAG, 50, {TierOutcome.HIT: 50}, 1.5, 9.25, 10.0, 0.96)
    rag = TierLatencySummary(Paradigm.RAG, 50, {TierOutcome.HIT: 50}, 40.0, 61.0, 2000.0, 1.0)
    report = render_retriever_measurements(
        [("search", [cag]), ("compression", [cag, rag])], notes="PARALLEL, before the fix"
    )
    assert report.startswith("# Orchestration Meta-Layer — RAG Retrievers in a PARALLEL Route")
    assert "PARALLEL, before the fix" in report
    assert "| Retriever | Tier | Attempts | Outcomes | p50 ms | p95 ms | Budget ms " in report
    assert "| search | CAG | 50 | hit 50 | 1.50 | 9.25 | 10 | 96% |" in report
    assert "| compression | CAG | 50 | hit 50 | 1.50 | 9.25 | 10 | 96% |" in report
    assert "| compression | RAG | 50 | hit 50 | 40.00 | 61.00 | 2000 | 100% |" in report
