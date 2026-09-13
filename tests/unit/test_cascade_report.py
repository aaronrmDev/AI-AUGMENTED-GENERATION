from evaluation.domain.cascade_metrics import (
    AllocatorTally,
    StaleAnswerTally,
    TierLatencySummary,
)
from evaluation.infrastructure.cascade_report import render_cascade_measurements
from src.orchestration.domain.entities import Paradigm, TierOutcome


def test_the_report_renders_tier_latency_staleness_and_allocator_sections():
    report = render_cascade_measurements(
        [
            TierLatencySummary(
                Paradigm.CAG, 4, {TierOutcome.HIT: 3, TierOutcome.MISS: 1}, 1.25, 3.5, 10.0, 1.0
            )
        ],
        [StaleAnswerTally("router off", 6, 6, 0, 0)],
        AllocatorTally(
            1_000, 10, dynamic_dropped=1, static_dropped=7, dynamic_tokens=900, static_tokens=400
        ),
        notes="measured here",
    )
    assert "measured here" in report
    assert "| CAG | 4 | hit 3, miss 1 | 1.25 | 3.50 | 10 | 100% |" in report
    assert "| router off | 6 | 6 | 0 | 0 | 100% |" in report
    assert "| 1000 | 10 | 1 | 7 | 900 | 400 |" in report
