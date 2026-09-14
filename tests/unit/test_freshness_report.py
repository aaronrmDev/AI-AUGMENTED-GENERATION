from datetime import UTC, datetime, timedelta

from evaluation.domain.freshness_metrics import MigrationObservation, PlacementTally, RouteRow
from evaluation.infrastructure.freshness_report import render_freshness_measurements


def test_the_report_renders_every_section_with_its_numbers():
    shift = datetime(2026, 1, 11, tzinfo=UTC)
    report = render_freshness_measurements(
        routes=[RouteRow("Stock prices", "seconds", "tenant", "RAG", "rag_only", True)],
        tallies=[PlacementTally("freshness-aware", "prices", 8, 0, 0, 8, 0, 0, 0, 0)],
        preloads={("freshness-aware", "prices"): (0, 0)},
        migrations=[
            MigrationObservation(
                "catalog", "cag_with_rag_backup", "rag_only", shift, shift + timedelta(hours=21)
            )
        ],
        ttl_tallies=[PlacementTally("ttl 0.5", "warranty", 240, 20, 0, 200, 20, 20, 20, 0)],
        notes="Corpus notes.",
    )
    assert report.startswith("# Freshness-Aware Data Router — Measurements")
    assert "Corpus notes." in report
    assert "| Stock prices | seconds | tenant | RAG | rag_only | yes |" in report
    assert "| freshness-aware | prices | 8 | 0 | 0 | 8 | 0 | 0% | 0% | 0 | 0 | 0 |" in report
    assert (
        "| catalog | cag_with_rag_backup → rag_only | 2026-01-11 00:00 | 2026-01-11 21:00 "
        "| 21.0 h |"
    ) in report
    assert "| ttl 0.5 | warranty | 240 | 20 |" in report
