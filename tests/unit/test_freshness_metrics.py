import math
from datetime import UTC, datetime, timedelta

from evaluation.domain.freshness_metrics import (
    MigrationObservation,
    PlacementTally,
    PreloadTracker,
    ProbeObservation,
    tally_probes,
)


def _probe(
    context: str, *, cag: bool = False, arm: str = "a", key: str = "k",
    foreign: tuple[str, ...] = (),
) -> ProbeObservation:
    return ProbeObservation(arm, key, context, "v3", ("v1", "v2"), cag, foreign)


def test_contexts_are_classified_by_which_versions_they_hold():
    [tally] = tally_probes(
        [_probe("v1"), _probe("v2 and v3", cag=True), _probe("v3", cag=True), _probe("nothing")]
    )
    assert (tally.probes, tally.stale_only, tally.mixed, tally.fresh_only, tally.neither) == (
        4, 1, 1, 1, 1,
    )
    assert (tally.cag_served, tally.stale_from_cag) == (2, 1)
    assert tally.stale_rate == 0.25
    assert tally.cag_share == 0.5


def test_foreign_text_reaching_a_probe_is_counted():
    [tally] = tally_probes(
        [_probe("v3 size 10", foreign=("size 10",)), _probe("v3", foreign=("size 10",))]
    )
    assert tally.foreign_exposures == 1


def test_tallies_are_grouped_by_arm_and_source_in_first_seen_order():
    tallies = tally_probes([_probe("v3", arm="b"), _probe("v3", arm="a"), _probe("v1", arm="b")])
    assert [(t.arm, t.probes) for t in tallies] == [("b", 2), ("a", 1)]


def test_rates_are_nan_over_zero_probes():
    empty = PlacementTally("a", "k", 0, 0, 0, 0, 0, 0, 0, 0)
    assert math.isnan(empty.stale_rate)
    assert math.isnan(empty.cag_share)


def test_a_preload_counts_as_used_once_any_probe_is_served_from_it():
    tracker = PreloadTracker()
    tracker.preloaded("a", "k")
    tracker.served("a", "k")
    tracker.served("a", "k")
    tracker.preloaded("a", "k")  # replaced before serving anything
    tracker.served("b", "k")  # no preload for this arm: ignored
    assert tracker.counts("a", "k") == (2, 1)
    assert tracker.counts("b", "k") == (0, 0)


def test_migration_lag_is_migration_time_minus_shift_time():
    shift = datetime(2026, 1, 11, tzinfo=UTC)
    observation = MigrationObservation(
        "catalog", "cag_with_rag_backup", "rag_only", shift, shift + timedelta(hours=21)
    )
    assert observation.lag == timedelta(hours=21)
