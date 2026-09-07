import pytest

from src.cag.domain.offloading_metrics import (
    effective_context_multiplier,
    latency_cost_ratio,
    overlap_efficiency,
)


def test_overlap_efficiency_is_one_when_every_transfer_hid_behind_compute():
    assert overlap_efficiency(transfer_ms_issued=40.0, transfer_ms_hidden=40.0) == pytest.approx(
        1.0
    )


def test_overlap_efficiency_is_zero_when_the_pipeline_stalled_for_all_of_it():
    assert overlap_efficiency(transfer_ms_issued=40.0, transfer_ms_hidden=0.0) == pytest.approx(
        0.0
    )


def test_overlap_efficiency_is_the_hidden_fraction():
    assert overlap_efficiency(transfer_ms_issued=40.0, transfer_ms_hidden=30.0) == pytest.approx(
        0.75
    )


def test_overlap_efficiency_of_a_run_that_moved_nothing_is_one_not_a_division_error():
    # Nothing was issued, so nothing failed to hide.
    assert overlap_efficiency(transfer_ms_issued=0.0, transfer_ms_hidden=0.0) == pytest.approx(1.0)


def test_overlap_efficiency_rejects_hiding_more_than_was_issued():
    with pytest.raises(ValueError):
        overlap_efficiency(transfer_ms_issued=10.0, transfer_ms_hidden=20.0)


def test_effective_context_multiplier_reports_how_far_past_gpu_capacity_a_run_reached():
    assert effective_context_multiplier(gpu_layer_capacity=8, layers_served=32) == pytest.approx(
        4.0
    )


def test_effective_context_multiplier_is_one_when_everything_fit_on_the_gpu():
    assert effective_context_multiplier(gpu_layer_capacity=32, layers_served=32) == pytest.approx(
        1.0
    )


def test_effective_context_multiplier_rejects_a_non_positive_capacity():
    with pytest.raises(ValueError):
        effective_context_multiplier(gpu_layer_capacity=0, layers_served=8)


def test_latency_cost_ratio_is_one_when_offloading_cost_no_time():
    assert latency_cost_ratio(baseline_ms=100.0, offloaded_ms=100.0) == pytest.approx(1.0)


def test_latency_cost_ratio_reports_the_slowdown():
    assert latency_cost_ratio(baseline_ms=100.0, offloaded_ms=250.0) == pytest.approx(2.5)


def test_latency_cost_ratio_rejects_a_non_positive_baseline():
    with pytest.raises(ValueError):
        latency_cost_ratio(baseline_ms=0.0, offloaded_ms=10.0)
