import pytest

from src.cag.domain.combination_metrics import (
    best_single_reduction,
    compounded_reduction,
    interference_drift,
    synergy_score,
)


def test_compounded_reduction_multiplies_every_stage():
    assert compounded_reduction([4.0, 4.0, 2.0]) == pytest.approx(32.0)


def test_best_single_reduction_is_the_strongest_stage_alone():
    assert best_single_reduction([4.0, 4.0, 2.0]) == pytest.approx(4.0)


def test_synergy_is_one_when_the_full_product_was_achieved():
    assert synergy_score(32.0, [4.0, 4.0, 2.0]) == pytest.approx(1.0)


def test_synergy_is_zero_when_the_combination_did_no_better_than_its_best_part():
    # The "merely compatible" null hypothesis: three techniques coexist
    # without reinforcing, so the result is whichever was strongest.
    assert synergy_score(4.0, [4.0, 4.0, 2.0]) == pytest.approx(0.0)


def test_synergy_is_fractional_in_between():
    # Halfway between 4.0 (best single) and 32.0 (full product) is 18.0.
    assert synergy_score(18.0, [4.0, 4.0, 2.0]) == pytest.approx(0.5)


def test_synergy_can_report_a_combination_that_actively_hurt():
    # Worse than the best single technique alone -- a real possible
    # outcome that a pass/fail check would have hidden.
    assert synergy_score(2.0, [4.0, 4.0, 2.0]) < 0.0


def test_synergy_of_a_single_stage_pipeline_is_defined_not_a_division_error():
    # One stage: floor and ceiling coincide, so there is no range to
    # place the result within.
    assert synergy_score(4.0, [4.0]) == pytest.approx(1.0)


def test_metrics_reject_a_non_positive_stage_reduction():
    with pytest.raises(ValueError):
        compounded_reduction([4.0, 0.0])


def test_metrics_reject_an_empty_pipeline():
    with pytest.raises(ValueError):
        compounded_reduction([])
    with pytest.raises(ValueError):
        best_single_reduction([])


def test_interference_drift_is_zero_when_a_stage_is_unaffected_by_the_one_before_it():
    # Measured behaviour of quantization after eviction: identical ratio
    # on the survivors as on the full cache.
    assert interference_drift(
        standalone_reduction=5.33, in_pipeline_reduction=5.33
    ) == pytest.approx(0.0)


def test_interference_drift_is_negative_when_the_earlier_stage_made_things_harder():
    # Measured behaviour of low-rank compression after aggressive
    # eviction: 7.53x alone, 5.33x on what eviction left.
    assert interference_drift(
        standalone_reduction=7.53, in_pipeline_reduction=5.33
    ) == pytest.approx(-0.292, abs=0.001)


def test_interference_drift_can_be_positive_when_stages_genuinely_help_each_other():
    assert interference_drift(
        standalone_reduction=4.0, in_pipeline_reduction=5.0
    ) == pytest.approx(0.25)


def test_interference_drift_rejects_non_positive_reductions():
    with pytest.raises(ValueError):
        interference_drift(standalone_reduction=0.0, in_pipeline_reduction=4.0)
