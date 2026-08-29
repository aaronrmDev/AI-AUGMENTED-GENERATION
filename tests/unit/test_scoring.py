import math

from src.mag.application.gating._scoring import safe_score


def test_a_positive_finite_score_passes_through_unchanged():
    assert safe_score(0.75) == 0.75


def test_a_negative_finite_score_passes_through_unchanged():
    assert safe_score(-0.75) == -0.75


def test_zero_passes_through_unchanged():
    assert safe_score(0.0) == 0.0


def test_negative_zero_passes_through_unchanged():
    # -0.0 is not NaN, so it must not be substituted -- and it must remain
    # distinguishable from +0.0 (math.copysign is the correct way to tell
    # them apart; == alone treats them as equal).
    result = safe_score(-0.0)
    assert result == 0.0
    assert math.copysign(1.0, result) == -1.0


def test_positive_infinity_passes_through_unchanged():
    assert safe_score(float("inf")) == float("inf")


def test_negative_infinity_passes_through_unchanged():
    assert safe_score(float("-inf")) == float("-inf")


def test_nan_is_replaced_with_negative_infinity():
    assert safe_score(float("nan")) == float("-inf")
