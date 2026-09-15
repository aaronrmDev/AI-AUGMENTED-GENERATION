import pytest

from src.cag.domain.session_metrics import (
    amortized_tokens_per_turn,
    is_flat_per_turn,
    recompute_ratio,
)


def test_amortized_cost_is_the_per_turn_average():
    assert amortized_tokens_per_turn(tokens_recomputed=500, num_turns=10) == pytest.approx(50.0)


def test_amortized_cost_rejects_a_session_with_no_turns():
    with pytest.raises(ValueError):
        amortized_tokens_per_turn(tokens_recomputed=100, num_turns=0)


def test_recompute_ratio_is_one_when_nothing_was_saved():
    assert recompute_ratio(strategy_tokens=800, naive_tokens=800) == pytest.approx(1.0)


def test_recompute_ratio_reports_the_saved_fraction():
    assert recompute_ratio(strategy_tokens=200, naive_tokens=800) == pytest.approx(0.25)


def test_recompute_ratio_rejects_a_non_positive_baseline():
    with pytest.raises(ValueError):
        recompute_ratio(strategy_tokens=10, naive_tokens=0)


def test_flatness_accepts_a_constant_per_turn_cost():
    assert is_flat_per_turn([50, 50, 50, 50]) is True


def test_flatness_rejects_a_cost_that_grows_with_the_session():
    # The shape the whole technique exists to escape.
    assert is_flat_per_turn([50, 100, 150, 200]) is False


def test_flatness_distinguishes_a_halved_quadratic_from_a_flat_line():
    # Halving every term of a growing series leaves it growing. A total
    # would call this an improvement; the structural question does not.
    assert is_flat_per_turn([25, 50, 75, 100]) is False


def test_flatness_allows_a_stated_tolerance_for_near_constant_cost():
    assert is_flat_per_turn([50, 52, 51], tolerance=0.1) is True
    assert is_flat_per_turn([50, 80, 51], tolerance=0.1) is False


def test_flatness_rejects_an_empty_session():
    with pytest.raises(ValueError):
        is_flat_per_turn([])
