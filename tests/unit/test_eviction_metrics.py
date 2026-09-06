import pytest

from src.cag.domain.eviction_metrics import (
    accumulate_attention_scores,
    memory_reduction_ratio,
    retained_attention_mass,
)


def test_accumulate_attention_scores_sums_columns_of_a_causal_matrix():
    # Causal: row i only attends to positions 0..i (later columns are 0).
    matrix = [
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.2, 0.3, 0.5],
    ]
    # column 0: 1.0 + 0.5 + 0.2 = 1.7
    # column 1: 0.0 + 0.5 + 0.3 = 0.8
    # column 2: 0.0 + 0.0 + 0.5 = 0.5
    assert accumulate_attention_scores(matrix) == pytest.approx([1.7, 0.8, 0.5])


def test_accumulate_attention_scores_rejects_empty_matrix():
    with pytest.raises(ValueError):
        accumulate_attention_scores([])


def test_accumulate_attention_scores_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        accumulate_attention_scores([[1.0, 0.0], [0.5, 0.5], [0.2, 0.3]])


def test_memory_reduction_ratio_two_x_when_half_the_tokens_survive():
    assert memory_reduction_ratio(original_token_count=100, kept_token_count=50) == pytest.approx(
        2.0
    )


def test_memory_reduction_ratio_one_x_when_nothing_is_evicted():
    assert memory_reduction_ratio(
        original_token_count=100, kept_token_count=100
    ) == pytest.approx(1.0)


def test_memory_reduction_ratio_rejects_zero_kept_tokens():
    with pytest.raises(ValueError):
        memory_reduction_ratio(original_token_count=100, kept_token_count=0)


def test_memory_reduction_ratio_rejects_kept_exceeding_original():
    with pytest.raises(ValueError):
        memory_reduction_ratio(original_token_count=10, kept_token_count=20)


def test_retained_attention_mass_is_one_when_everything_is_kept():
    scores = [1.0, 2.0, 3.0]
    assert retained_attention_mass(scores, [0, 1, 2]) == pytest.approx(1.0)


def test_retained_attention_mass_reflects_the_kept_fraction():
    scores = [1.0, 2.0, 3.0, 4.0]
    # total = 10.0, keeping indices 2 and 3 (3.0 + 4.0) -> 0.7
    assert retained_attention_mass(scores, [2, 3]) == pytest.approx(0.7)


def test_retained_attention_mass_is_zero_when_nothing_is_kept():
    scores = [1.0, 2.0, 3.0]
    assert retained_attention_mass(scores, []) == pytest.approx(0.0)


def test_retained_attention_mass_rejects_empty_scores():
    with pytest.raises(ValueError):
        retained_attention_mass([], [0])
