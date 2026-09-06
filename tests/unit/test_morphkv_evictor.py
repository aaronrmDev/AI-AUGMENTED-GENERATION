import pytest

from src.cag.infrastructure.morphkv_evictor import MorphKVEvictor


def test_keeps_everything_when_budget_covers_the_whole_sequence():
    windows = [[1.0, 2.0, 3.0]]
    decision = MorphKVEvictor().select_keep_indices(windows, budget=3)
    assert decision.keep_indices == [0, 1, 2]
    assert decision.evicted_count == 0


def test_sum_max_fusion_picks_the_indices_with_the_highest_combined_score():
    window_0 = [0.1, 0.2, 0.3, 0.4]
    window_1 = [0.4, 0.1, 0.1, 0.4]
    # sum:  [0.5, 0.3, 0.4, 0.8]
    # max:  [0.4, 0.2, 0.3, 0.4]
    # fused:[0.9, 0.5, 0.7, 1.2]
    decision = MorphKVEvictor().select_keep_indices([window_0, window_1], budget=2)
    assert decision.keep_indices == [0, 3]
    assert decision.evicted_count == 2


def test_rejects_empty_windows_list():
    with pytest.raises(ValueError):
        MorphKVEvictor().select_keep_indices([], budget=1)


def test_rejects_windows_of_mismatched_length():
    with pytest.raises(ValueError):
        MorphKVEvictor().select_keep_indices([[1.0, 2.0], [1.0, 2.0, 3.0]], budget=1)


def test_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        MorphKVEvictor().select_keep_indices([[1.0, 2.0]], budget=0)
