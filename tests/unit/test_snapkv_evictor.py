import pytest

from src.cag.infrastructure.snapkv_evictor import SnapKVEvictor


def test_keeps_everything_when_budget_covers_the_whole_sequence():
    evictor = SnapKVEvictor(pool_kernel_size=3, recent_window=1)
    decision = evictor.select_keep_indices([1.0, 2.0, 3.0], budget=3)
    assert decision.keep_indices == [0, 1, 2]
    assert decision.evicted_count == 0


def test_pooling_spreads_importance_to_a_high_scorers_immediate_neighbors():
    # A single sharp spike at index 1; pooling with kernel=3 should make
    # indices 0, 1, 2 all look important (their pooled window includes
    # the spike), beating the flat 0.1 tail at indices 3 and 4 even
    # though those raw scores are the same as index 0's and 2's.
    scores = [0.1, 5.0, 0.1, 0.1, 0.1, 0.1]
    evictor = SnapKVEvictor(pool_kernel_size=3, recent_window=1)
    decision = evictor.select_keep_indices(scores, budget=3)
    # recent window = {5}; two top-up slots go to the pooled-importance
    # neighborhood around the spike (0 and 1, first by stable order).
    assert decision.keep_indices == [0, 1, 5]
    assert decision.evicted_count == 3


def test_kernel_size_one_reduces_to_unpooled_top_scoring_selection():
    scores = [10.0, 1.0, 1.0, 0.1, 0.1]
    evictor = SnapKVEvictor(pool_kernel_size=1, recent_window=2)
    decision = evictor.select_keep_indices(scores, budget=3)
    assert decision.keep_indices == [0, 3, 4]


def test_falls_back_to_purely_recent_when_budget_is_tighter_than_the_window():
    scores = [10.0, 1.0, 1.0, 0.1, 0.1]
    evictor = SnapKVEvictor(pool_kernel_size=3, recent_window=2)
    decision = evictor.select_keep_indices(scores, budget=1)
    assert decision.keep_indices == [4]
    assert decision.evicted_count == 4


def test_rejects_non_positive_pool_kernel_size():
    with pytest.raises(ValueError):
        SnapKVEvictor(pool_kernel_size=0, recent_window=1)


def test_rejects_empty_scores():
    with pytest.raises(ValueError):
        SnapKVEvictor(pool_kernel_size=3, recent_window=1).select_keep_indices([], budget=1)
