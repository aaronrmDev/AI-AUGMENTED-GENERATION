import pytest

from src.cag.infrastructure.h2o_evictor import H2OEvictor


def test_keeps_everything_when_budget_covers_the_whole_sequence():
    evictor = H2OEvictor(recent_window=2)
    decision = evictor.select_keep_indices([1.0, 2.0, 3.0], budget=3)
    assert decision.keep_indices == [0, 1, 2]
    assert decision.evicted_count == 0


def test_always_keeps_the_recent_window_regardless_of_score():
    # Recent window = last 2 positions (indices 3, 4), even though they
    # have the lowest scores -- budget=3 gives one heavy-hitter slot,
    # which must go to index 0 (score 10.0), the top scorer outside the
    # recent window.
    scores = [10.0, 1.0, 1.0, 0.1, 0.1]
    evictor = H2OEvictor(recent_window=2)
    decision = evictor.select_keep_indices(scores, budget=3)
    assert decision.keep_indices == [0, 3, 4]
    assert decision.evicted_count == 2


def test_falls_back_to_purely_recent_when_budget_is_tighter_than_the_window():
    scores = [10.0, 1.0, 1.0, 0.1, 0.1]
    evictor = H2OEvictor(recent_window=2)
    decision = evictor.select_keep_indices(scores, budget=1)
    # Only room for 1 token and it's tighter than the recent window --
    # keep the single most recent position.
    assert decision.keep_indices == [4]
    assert decision.evicted_count == 4


def test_rejects_empty_scores():
    with pytest.raises(ValueError):
        H2OEvictor(recent_window=1).select_keep_indices([], budget=1)


def test_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        H2OEvictor(recent_window=1).select_keep_indices([1.0, 2.0], budget=0)


def test_rejects_negative_recent_window():
    with pytest.raises(ValueError):
        H2OEvictor(recent_window=-1)
