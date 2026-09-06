import pytest

from src.cag.infrastructure.nacl_evictor import NACLEvictor


def test_zero_random_fraction_is_plain_top_scoring_selection():
    scores = [10.0, 8.0, 6.0, 4.0, 2.0]
    decision = NACLEvictor(random_fraction=0.0).select_keep_indices(scores, budget=3)
    assert decision.keep_indices == [0, 1, 2]
    assert decision.evicted_count == 2


def test_keeps_everything_when_budget_covers_the_whole_sequence():
    scores = [1.0, 2.0, 3.0]
    decision = NACLEvictor(random_fraction=0.5).select_keep_indices(scores, budget=3)
    assert decision.keep_indices == [0, 1, 2]
    assert decision.evicted_count == 0


def test_random_fraction_still_guarantees_the_deterministic_top_scorers_survive():
    scores = [float(v) for v in range(20, 0, -1)]  # strictly descending: index i has score 20-i
    decision = NACLEvictor(random_fraction=0.4, random_seed=42).select_keep_indices(
        scores, budget=10
    )
    # deterministic_budget = 10 - round(10 * 0.4) = 6 -> indices 0..5 (top 6 scorers)
    # must always be present regardless of the random slice.
    assert set(range(6)).issubset(set(decision.keep_indices))
    assert len(decision.keep_indices) == 10
    assert decision.evicted_count == 10


def test_same_seed_produces_the_same_random_eviction():
    scores = [float(v) for v in range(20, 0, -1)]
    first = NACLEvictor(random_fraction=0.4, random_seed=7).select_keep_indices(
        scores, budget=10
    )
    second = NACLEvictor(random_fraction=0.4, random_seed=7).select_keep_indices(
        scores, budget=10
    )
    assert first.keep_indices == second.keep_indices


def test_rejects_random_fraction_above_one():
    with pytest.raises(ValueError):
        NACLEvictor(random_fraction=1.5)


def test_rejects_negative_random_fraction():
    with pytest.raises(ValueError):
        NACLEvictor(random_fraction=-0.1)


def test_rejects_empty_scores():
    with pytest.raises(ValueError):
        NACLEvictor(random_fraction=0.0).select_keep_indices([], budget=1)
