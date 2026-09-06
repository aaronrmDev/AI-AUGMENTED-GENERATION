import pytest

from src.cag.infrastructure.hashevict_evictor import HASHEVICTEvictor


def test_keeps_everything_when_budget_covers_the_whole_sequence():
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    decision = HASHEVICTEvictor(num_hash_bits=4, random_seed=1).select_keep_indices(
        vectors, budget=3
    )
    assert decision.keep_indices == [0, 1, 2]
    assert decision.evicted_count == 0


def test_identical_vectors_collapse_to_their_most_recent_occurrence():
    # idx0 == idx2 and idx1 == idx3 exactly -- identical vectors ALWAYS
    # land in the same SimHash bucket regardless of which random
    # hyperplanes were drawn (the dot product's sign against any
    # hyperplane is identical for identical vectors), so this is
    # deterministic irrespective of num_hash_bits or seed.
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    decision = HASHEVICTEvictor(num_hash_bits=8, random_seed=1).select_keep_indices(
        vectors, budget=3
    )
    assert decision.keep_indices == [2, 3]
    assert decision.evicted_count == 2


def test_further_trimming_keeps_the_most_recent_survivors_when_budget_is_tighter_than_dedup():
    # a and -a are antipodal: for ANY hyperplane, dot(a, h) and
    # dot(-a, h) have opposite sign (barring a measure-zero exact-zero
    # dot product), so their SimHash buckets are exact bitwise
    # complements of each other -- guaranteed distinct regardless of
    # num_hash_bits or seed. Dedup alone leaves 2 survivors (index 1
    # from bucket(a), index 3 from bucket(-a)); a budget of 1 forces a
    # further recency-based trim down to the single most recent.
    a = [1.0, 0.0]
    neg_a = [-1.0, 0.0]
    vectors = [a, a, neg_a, neg_a]
    decision = HASHEVICTEvictor(num_hash_bits=8, random_seed=1).select_keep_indices(
        vectors, budget=1
    )
    assert decision.keep_indices == [3]
    assert decision.evicted_count == 3


def test_rejects_non_positive_hash_bits():
    with pytest.raises(ValueError):
        HASHEVICTEvictor(num_hash_bits=0, random_seed=1)


def test_rejects_empty_vectors():
    with pytest.raises(ValueError):
        HASHEVICTEvictor(num_hash_bits=4, random_seed=1).select_keep_indices([], budget=1)
