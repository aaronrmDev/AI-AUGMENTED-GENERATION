import pytest

from src.cag.infrastructure.infinipot_distiller import InfiniPotDistiller


def test_returns_the_cache_unchanged_when_budget_covers_the_whole_sequence():
    kv = [[1.0, 2.0], [3.0, 4.0]]
    result = InfiniPotDistiller().distill(kv, budget=2)
    assert result == kv


def test_distills_contiguous_groups_into_their_centroid():
    kv = [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]]
    result = InfiniPotDistiller().distill(kv, budget=2)
    expected = [[1.0, 1.0], [5.0, 5.0]]
    assert len(result) == len(expected)
    for actual_row, expected_row in zip(result, expected, strict=True):
        assert actual_row == pytest.approx(expected_row)


def test_uneven_partition_gives_earlier_groups_the_extra_row():
    kv = [[0.0], [10.0], [20.0]]
    # base_size=1, remainder=1 -> group 0 gets 2 rows (mean of 0, 10),
    # group 1 gets 1 row (just 20).
    result = InfiniPotDistiller().distill(kv, budget=2)
    expected = [[5.0], [20.0]]
    assert len(result) == len(expected)
    for actual_row, expected_row in zip(result, expected, strict=True):
        assert actual_row == pytest.approx(expected_row)


def test_rejects_empty_kv():
    with pytest.raises(ValueError):
        InfiniPotDistiller().distill([], budget=1)


def test_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        InfiniPotDistiller().distill([[1.0]], budget=0)
