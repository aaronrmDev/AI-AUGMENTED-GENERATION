import itertools

import pytest

from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import BudgetShares, Paradigm

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


def _slices(allocation):
    return (allocation.cag, allocation.mag, allocation.rag, allocation.query, allocation.reserve)


def test_every_paradigm_contributing_gets_the_source_128k_split():
    assert _slices(allocate(128_000, {CAG, MAG, RAG})) == (51_200, 32_000, 25_600, 12_800, 6_400)


def test_an_idle_rag_slice_expands_mag():
    assert _slices(allocate(128_000, {CAG, MAG})) == (51_200, 57_600, 0, 12_800, 6_400)


def test_an_idle_mag_slice_splits_between_cag_and_rag_by_base_share():
    # 32_000 * 0.4/0.6 = 21_333.3 -> 21_333; 32_000 * 0.2/0.6 = 10_666.6 -> 10_666;
    # the one leftover token goes to reserve.
    assert _slices(allocate(128_000, {CAG, RAG})) == (72_533, 0, 36_266, 12_800, 6_401)


def test_a_cag_miss_expands_rag():
    assert _slices(allocate(128_000, {MAG, RAG})) == (0, 32_000, 76_800, 12_800, 6_400)


def test_cag_alone_absorbs_both_idle_slices():
    assert _slices(allocate(128_000, {CAG})) == (108_800, 0, 0, 12_800, 6_400)


def test_mag_alone_absorbs_both_idle_slices():
    assert _slices(allocate(128_000, {MAG})) == (0, 108_800, 0, 12_800, 6_400)


def test_rag_alone_absorbs_both_idle_slices():
    assert _slices(allocate(128_000, {RAG})) == (0, 0, 108_800, 12_800, 6_400)


def test_nothing_contributing_moves_every_paradigm_slice_into_reserve():
    assert _slices(allocate(128_000, set())) == (0, 0, 0, 12_800, 115_200)


_SUBSETS = [
    set(combo) for size in range(4) for combo in itertools.combinations((CAG, MAG, RAG), size)
]


@pytest.mark.parametrize(
    "total", [0, 1, 7, 99, 1_000, 4_096, 128_000, 200_003, 10**9, 10**12]
)
@pytest.mark.parametrize("contributing", _SUBSETS)
def test_the_five_slices_always_sum_to_the_total_and_are_never_negative(total, contributing):
    allocation = allocate(total, contributing)
    assert allocation.total == total
    assert min(_slices(allocation)) >= 0


def test_the_order_contributing_arrives_in_does_not_change_the_allocation():
    assert allocate(128_000, [RAG, CAG]) == allocate(128_000, [CAG, RAG])


def test_custom_shares_are_honored():
    shares = BudgetShares(cag=0.5, mag=0.2, rag=0.2, query=0.05, reserve=0.05)
    assert _slices(allocate(1_000, {CAG, MAG, RAG}, shares)) == (500, 200, 200, 50, 50)


def test_a_negative_total_is_rejected():
    with pytest.raises(ValueError):
        allocate(-1, {CAG})
