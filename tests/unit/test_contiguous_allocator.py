import pytest

from src.cag.domain.allocation_metrics import external_fragmentation, internal_fragmentation
from src.cag.infrastructure.contiguous_allocator import ContiguousKVAllocator
from src.cag.infrastructure.paged_allocator import PagedKVAllocator


def test_a_sequence_occupies_exactly_its_own_length_with_no_internal_waste():
    allocator = ContiguousKVAllocator(total_slots=100)
    assert allocator.allocate("a", 17) is True
    assert allocator.physical_slots_used() == 17
    assert internal_fragmentation(
        allocator.physical_slots_used(), allocator.logical_tokens_held()
    ) == pytest.approx(0.0)


def test_allocation_fails_despite_enough_total_free_space_when_it_is_scattered():
    # The failure mode paging exists to remove: 40 slots free, but in
    # two separate runs of 20, so a 30-slot request cannot be served.
    allocator = ContiguousKVAllocator(total_slots=100)
    for name, start in (("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)):
        assert allocator.allocate(name, 20) is True, start
    allocator.free("b")
    allocator.free("d")

    free_runs = allocator.free_runs()
    assert sum(free_runs) == 40
    assert max(free_runs) == 20
    assert external_fragmentation(free_runs) == pytest.approx(0.5)
    assert allocator.allocate("late", 30) is False


def test_the_paged_allocator_serves_the_request_that_fragmentation_denied():
    # Head-to-head on the identical workload: same pool size, same
    # sequence sizes, same frees, same final request. This is CAG.md's
    # "no fragmentation since blocks are allocated on demand", made
    # falsifiable rather than asserted.
    paged = PagedKVAllocator(total_blocks=5, block_size=20)
    for name in ("a", "b", "c", "d", "e"):
        assert paged.allocate(name, 20) is True
    paged.free("b")
    paged.free("d")

    assert external_fragmentation(paged.free_runs()) == pytest.approx(0.0)
    assert paged.allocate("late", 30) is True


def test_a_fork_costs_a_full_copy_of_the_parent():
    allocator = ContiguousKVAllocator(total_slots=100)
    allocator.allocate("parent", 30)
    assert allocator.fork("parent", "child") is True
    # No block table to share through, so the child duplicates every slot.
    assert allocator.physical_slots_used() == 60


def test_a_fork_fails_when_there_is_no_room_for_the_copy():
    allocator = ContiguousKVAllocator(total_slots=50)
    allocator.allocate("parent", 30)
    assert allocator.fork("parent", "child") is False


def test_growth_stays_in_place_when_the_slots_after_it_are_free():
    allocator = ContiguousKVAllocator(total_slots=100)
    allocator.allocate("a", 10)
    assert allocator.extend("a", 10) is True
    assert allocator.physical_slots_used() == 20
    assert allocator.free_runs() == [80]


def test_growth_relocates_when_a_neighbour_blocks_it_in_place():
    allocator = ContiguousKVAllocator(total_slots=100)
    allocator.allocate("a", 10)
    allocator.allocate("b", 10)  # sits immediately after a
    assert allocator.extend("a", 30) is True
    # a could not grow in place, so it moved -- a real copy of the whole
    # sequence, which paging never needs to perform.
    assert allocator.physical_slots_used() == 50


def test_rejects_a_non_positive_pool():
    with pytest.raises(ValueError):
        ContiguousKVAllocator(total_slots=0)
