import pytest

from src.cag.domain.allocation_metrics import external_fragmentation
from src.cag.infrastructure.paged_allocator import PagedKVAllocator


def test_allocation_rounds_up_to_whole_blocks():
    allocator = PagedKVAllocator(total_blocks=8, block_size=16)
    assert allocator.allocate("a", 17) is True
    # 17 tokens needs 2 blocks = 32 slots, of which 15 sit empty.
    assert allocator.physical_slots_used() == 32
    assert allocator.logical_tokens_held() == 17


def test_allocation_fails_when_the_pool_has_too_few_blocks_left():
    allocator = PagedKVAllocator(total_blocks=2, block_size=16)
    assert allocator.allocate("a", 32) is True
    assert allocator.allocate("b", 1) is False


def test_a_fork_costs_no_additional_blocks_at_all():
    # CAG.md's parallel-sampling claim, stated as a test: N branches of
    # one prompt hold ONE physical copy of it.
    allocator = PagedKVAllocator(total_blocks=64, block_size=16)
    allocator.allocate("parent", 160)
    before = allocator.physical_slots_used()
    for i in range(7):
        assert allocator.fork("parent", f"child{i}") is True
    assert allocator.physical_slots_used() == before
    # Eight sequences now hold 160 logical tokens each off that one copy.
    assert allocator.logical_tokens_held() == 160 * 8


def test_copy_on_write_gives_a_forked_branch_its_own_last_block_when_it_grows():
    allocator = PagedKVAllocator(total_blocks=64, block_size=16)
    allocator.allocate("parent", 20)  # 2 blocks, last one partly filled
    allocator.fork("parent", "child")
    before = allocator.physical_slots_used()
    assert allocator.extend("child", 1) is True
    # Exactly one block copied -- the shared, partly-filled tail.
    assert allocator.physical_slots_used() == before + 16


def test_blocks_return_to_the_pool_only_once_every_sharer_has_freed_them():
    allocator = PagedKVAllocator(total_blocks=4, block_size=16)
    allocator.allocate("parent", 64)
    allocator.fork("parent", "child")
    allocator.free("parent")
    # The child still holds every block, so nothing is reusable yet.
    assert allocator.allocate("other", 16) is False
    allocator.free("child")
    assert allocator.allocate("other", 16) is True


def test_free_space_is_never_externally_fragmented_by_construction():
    allocator = PagedKVAllocator(total_blocks=16, block_size=16)
    for name, size in (("a", 32), ("b", 48), ("c", 16)):
        allocator.allocate(name, size)
    allocator.free("b")  # frees blocks from the middle of the pool
    assert external_fragmentation(allocator.free_runs()) == pytest.approx(0.0)


def test_rejects_allocating_the_same_sequence_twice():
    allocator = PagedKVAllocator(total_blocks=8, block_size=16)
    allocator.allocate("a", 16)
    with pytest.raises(ValueError):
        allocator.allocate("a", 16)


def test_rejects_a_non_positive_block_size():
    with pytest.raises(ValueError):
        PagedKVAllocator(total_blocks=8, block_size=0)
