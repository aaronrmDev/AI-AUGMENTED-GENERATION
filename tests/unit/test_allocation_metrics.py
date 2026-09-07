import pytest

from src.cag.domain.allocation_metrics import external_fragmentation, internal_fragmentation


def test_internal_fragmentation_is_zero_when_every_slot_holds_a_token():
    assert internal_fragmentation(physical_slots_used=64, logical_tokens_held=64) == pytest.approx(
        0.0
    )


def test_internal_fragmentation_is_the_wasted_fraction_of_allocated_slots():
    # 64 slots held for 48 real tokens -> a quarter of the space is the
    # tail of a partly-filled block.
    assert internal_fragmentation(physical_slots_used=64, logical_tokens_held=48) == pytest.approx(
        0.25
    )


def test_internal_fragmentation_of_an_empty_pool_is_zero_not_a_division_error():
    assert internal_fragmentation(
        physical_slots_used=0, logical_tokens_held=0
    ) == pytest.approx(0.0)


def test_internal_fragmentation_rejects_holding_more_tokens_than_slots():
    with pytest.raises(ValueError):
        internal_fragmentation(physical_slots_used=16, logical_tokens_held=32)


def test_external_fragmentation_is_zero_when_all_free_space_is_one_run():
    assert external_fragmentation([128]) == pytest.approx(0.0)


def test_external_fragmentation_reflects_free_space_scattered_into_slivers():
    # 100 slots free but the biggest usable run is only 10 of them.
    assert external_fragmentation([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]) == pytest.approx(0.9)


def test_external_fragmentation_of_a_full_pool_is_zero():
    assert external_fragmentation([]) == pytest.approx(0.0)


def test_external_fragmentation_rejects_negative_runs():
    with pytest.raises(ValueError):
        external_fragmentation([16, -1])
