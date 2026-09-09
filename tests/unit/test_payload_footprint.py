import pytest

from src.cag.domain.payload_footprint import dense_bits, payload_bits


def test_dense_bits_is_rows_times_channels_times_width():
    assert dense_bits(rows=16, channels=8, bit_width=32) == pytest.approx(4096.0)


def test_quantized_ints_are_charged_the_quantized_width():
    payload = {"quantized": [[1, 2, 3, 4]]}
    assert payload_bits(payload, quantized_bit_width=4) == pytest.approx(16.0)


def test_full_precision_floats_are_charged_thirty_two_bits():
    payload = {"scales": [0.5, 0.25]}
    assert payload_bits(payload, quantized_bit_width=4) == pytest.approx(64.0)


def test_a_mixed_payload_charges_each_part_at_its_own_width():
    # The shape a real quantizer produces: quantized values plus
    # full-precision scales and zero-points kept alongside them.
    payload = {"quantized": [[1, 2, 3, 4]], "scales": [0.5], "zero_points": [0.0]}
    assert payload_bits(payload, quantized_bit_width=4) == pytest.approx(16.0 + 32.0 + 32.0)


def test_counting_elements_instead_of_bits_would_have_reported_inflation():
    # The bug this module exists to prevent, pinned as a test. Sixteen
    # values quantized to 4 bits plus two float scales is 4 elements
    # more than the 16 originals -- "bigger" by element count -- while
    # being nearly 4x smaller in bits, which is what actually matters.
    payload = {"quantized": [[1] * 16], "scales": [0.5, 0.25]}
    element_count = 16 + 2
    assert element_count > 16
    assert payload_bits(payload, quantized_bit_width=4) < dense_bits(1, 16, 32)


def test_booleans_are_not_mistaken_for_quantized_values():
    # bool is a subclass of int in Python, so an unguarded check would
    # charge a flag as if it were cache data.
    assert payload_bits({"flag": [True, False]}, quantized_bit_width=4) == pytest.approx(0.0)


def test_rejects_a_non_positive_quantized_width():
    with pytest.raises(ValueError):
        payload_bits({"a": [1]}, quantized_bit_width=0)


def test_dense_bits_rejects_negative_dimensions():
    with pytest.raises(ValueError):
        dense_bits(rows=-1, channels=8)
