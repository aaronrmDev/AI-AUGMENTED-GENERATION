import pytest

from src.cag.domain.compression_metrics import compression_ratio, reconstruction_error


def test_compression_ratio_four_x_from_sixteen_bit_to_four_bit():
    assert compression_ratio(
        original_bit_width=16, original_element_count=100,
        compressed_bit_width=4, compressed_element_count=100,
    ) == pytest.approx(4.0)


def test_compression_ratio_accounts_for_a_smaller_compressed_element_count():
    # PALU-style: same bit width, but far fewer elements survive
    # (a low-rank projection stores a compact latent, not a same-sized
    # array at a smaller bit width) -- the ratio must reflect count
    # reduction, not just bit-width reduction.
    assert compression_ratio(
        original_bit_width=16, original_element_count=100,
        compressed_bit_width=16, compressed_element_count=25,
    ) == pytest.approx(4.0)


def test_compression_ratio_of_one_when_nothing_is_saved():
    assert compression_ratio(
        original_bit_width=16, original_element_count=100,
        compressed_bit_width=16, compressed_element_count=100,
    ) == pytest.approx(1.0)


def test_compression_ratio_rejects_zero_compressed_bits():
    with pytest.raises(ValueError):
        compression_ratio(
            original_bit_width=16, original_element_count=100,
            compressed_bit_width=0, compressed_element_count=100,
        )


def test_reconstruction_error_is_zero_for_an_identical_reconstruction():
    original = [[1.0, 2.0], [3.0, 4.0]]
    assert reconstruction_error(original, original) == pytest.approx(0.0)


def test_reconstruction_error_is_the_mean_absolute_difference():
    original = [[1.0, 2.0], [3.0, 4.0]]
    reconstructed = [[1.0, 3.0], [3.0, 6.0]]
    # differences: 0, 1, 0, 2 -> mean = 0.75
    assert reconstruction_error(original, reconstructed) == pytest.approx(0.75)


def test_reconstruction_error_rejects_mismatched_row_counts():
    with pytest.raises(ValueError):
        reconstruction_error([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]])


def test_reconstruction_error_rejects_mismatched_column_counts():
    with pytest.raises(ValueError):
        reconstruction_error([[1.0, 2.0]], [[1.0, 2.0, 3.0]])


def test_reconstruction_error_rejects_empty_input():
    with pytest.raises(ValueError):
        reconstruction_error([], [])
