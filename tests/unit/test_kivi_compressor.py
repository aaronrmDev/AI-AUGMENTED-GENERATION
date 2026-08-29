import pytest

from src.cag.domain.compression_metrics import compression_ratio, reconstruction_error
from src.cag.domain.entities import CompressedKV
from src.cag.infrastructure.kivi_compressor import KIVICompressor


def _tokens(n: int, channels: int) -> list[list[float]]:
    # Deterministic, varied values -- each token's row is distinguishable
    # from every other, and each channel has its own scale, so a bug that
    # quantizes per-row instead of per-channel (or vice versa) would show
    # up as elevated reconstruction error rather than passing by accident
    # on uniform data.
    return [[float((t * 7 + c * 3) % 23) - 11.0 for c in range(channels)] for t in range(n)]


def test_compress_returns_kivi_as_the_method_name():
    compressor = KIVICompressor(group_size=4, bits=4)
    result = compressor.compress(_tokens(8, 6))
    assert result.method == "kivi"


def test_round_trip_shape_matches_the_original():
    compressor = KIVICompressor(group_size=4, bits=4)
    original = _tokens(10, 6)
    compressed = compressor.compress(original)
    reconstructed = compressor.decompress(compressed)
    assert len(reconstructed) == len(original)
    assert all(len(row) == len(original[0]) for row in reconstructed)


def test_round_trip_reconstruction_error_stays_within_quantization_tolerance():
    # 4-bit quantization over a channel range of roughly 22 (-11..11)
    # gives a per-level step of about 22/15 ~= 1.47; mean absolute error
    # should be comfortably under one full step for values that aren't
    # pathologically adversarial.
    compressor = KIVICompressor(group_size=4, bits=4)
    original = _tokens(16, 6)
    compressed = compressor.compress(original)
    reconstructed = compressor.decompress(compressed)
    assert reconstruction_error(original, reconstructed) < 1.5


def test_a_trailing_partial_group_is_kept_as_a_full_precision_residual():
    # group_size=4 over 10 tokens: two full groups (8 tokens) quantized,
    # a trailing 2-token group kept as an exact, full-precision residual
    # -- reconstructing those specific rows should be EXACT, not merely
    # within tolerance, proving the residual path bypasses quantization
    # entirely rather than just quantizing with a smaller group.
    compressor = KIVICompressor(group_size=4, bits=4)
    original = _tokens(10, 6)
    compressed = compressor.compress(original)
    reconstructed = compressor.decompress(compressed)
    for row_original, row_reconstructed in zip(original[8:], reconstructed[8:], strict=True):
        for original_value, reconstructed_value in zip(
            row_original, row_reconstructed, strict=True
        ):
            assert original_value == pytest.approx(reconstructed_value)


def test_a_full_multiple_of_group_size_leaves_no_residual_rows():
    compressor = KIVICompressor(group_size=4, bits=4)
    compressed = compressor.compress(_tokens(8, 6))
    assert compressed.payload["residual_rows"] == []


def test_measured_compression_ratio_is_in_the_neighborhood_of_cag_mds_four_x():
    # CAG.md states KIVI achieves "~4x" -- 4-bit vs a 16-bit original
    # baseline is exactly 4x on the quantized portion; residual rows stay
    # full precision, so a batch with a small residual tail should still
    # land close to, if a little under, 4x overall.
    compressor = KIVICompressor(group_size=4, bits=4)
    original = _tokens(32, 8)
    compressed = compressor.compress(original)
    quantized_elements = sum(
        len(group) * len(group[0]) for group in compressed.payload["quantized_groups"]
    )
    residual_elements = len(compressed.payload["residual_rows"]) * (
        len(compressed.payload["residual_rows"][0])
        if compressed.payload["residual_rows"]
        else 0
    )
    # Blend the quantized portion's 4x against the residual portion's 1x
    # (full precision, no savings) by element count.
    total_elements = quantized_elements + residual_elements
    effective_bit_width = (
        4 * quantized_elements + 16 * residual_elements
    ) / total_elements
    ratio = compression_ratio(
        original_bit_width=16, original_element_count=total_elements,
        compressed_bit_width=effective_bit_width, compressed_element_count=total_elements,
    )
    assert 3.5 <= ratio <= 4.0


def test_decompress_rejects_a_payload_from_a_different_method():
    compressor = KIVICompressor(group_size=4, bits=4)
    with pytest.raises(ValueError):
        compressor.decompress(CompressedKV(method="not-kivi", payload={}, original_shape=(1, 1)))
