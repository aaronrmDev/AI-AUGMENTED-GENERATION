import pytest

from src.cag.domain.compression_metrics import compression_ratio, reconstruction_error
from src.cag.domain.entities import CompressedKV
from src.cag.infrastructure.palu_compressor import PALUCompressor
from src.cag.infrastructure.shadowkv_compressor import ShadowKVCompressor


def _tokens(n: int, channels: int) -> list[list[float]]:
    # Same deterministic generator KIVI/PALU/KVQuant's own test files use,
    # so a row/column mix-up shows up as elevated reconstruction error
    # instead of passing by accident on data with no per-axis distinction.
    return [[float((t * 7 + c * 3) % 23) - 11.0 for c in range(channels)] for t in range(n)]


def test_compress_returns_shadowkv_as_the_method_name():
    compressor = ShadowKVCompressor(rank=4, sparsity_ratio=0.5, bits=4)
    result = compressor.compress(_tokens(16, 16))
    assert result.method == "shadowkv"


def test_round_trip_shape_matches_the_original():
    compressor = ShadowKVCompressor(rank=4, sparsity_ratio=0.5, bits=4)
    original = _tokens(20, 16)
    compressed = compressor.compress(original)
    reconstructed = compressor.decompress(compressed)
    assert len(reconstructed) == len(original)
    assert all(len(row) == len(original[0]) for row in reconstructed)


def test_zero_sparsity_and_high_bits_reduces_to_roughly_palus_own_error():
    # Sparsity ratio 0.0 (keep every element) and enough bits that
    # quantization error is negligible should leave ShadowKV's own
    # low-rank step doing essentially all the work -- its reconstruction
    # error should land close to PALU's own error on the identical input
    # and rank, proving the composition doesn't silently corrupt the
    # underlying low-rank step it's built on.
    original = _tokens(20, 16)
    rank = 4
    palu_only = PALUCompressor(rank=rank)
    palu_error = reconstruction_error(original, palu_only.decompress(palu_only.compress(original)))

    shadowkv = ShadowKVCompressor(rank=rank, sparsity_ratio=0.0, bits=16)
    shadowkv_error = reconstruction_error(
        original, shadowkv.decompress(shadowkv.compress(original))
    )
    # Not bit-for-bit equal: 16-bit quantization is close to lossless but
    # not literally lossless, so a small residual step-size error on top
    # of PALU's own truncation error is expected, not a bug -- abs=1e-3
    # is comfortably above that real 16-bit rounding noise while still
    # ruling out a much coarser discrepancy (e.g. a sparsity or
    # quantization bug corrupting far more than a rounding-level amount).
    assert shadowkv_error == pytest.approx(palu_error, abs=1e-3)


def test_higher_sparsity_ratio_increases_reconstruction_error_monotonically():
    # Dropping more of the low-rank latent's own entries can only lose
    # more information, never less -- a real, checkable monotonicity
    # property of the sparsity mechanism itself, independent of the exact
    # error values.
    original = _tokens(24, 16)
    errors = []
    for sparsity_ratio in (0.0, 0.3, 0.6, 0.9):
        compressor = ShadowKVCompressor(rank=4, sparsity_ratio=sparsity_ratio, bits=16)
        compressed = compressor.compress(original)
        errors.append(reconstruction_error(original, compressor.decompress(compressed)))
    assert errors == sorted(errors)


def test_sparse_h_entry_count_matches_the_requested_sparsity_ratio():
    compressor = ShadowKVCompressor(rank=4, sparsity_ratio=0.5, bits=4)
    original = _tokens(20, 16)
    compressed = compressor.compress(original)
    total_h_elements = compressed.payload["h_shape"][0] * compressed.payload["h_shape"][1]
    surviving = len(compressed.payload["sparse_h_entries"])
    # round() on an exact half-split can land one element off depending on
    # tie-breaking -- allow a slack of 1 rather than asserting exact
    # equality against a rounding-sensitive target.
    assert abs(surviving - round(total_h_elements * 0.5)) <= 1


def test_measured_compression_ratio_is_in_the_neighborhood_of_cag_mds_six_x():
    # Composes PALU's own ~4x (32x32 tensor, rank=4, established in
    # palu_compressor's own tests) with 50% sparsity plus 4-bit
    # quantization of the surviving low-rank entries -- stacking
    # mechanisms is CAG.md's own stated reason ShadowKV reaches further
    # than any single technique alone ("~6x... by stacking mechanisms
    # rather than relying on any one of them alone").
    bits = 4
    compressor = ShadowKVCompressor(rank=4, sparsity_ratio=0.5, bits=bits)
    original = _tokens(32, 32)
    compressed = compressor.compress(original)
    b = compressed.payload["palu_B"]
    b_elements = len(b) * len(b[0])
    sparse_entries = len(compressed.payload["sparse_h_entries"])
    # Each sparse entry costs its quantized value (bits) plus its (row,
    # col) position -- 32 bits total is a generous, real estimate for an
    # int32 pair of coordinates at this tensor's small scale, not a
    # number picked to flatter the ratio. Reuses the local `bits` this
    # test already passed into the constructor, matching every sibling
    # compressor test's own convention, rather than reading it back off
    # the compressor instance (review-caught: ShadowKV was the only
    # compressor exposing this as a public attribute, solely to serve
    # this one line -- now private like every sibling's equivalent state).
    compressed_bits = b_elements * 32 + sparse_entries * (bits + 32)
    ratio = compression_ratio(
        original_bit_width=32, original_element_count=32 * 32,
        compressed_bit_width=1, compressed_element_count=compressed_bits,
    )
    assert ratio >= 4.5  # strictly beats PALU's own ~4x alone; CAG.md's ~6x is the target


def test_decompress_rejects_a_payload_from_a_different_method():
    compressor = ShadowKVCompressor(rank=4, sparsity_ratio=0.5, bits=4)
    with pytest.raises(ValueError):
        compressor.decompress(
            CompressedKV(method="not-shadowkv", payload={}, original_shape=(1, 1))
        )


def test_rejects_an_out_of_range_sparsity_ratio():
    with pytest.raises(ValueError):
        ShadowKVCompressor(rank=4, sparsity_ratio=1.5, bits=4)
    with pytest.raises(ValueError):
        ShadowKVCompressor(rank=4, sparsity_ratio=-0.1, bits=4)
