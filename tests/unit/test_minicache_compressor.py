import math

import pytest

from src.cag.domain.compression_metrics import compression_ratio, reconstruction_error
from src.cag.domain.entities import CompressedKV
from src.cag.infrastructure.minicache_compressor import MiniCacheCompressor


def _rows(n: int, channels: int, offset: float = 0.0) -> list[list[float]]:
    # Deterministic, varied, and every component is >= 1.0 (never zero),
    # so magnitude/direction decomposition is well-defined for every row
    # without accidentally hitting the zero-vector edge case, which gets
    # its own dedicated test below.
    return [
        [float((t * 5 + c * 2) % 17) + 1.0 + offset for c in range(channels)] for t in range(n)
    ]


def test_compress_returns_minicache_as_the_method_name():
    compressor = MiniCacheCompressor()
    result = compressor.compress(_rows(4, 3), _rows(4, 3, offset=2.0))
    assert result.method == "minicache"


def test_round_trip_shape_matches_the_original_for_both_layers():
    compressor = MiniCacheCompressor()
    layer_a = _rows(10, 6)
    layer_b = _rows(10, 6, offset=3.0)
    compressed = compressor.compress(layer_a, layer_b)
    reconstructed_a, reconstructed_b = compressor.decompress(compressed)
    assert len(reconstructed_a) == len(layer_a)
    assert len(reconstructed_b) == len(layer_b)
    assert all(len(row) == len(layer_a[0]) for row in reconstructed_a)
    assert all(len(row) == len(layer_b[0]) for row in reconstructed_b)


def test_identical_layers_reconstruct_near_exactly():
    # Two identical rows have zero angle between their directions -- SLERP
    # of a direction with itself IS that direction exactly (the near-
    # parallel fallback's own linear-interpolation formula reduces to
    # 0.5*u + 0.5*u == u when u == v), so reconstruction should recover the
    # original almost bit-for-bit, not merely "close," proving the edge-case
    # handling doesn't corrupt the easy case.
    compressor = MiniCacheCompressor()
    rows = _rows(6, 5)
    compressed = compressor.compress(rows, rows)
    reconstructed_a, reconstructed_b = compressor.decompress(compressed)
    assert reconstruction_error(rows, reconstructed_a) < 1e-9
    assert reconstruction_error(rows, reconstructed_b) < 1e-9


def test_slerp_handles_near_parallel_vectors_without_nan_or_crash():
    # theta here is on the order of 1e-8 to 1e-6 (measured directly while
    # designing this test) -- small enough that sin(theta) sits under the
    # implementation's near-parallel threshold, so plain SLERP's division
    # would be dividing by a value indistinguishable from zero, but the
    # rows are NOT literally identical, so this exercises the fallback
    # branch on a genuinely distinct (if tiny) angle, not the exact-
    # equality case covered above.
    compressor = MiniCacheCompressor()
    layer_a = [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [2.0, 2.0, 2.0]]
    layer_b = [[1.0 + 1e-7, 2.0, 3.0], [4.0 + 1e-8, -1.0, 0.5], [2.0, 2.0, 2.0 + 5e-7]]
    compressed = compressor.compress(layer_a, layer_b)
    reconstructed_a, reconstructed_b = compressor.decompress(compressed)
    for row in (*reconstructed_a, *reconstructed_b):
        assert all(math.isfinite(value) for value in row)
    assert reconstruction_error(layer_a, reconstructed_a) < 1e-4
    assert reconstruction_error(layer_b, reconstructed_b) < 1e-4


def test_a_zero_vector_row_does_not_crash_or_produce_nan():
    # A token row that is all zeros has no defined direction -- the
    # implementation must not divide by zero computing its unit vector.
    # The zero row itself reconstructs exactly (magnitude 0 times anything
    # is 0); the OTHER layer's row at that same token index necessarily
    # loses fidelity (its direction gets blended against an arbitrary
    # zero-vector "direction"), which is honest, disclosed lossy behavior
    # for this pathological input, not a bug to paper over.
    compressor = MiniCacheCompressor()
    layer_a = [[0.0, 0.0, 0.0]]
    layer_b = [[1.0, 2.0, 2.0]]
    compressed = compressor.compress(layer_a, layer_b)
    reconstructed_a, reconstructed_b = compressor.decompress(compressed)
    for row in (*reconstructed_a, *reconstructed_b):
        assert all(math.isfinite(value) for value in row)
    assert reconstructed_a[0] == pytest.approx([0.0, 0.0, 0.0])


def test_round_trip_reconstruction_error_matches_the_orthogonal_geometry_exactly():
    # layer_a's directions are all (1, 0), layer_b's are all (0, 1) --
    # maximally different unit directions (90 degrees apart). SLERP at
    # t=0.5 between two orthogonal unit vectors lands exactly on their
    # bisector (sqrt(2)/2, sqrt(2)/2), and for THIS specific 2-channel,
    # axis-aligned geometry the per-row mean absolute reconstruction error
    # works out to exactly magnitude/2: each row's two per-channel errors
    # are s*m and (1-s)*m for s = sqrt(2)/2, which always sum to exactly m
    # regardless of s, so MAE over 2 channels is m/2. Averaged over tokens
    # with different magnitudes, the expected overall MAE is the mean
    # magnitude / 2 (worked out by hand and confirmed numerically before
    # writing this assertion).
    compressor = MiniCacheCompressor()
    magnitudes_a = [2.0, 4.0, 6.0, 8.0]
    magnitudes_b = [1.0, 3.0, 5.0, 7.0]
    layer_a = [[m, 0.0] for m in magnitudes_a]
    layer_b = [[0.0, m] for m in magnitudes_b]
    compressed = compressor.compress(layer_a, layer_b)
    reconstructed_a, reconstructed_b = compressor.decompress(compressed)
    expected_error_a = (sum(magnitudes_a) / len(magnitudes_a)) / 2
    expected_error_b = (sum(magnitudes_b) / len(magnitudes_b)) / 2
    assert reconstruction_error(layer_a, reconstructed_a) == pytest.approx(expected_error_a)
    assert reconstruction_error(layer_b, reconstructed_b) == pytest.approx(expected_error_b)


def test_measured_compression_ratio_is_in_the_neighborhood_of_cag_mds_two_to_three_x():
    # CAG.md states MiniCache achieves "~2-3x". The mechanism here shares
    # ONE direction array between the two layers and keeps only two small
    # per-layer magnitude arrays -- element count math:
    #   original:   2 * num_tokens * num_channels   (two full layers)
    #   compressed: num_tokens * num_channels        (shared direction)
    #             + num_tokens + num_tokens           (magnitudes_a, magnitudes_b)
    # At equal bit width this reduces to 2*C/(C+2), which APPROACHES but
    # never REACHES 2x as num_channels grows, because the shared direction
    # array alone already costs half the original element count and the
    # magnitude arrays are only a small addition on top. For a realistic
    # per-head channel count (64, matching distilgpt2's head dim) that
    # works out to ~1.94x -- honestly short of CAG.md's stated "2-3x" for
    # this pure structural-sharing mechanism; see the report for why.
    compressor = MiniCacheCompressor()
    num_tokens, num_channels = 10, 64
    layer_a = _rows(num_tokens, num_channels)
    layer_b = _rows(num_tokens, num_channels, offset=5.0)
    compressed = compressor.compress(layer_a, layer_b)
    payload = compressed.payload
    direction_elements = len(payload["shared_directions"]) * len(payload["shared_directions"][0])
    magnitude_elements = len(payload["magnitudes_a"]) + len(payload["magnitudes_b"])
    compressed_elements = direction_elements + magnitude_elements
    original_elements = 2 * num_tokens * num_channels
    ratio = compression_ratio(
        original_bit_width=32,
        original_element_count=original_elements,
        compressed_bit_width=32,
        compressed_element_count=compressed_elements,
    )
    assert 1.9 <= ratio <= 2.0


def test_decompress_rejects_a_payload_from_a_different_method():
    compressor = MiniCacheCompressor()
    with pytest.raises(ValueError):
        compressor.decompress(
            CompressedKV(method="not-minicache", payload={}, original_shape=(1, 1))
        )
