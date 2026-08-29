import numpy as np
import pytest

from src.cag.domain.compression_metrics import compression_ratio, reconstruction_error
from src.cag.domain.entities import CompressedKV
from src.cag.infrastructure.palu_compressor import PALUCompressor


def _tokens(n: int, channels: int) -> list[list[float]]:
    # Deterministic, varied values -- same generator KIVI's own test file
    # uses, so a bug that mixes up rows and columns shows up as elevated
    # reconstruction error rather than passing by accident on data with
    # no per-row/per-column distinction.
    return [[float((t * 7 + c * 3) % 23) - 11.0 for c in range(channels)] for t in range(n)]


def _low_rank_tokens(
    n: int, channels: int, components: list[tuple[list[float], list[float]]]
) -> list[list[float]]:
    # Build kv as an EXACT sum of a handful of rank-1 outer products
    # (u_i outer v_i) -- its true rank is len(components) by
    # construction, not by approximation. Setting PALU's rank parameter
    # to that same number should let the SVD truncation capture the
    # tensor fully, so reconstruction can be checked for being
    # near-EXACT rather than merely "small enough" -- proving the SVD
    # decompose/truncate/reconstruct math is genuinely correct rather
    # than just plausible-looking on data no one can independently verify.
    matrix = np.zeros((n, channels))
    for u, v in components:
        matrix += np.outer(np.array(u), np.array(v))
    result: list[list[float]] = matrix.tolist()
    return result


def test_compress_returns_palu_as_the_method_name():
    compressor = PALUCompressor(rank=4)
    result = compressor.compress(_tokens(8, 6))
    assert result.method == "palu"


def test_round_trip_shape_matches_the_original():
    compressor = PALUCompressor(rank=4)
    original = _tokens(10, 6)
    compressed = compressor.compress(original)
    reconstructed = compressor.decompress(compressed)
    assert len(reconstructed) == len(original)
    assert all(len(row) == len(original[0]) for row in reconstructed)


def test_round_trip_reconstruction_error_matches_the_theoretical_truncated_svd_bound():
    # Eckart-Young: the best possible rank-k approximation's Frobenius
    # error is exactly sqrt(sum of the squared singular values PAST rank
    # k). Computing that bound independently here (via numpy's own SVD,
    # not the compressor's) and converting it to a per-element RMS gives
    # a ceiling on mean absolute error derived from the algorithm itself
    # -- MAE <= RMSE always holds (Cauchy-Schwarz) -- rather than an
    # arbitrary constant picked to make the test pass.
    original = _tokens(16, 6)
    rank = 4
    matrix = np.array(original, dtype=np.float64)
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    discarded = s[rank:]
    frobenius_error_ceiling = float(np.sqrt(np.sum(discarded**2)))
    rmse_ceiling = frobenius_error_ceiling / np.sqrt(matrix.size)

    compressor = PALUCompressor(rank=rank)
    compressed = compressor.compress(original)
    reconstructed = compressor.decompress(compressed)
    # 1% float slack: our implementation runs the identical numpy SVD
    # truncated to the same rank, so it should land at essentially the
    # same optimum this bound was computed from, not merely under it.
    assert reconstruction_error(original, reconstructed) <= rmse_ceiling * 1.01


def test_round_trip_is_near_exact_when_kv_is_genuinely_low_rank_by_construction():
    rng = np.random.default_rng(42)
    components = [(rng.normal(size=12).tolist(), rng.normal(size=8).tolist()) for _ in range(3)]
    original = _low_rank_tokens(12, 8, components)
    compressor = PALUCompressor(rank=3)
    compressed = compressor.compress(original)
    reconstructed = compressor.decompress(compressed)
    assert reconstruction_error(original, reconstructed) < 1e-9


def test_a_rank_larger_than_the_matrix_supports_is_clamped_to_full_rank():
    # SVD can never produce more than min(tokens, channels) singular
    # values/vectors -- asking PALU for more rank than that ceiling
    # should behave like asking for exactly full rank (near-exact
    # reconstruction, since full rank IS exact), not raise or silently
    # produce a mis-shaped payload.
    original = _tokens(5, 4)  # min(tokens, channels) = 4
    compressor = PALUCompressor(rank=100)
    compressed = compressor.compress(original)
    reconstructed = compressor.decompress(compressed)
    assert compressed.payload["rank"] == 4
    assert reconstruction_error(original, reconstructed) < 1e-9


def test_measured_compression_ratio_is_in_the_neighborhood_of_cag_mds_four_x():
    # Default rank=4, chosen so a square [N, N] KV tensor's compressed
    # element count (H's N*rank plus B's rank*N = 2*rank*N) lands near
    # CAG.md's "~4x": ratio = N^2 / (2*rank*N) = N / (2*rank) = 32/8 =
    # 4.0 exactly for a 32-token, 32-channel tensor. PALU is
    # dimensionality reduction, not precision reduction, so bit width is
    # unchanged between original and compressed -- the entire saving
    # comes from storing fewer elements, which is why the same bit width
    # is passed for both sides here.
    compressor = PALUCompressor(rank=4)
    original = _tokens(32, 32)
    compressed = compressor.compress(original)
    h = compressed.payload["H"]
    b = compressed.payload["B"]
    compressed_element_count = len(h) * len(h[0]) + len(b) * len(b[0])
    ratio = compression_ratio(
        original_bit_width=32,
        original_element_count=32 * 32,
        compressed_bit_width=32,
        compressed_element_count=compressed_element_count,
    )
    assert 3.5 <= ratio <= 4.5


def test_decompress_rejects_a_payload_from_a_different_method():
    compressor = PALUCompressor(rank=4)
    with pytest.raises(ValueError):
        compressor.decompress(CompressedKV(method="not-palu", payload={}, original_shape=(1, 1)))
