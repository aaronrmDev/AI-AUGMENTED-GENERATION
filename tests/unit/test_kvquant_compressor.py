import pytest

from src.cag.domain.compression_metrics import compression_ratio, reconstruction_error
from src.cag.domain.entities import CompressedKV
from src.cag.infrastructure.kivi_compressor import KIVICompressor
from src.cag.infrastructure.kvquant_compressor import KVQuantCompressor


def _tokens(n: int, channels: int) -> list[list[float]]:
    # Same generator KIVI's own test file uses: deterministic, varied,
    # roughly-uniformly-spread values per channel (mod 23, no channel value
    # ever drifts far enough from its own channel mean to be flagged an
    # outlier at the default threshold) so a bug that quantizes per-row
    # instead of per-channel shows up as elevated error rather than passing
    # by accident on uniform data.
    return [[float((t * 7 + c * 3) % 23) - 11.0 for c in range(channels)] for t in range(n)]


def _clustered_tokens_with_one_outlier(
    n: int, channels: int, outlier_row: int, outlier_channel: int, outlier_value: float
) -> list[list[float]]:
    # A tight, low-variance cluster (0.01 spacing) in every channel, with a
    # single deliberately huge value dropped into one (row, channel) cell.
    # n must be large enough that a single extreme point doesn't drag the
    # channel's own mean/std up enough to mask itself (see this file's
    # calibration note on test_outlier_values_are_reconstructed_exactly).
    rows = [[10.0 + 0.01 * t for _ in range(channels)] for t in range(n)]
    rows[outlier_row][outlier_channel] = outlier_value
    return rows


def test_compress_returns_kvquant_as_the_method_name():
    compressor = KVQuantCompressor(bits=4)
    result = compressor.compress(_tokens(8, 6))
    assert result.method == "kvquant"


def test_round_trip_shape_matches_the_original():
    compressor = KVQuantCompressor(bits=4)
    original = _tokens(10, 6)
    compressed = compressor.compress(original)
    reconstructed = compressor.decompress(compressed)
    assert len(reconstructed) == len(original)
    assert all(len(row) == len(original[0]) for row in reconstructed)


def test_round_trip_reconstruction_error_stays_within_quantization_tolerance():
    # 4-bit quantization gives 16 quantile buckets. This data's channel
    # range is about 22 (-11..11), so an equal-WIDTH 4-bit step would be
    # about 22/16 ~= 1.375; equal-POPULATION buckets on this
    # near-uniformly-distributed data should do at least that well, since
    # they place bucket boundaries exactly where the data's own order
    # statistics fall rather than assuming an even spread. Mean absolute
    # error should be comfortably under one such step. (Measured on this
    # exact data during development: ~0.32 -- the 0.75 bound below leaves
    # real margin rather than pinning the test to that precise figure.)
    compressor = KVQuantCompressor(bits=4)
    original = _tokens(24, 6)
    compressed = compressor.compress(original)
    reconstructed = compressor.decompress(compressed)
    assert reconstruction_error(original, reconstructed) < 0.75


def test_outlier_values_are_reconstructed_exactly():
    # KVQuant's own distinguishing special case (vs KIVI's trailing-residual
    # exact case): a value flagged as a per-channel outlier is stored at
    # full precision by (row, channel) position and must round-trip EXACT,
    # not merely within quantization tolerance, proving the outlier path
    # bypasses the quantile codebook entirely rather than just landing in
    # whatever bucket happens to be nearest.
    original = _clustered_tokens_with_one_outlier(
        n=10, channels=3, outlier_row=5, outlier_channel=1, outlier_value=1000.0
    )
    compressor = KVQuantCompressor(bits=4, outlier_std_threshold=2.5)
    compressed = compressor.compress(original)
    assert (5, 1) in compressed.payload["outliers"]
    reconstructed = compressor.decompress(compressed)
    assert reconstructed[5][1] == pytest.approx(1000.0)
    # The rest of that same row (untouched, tightly-clustered channels)
    # should still be close to exact too -- only the outlier CELL was
    # special-cased, not the whole row.
    assert reconstructed[5][0] == pytest.approx(original[5][0], abs=0.1)
    assert reconstructed[5][2] == pytest.approx(original[5][2], abs=0.1)


def test_quantile_bucket_boundaries_adapt_to_a_skewed_distribution():
    # The distinguishing characteristic vs KIVI's uniform linear
    # scale/zero-point: bucket boundaries are equal-POPULATION (quantile),
    # not equal-WIDTH. On a skewed channel -- a tight cluster of 15 values
    # plus 5 more spread far apart -- equal-population buckets should pack
    # several narrow buckets inside the dense cluster and leave the sparse
    # tail in one wide bucket, so the codebook's own decode values land
    # much closer together in the dense region than in the sparse one.
    # outlier_std_threshold is set very high here specifically to isolate
    # bucket-shape behavior from outlier detection (a separate mechanism,
    # covered by its own test above) -- none of this data should be pulled
    # out as an outlier for this test to mean what it claims.
    dense = [round(0.05 * t, 4) for t in range(15)]
    sparse = [5.0, 7.0, 9.0, 11.0, 13.0]
    values = dense + sparse
    kv = [[value, value] for value in values]
    compressor = KVQuantCompressor(bits=2, outlier_std_threshold=100.0)
    compressed = compressor.compress(kv)
    assert compressed.payload["outliers"] == {}
    codebook = compressed.payload["codebooks"][0]
    gaps = [codebook[i + 1] - codebook[i] for i in range(len(codebook) - 1)]
    # A uniform (equal-width) scheme would produce roughly equal gaps; a
    # quantile scheme packs buckets into the dense cluster, so the widest
    # gap (spanning the sparse tail) should dwarf the narrowest one.
    assert max(gaps) > 5 * min(gaps)


def test_measured_compression_ratio_at_low_bit_width_approaches_cag_mds_up_to_8x():
    # CAG.md states KVQuant reaches "up to 8x" -- a ceiling, not KIVI's
    # flatter "~4x". The compression-ratio arithmetic itself is identical
    # to KIVI's (original bit width over compressed bit width): what makes
    # 8x reachable at all is that outlier isolation no longer forces the
    # quantized range wide, so a LOWER bit width than KIVI's usual 4-bit
    # baseline still holds acceptable fidelity. At bits=2, 16/2 = 8x on the
    # quantized portion; this dataset happens to produce zero outliers (see
    # _tokens' own docstring-comment above), so nothing dilutes that ceiling
    # and the measured ratio should land at, not below, 8x.
    compressor = KVQuantCompressor(bits=2, outlier_std_threshold=2.5)
    original = _tokens(32, 8)
    compressed = compressor.compress(original)
    num_outliers = len(compressed.payload["outliers"])
    total_elements = compressed.original_shape[0] * compressed.original_shape[1]
    quantized_elements = total_elements - num_outliers
    effective_bit_width = (2 * quantized_elements + 16 * num_outliers) / total_elements
    ratio = compression_ratio(
        original_bit_width=16, original_element_count=total_elements,
        compressed_bit_width=effective_bit_width, compressed_element_count=total_elements,
    )
    assert 7.5 <= ratio <= 8.0


def test_kvquant_reconstruction_error_beats_kivis_at_equal_bit_width_when_outliers_are_present():
    # The whole point of explicit outlier preservation: at the SAME bit
    # width, KIVI's uniform min-max scale gets stretched by an outlier
    # (every other value in that group loses precision to make room for
    # it), while KVQuant pulls the outlier out first and quantizes the
    # remaining inliers over their own, much narrower range.
    n, channels = 32, 4
    base = [[float((t * 7 + c * 3) % 23) - 11.0 for c in range(channels)] for t in range(n)]
    for row in (3, 19):
        base[row][1] = 500.0

    kivi = KIVICompressor(group_size=32, bits=4)
    kivi_reconstructed = kivi.decompress(kivi.compress(base))
    kivi_error = reconstruction_error(base, kivi_reconstructed)

    kvquant = KVQuantCompressor(bits=4, outlier_std_threshold=2.5)
    kvquant_reconstructed = kvquant.decompress(kvquant.compress(base))
    kvquant_error = reconstruction_error(base, kvquant_reconstructed)

    assert kvquant_error < kivi_error * 0.5


def test_decompress_rejects_a_payload_from_a_different_method():
    compressor = KVQuantCompressor(bits=4)
    with pytest.raises(ValueError):
        compressor.decompress(
            CompressedKV(method="not-kvquant", payload={}, original_shape=(1, 1))
        )
