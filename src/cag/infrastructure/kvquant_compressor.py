import bisect
import statistics
from typing import Any

from src.cag.domain.entities import CompressedKV
from src.cag.domain.ports import KVCacheCompressor

_METHOD = "kvquant"


class KVQuantCompressor(KVCacheCompressor):
    # KVQuant's two genuinely single-tensor-testable characteristics against
    # KIVI's sibling implementation (CAG.md's third detail -- quantizing
    # before the RoPE positional transform -- is a timing property of a
    # full attention pipeline, not something expressible on a standalone
    # [tokens x channels] tensor with no model around it; KIVI's own file
    # already disclosed a comparable deliberate simplification for
    # per-token grouping, so this file discloses its own instead of
    # pretending to implement what isn't implementable here):
    #
    # 1. Non-uniform, quantile-based (equal-population) bucket boundaries
    #    per channel, in place of KIVI's uniform linear scale/zero-point --
    #    boundaries are set from the inlier values' own order statistics,
    #    so buckets pack tightly wherever the data actually clusters
    #    instead of assuming an even spread across the min-max range.
    # 2. Explicit per-channel outlier preservation: any value more than
    #    `outlier_std_threshold` standard deviations from that channel's
    #    own mean is pulled out and kept at full precision, addressed by
    #    (row, channel), rather than being left in to stretch the
    #    quantile boundaries the way a single extreme value would stretch
    #    a naive min-max range.
    #
    # Because outliers no longer force the quantized range wide, a given
    # reconstruction-fidelity target is reachable at a lower bit width than
    # KIVI's uniform scheme needs for the same fidelity -- CAG.md's "up to
    # 8x" (vs KIVI's "~4x") is a consequence of that headroom, not an
    # independent claim baked into a different ratio formula. The
    # compression-ratio arithmetic (original bits / compressed bits) is the
    # same one KIVI's own file uses; see this file's own test for the bit
    # width that actually reaches the 8x ceiling.
    def __init__(self, bits: int = 4, outlier_std_threshold: float = 2.5) -> None:
        if not (1 <= bits <= 16):
            raise ValueError("bits must be between 1 and 16")
        if outlier_std_threshold <= 0:
            raise ValueError("outlier_std_threshold must be positive")
        self._bits = bits
        self._num_levels = 2**bits
        self._outlier_std_threshold = outlier_std_threshold

    def compress(self, kv: list[list[float]]) -> CompressedKV:
        if not kv:
            raise ValueError("kv must be non-empty")
        num_tokens = len(kv)
        num_channels = len(kv[0])
        if any(len(row) != num_channels for row in kv):
            raise ValueError("every row must have the same number of channels")

        codes: list[list[int]] = [[-1] * num_channels for _ in range(num_tokens)]
        codebooks: list[list[float]] = []
        outliers: dict[tuple[int, int], float] = {}

        for channel in range(num_channels):
            channel_values = [row[channel] for row in kv]
            channel_mean = statistics.fmean(channel_values)
            # Population stdev (not sample): this is a description of the
            # channel's own observed values, not an estimate of some larger
            # population, and it stays well-defined (0.0) for a single row.
            channel_spread = statistics.pstdev(channel_values)

            outlier_rows: set[int] = set()
            if channel_spread > 0:
                for row_index, value in enumerate(channel_values):
                    if abs(value - channel_mean) > self._outlier_std_threshold * channel_spread:
                        outlier_rows.add(row_index)
                        outliers[(row_index, channel)] = value

            inlier_values = [
                channel_values[row_index]
                for row_index in range(num_tokens)
                if row_index not in outlier_rows
            ]
            boundaries, codebook = self._build_quantile_buckets(inlier_values, channel_mean)
            codebooks.append(codebook)

            for row_index in range(num_tokens):
                if row_index in outlier_rows:
                    continue
                bucket = bisect.bisect_right(boundaries, channel_values[row_index])
                codes[row_index][channel] = min(bucket, self._num_levels - 1)

        payload: dict[str, Any] = {
            "bits": self._bits,
            "num_levels": self._num_levels,
            "codes": codes,
            "codebooks": codebooks,
            "outliers": outliers,
            "outlier_std_threshold": self._outlier_std_threshold,
        }
        return CompressedKV(
            method=_METHOD, payload=payload, original_shape=(num_tokens, num_channels)
        )

    def _build_quantile_buckets(
        self, inlier_values: list[float], channel_mean: float
    ) -> tuple[list[float], list[float]]:
        # Equal-POPULATION boundaries: split the sorted inlier values into
        # `num_levels` groups of roughly equal count using order statistics,
        # rather than assuming the values are spread evenly across their
        # own min-max range the way an equal-WIDTH (uniform) scheme would.
        if not inlier_values:
            return [], [channel_mean] * self._num_levels

        sorted_values = sorted(inlier_values)
        count = len(sorted_values)
        boundaries = [
            sorted_values[min(round(level * count / self._num_levels), count - 1)]
            for level in range(1, self._num_levels)
        ]

        buckets: list[list[float]] = [[] for _ in range(self._num_levels)]
        for value in sorted_values:
            bucket = min(bisect.bisect_right(boundaries, value), self._num_levels - 1)
            buckets[bucket].append(value)

        # Each bucket decodes to the mean of the real values actually routed
        # there -- the closest achievable representative of that bucket's
        # own population, not an arbitrary bucket-center the way an
        # equal-width scheme would use. A bucket nothing landed in (fewer
        # distinct inlier values than levels) falls back to the channel
        # mean; nothing ever decodes to it, since no code points there.
        codebook = [statistics.fmean(bucket) if bucket else channel_mean for bucket in buckets]
        return boundaries, codebook

    def decompress(self, compressed: CompressedKV) -> list[list[float]]:
        if compressed.method != _METHOD:
            raise ValueError(f"expected a '{_METHOD}' payload, got '{compressed.method}'")
        payload = compressed.payload
        codes: list[list[int]] = payload["codes"]
        codebooks: list[list[float]] = payload["codebooks"]
        outliers: dict[tuple[int, int], float] = payload["outliers"]

        rows: list[list[float]] = []
        for row_index, code_row in enumerate(codes):
            row: list[float] = []
            for channel, code in enumerate(code_row):
                if (row_index, channel) in outliers:
                    row.append(outliers[(row_index, channel)])
                else:
                    row.append(codebooks[channel][code])
            rows.append(row)
        return rows
