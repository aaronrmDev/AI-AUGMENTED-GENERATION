from typing import TypedDict, cast

from src.cag.domain.entities import CompressedKV
from src.cag.domain.ports import KVCacheCompressor

_METHOD = "kivi"


class _KIVIPayload(TypedDict):
    # CompressedKV.payload stays dict[str, Any] at the domain boundary
    # (five compressors, five genuinely different shapes -- see its own
    # comment), but within this file a TypedDict gives mypy something
    # real to check: compress()'s dict literal is verified against this
    # exact key set at construction, and decompress() reads through the
    # same type, so a future rename on one side without the other is a
    # real mypy error here instead of a runtime KeyError the first time
    # that path actually executes (review-caught, low severity but a
    # cheap, safe fix).
    quantized_groups: list[list[list[int]]]
    scales: list[list[float]]
    zero_points: list[list[float]]
    residual_rows: list[list[float]]
    group_size: int


class KIVICompressor(KVCacheCompressor):
    # Per-channel quantization (CAG.md: "per-channel quantization to
    # keys"), the harder and more KIVI-specific of the two axes the
    # source describes (per-token, for values, is a plain transpose of
    # the same mechanism and isn't separately implemented here). Tokens
    # are grouped by `group_size`; every full group gets quantized
    # per-channel, and a trailing partial group -- not yet large enough
    # to cross the grouping threshold -- stays in full precision as a
    # residual, matching "keep recent tokens in full precision... merge
    # residuals into a group after threshold."
    def __init__(self, group_size: int = 32, bits: int = 4) -> None:
        if group_size < 1:
            raise ValueError("group_size must be at least 1")
        if not (1 <= bits <= 16):
            raise ValueError("bits must be between 1 and 16")
        self._group_size = group_size
        self._levels = (2**bits) - 1

    def compress(self, kv: list[list[float]]) -> CompressedKV:
        if not kv:
            raise ValueError("kv must be non-empty")
        num_channels = len(kv[0])
        quantized_groups: list[list[list[int]]] = []
        scales: list[list[float]] = []
        zero_points: list[list[float]] = []
        residual_rows: list[list[float]] = []

        for group_start in range(0, len(kv), self._group_size):
            group = kv[group_start : group_start + self._group_size]
            if len(group) < self._group_size:
                residual_rows = [list(row) for row in group]
                break
            group_scales = []
            group_zero_points = []
            quantized_group: list[list[int]] = [[0] * num_channels for _ in group]
            for channel in range(num_channels):
                channel_values = [row[channel] for row in group]
                channel_min = min(channel_values)
                channel_max = max(channel_values)
                # A flat channel (all values equal) would otherwise
                # divide by zero -- treat it as needing no precision at
                # all beyond the single value itself.
                channel_range = channel_max - channel_min
                scale = channel_range / self._levels if channel_range > 0 else 1.0
                group_scales.append(scale)
                group_zero_points.append(channel_min)
                for row_index, value in enumerate(channel_values):
                    quantized_group[row_index][channel] = round((value - channel_min) / scale)
            quantized_groups.append(quantized_group)
            scales.append(group_scales)
            zero_points.append(group_zero_points)

        payload: _KIVIPayload = {
            "quantized_groups": quantized_groups,
            "scales": scales,
            "zero_points": zero_points,
            "residual_rows": residual_rows,
            "group_size": self._group_size,
        }
        return CompressedKV(
            method=_METHOD, payload=dict(payload), original_shape=(len(kv), num_channels)
        )

    def decompress(self, compressed: CompressedKV) -> list[list[float]]:
        if compressed.method != _METHOD:
            raise ValueError(f"expected a '{_METHOD}' payload, got '{compressed.method}'")
        payload = cast(_KIVIPayload, compressed.payload)
        rows: list[list[float]] = []
        for quantized_group, group_scales, group_zero_points in zip(
            payload["quantized_groups"], payload["scales"], payload["zero_points"], strict=True
        ):
            for quantized_row in quantized_group:
                rows.append(
                    [
                        group_zero_points[channel] + quantized_row[channel] * group_scales[channel]
                        for channel in range(len(quantized_row))
                    ]
                )
        rows.extend(list(row) for row in payload["residual_rows"])
        return rows
