from collections.abc import Iterable
from typing import TypedDict, cast

from src.cag.domain.entities import CompressedKV
from src.cag.domain.ports import KVCacheCompressor
from src.cag.infrastructure.palu_compressor import PALUCompressor, PALUPayload

_METHOD = "shadowkv"


class _ShadowKVPayload(TypedDict):
    # Same reasoning as KIVI's sibling _KIVIPayload -- mypy checks this
    # file's own compress()/decompress() key usage against one shared
    # type. palu_B is read straight from PALUPayload's own "B" field
    # (typed via that TypedDict at the read site below, not re-declared
    # here), the actual cross-file link this file's composition relies on.
    palu_B: list[list[float]]
    sparse_h_entries: list[tuple[int, int, int]]
    h_shape: tuple[int, int]
    scale: float
    zero_point: float


class ShadowKVCompressor(KVCacheCompressor):
    # CAG.md: "ShadowKV combines several of the above ideas into a single
    # hybrid strategy: it offloads value tensors elsewhere, keeps a
    # low-rank representation of keys, and layers sparsity and
    # quantization on top of both -- reaching roughly 6x compression by
    # stacking mechanisms rather than relying on any one of them alone."
    # The "offloads value tensors elsewhere" half is a serving-engine
    # placement decision (where bytes physically live), not something a
    # standalone tensor-in, tensor-out compressor can express -- same
    # disclosed-simplification pattern KIVI, KVQuant, and PALU's own
    # files already used for the parts of their source description that
    # don't survive the single-tensor port's scope. What IS genuinely
    # implemented and stacked here: PALU's real low-rank decomposition
    # (composed directly, not reimplemented) for "keeps a low-rank
    # representation," then real magnitude-based sparsity plus real
    # uniform quantization applied to that low-rank latent for "layers
    # sparsity and quantization on top."
    def __init__(self, rank: int = 4, sparsity_ratio: float = 0.5, bits: int = 4) -> None:
        if not (0.0 <= sparsity_ratio <= 1.0):
            raise ValueError("sparsity_ratio must be between 0.0 and 1.0")
        if not (1 <= bits <= 16):
            raise ValueError("bits must be between 1 and 16")
        self._palu = PALUCompressor(rank=rank)
        self._sparsity_ratio = sparsity_ratio
        self._bits = bits
        self._levels = (2**bits) - 1

    def compress(self, kv: list[list[float]]) -> CompressedKV:
        palu_compressed = self._palu.compress(kv)
        palu_payload = cast(PALUPayload, palu_compressed.payload)
        h = palu_payload["H"]
        b = palu_payload["B"]
        num_rows = len(h)
        num_cols = len(h[0]) if h else 0

        # Rank every (row, col) entry of H by |value|, drop the smallest
        # sparsity_ratio fraction -- the entries contributing the least
        # to H's own magnitude are the ones a real sparse representation
        # would omit first.
        flat_entries = [
            (row, col, h[row][col]) for row in range(num_rows) for col in range(num_cols)
        ]
        flat_entries.sort(key=lambda entry: abs(entry[2]))
        drop_count = round(len(flat_entries) * self._sparsity_ratio)
        surviving = flat_entries[drop_count:]

        scale, zero_point = self._quantization_params(entry[2] for entry in surviving)
        sparse_h_entries = [
            (row, col, self._quantize(value, scale, zero_point))
            for row, col, value in surviving
        ]

        payload: _ShadowKVPayload = {
            "palu_B": b,
            "sparse_h_entries": sparse_h_entries,
            "h_shape": (num_rows, num_cols),
            "scale": scale,
            "zero_point": zero_point,
        }
        return CompressedKV(
            method=_METHOD, payload=dict(payload), original_shape=palu_compressed.original_shape
        )

    def decompress(self, compressed: CompressedKV) -> list[list[float]]:
        if compressed.method != _METHOD:
            raise ValueError(f"expected a '{_METHOD}' payload, got '{compressed.method}'")
        payload = cast(_ShadowKVPayload, compressed.payload)
        num_rows, num_cols = payload["h_shape"]
        scale = payload["scale"]
        zero_point = payload["zero_point"]

        h = [[0.0] * num_cols for _ in range(num_rows)]
        for row, col, quantized_value in payload["sparse_h_entries"]:
            h[row][col] = self._dequantize(quantized_value, scale, zero_point)

        b: list[list[float]] = payload["palu_B"]
        return [
            [sum(h[row][k] * b[k][col] for k in range(num_cols)) for col in range(len(b[0]))]
            for row in range(num_rows)
        ]

    def _quantization_params(self, values: Iterable[float]) -> tuple[float, float]:
        values = list(values)
        if not values:
            return 1.0, 0.0
        low, high = min(values), max(values)
        value_range = high - low
        scale = value_range / self._levels if value_range > 0 else 1.0
        return scale, low

    def _quantize(self, value: float, scale: float, zero_point: float) -> int:
        return round((value - zero_point) / scale)

    def _dequantize(self, quantized_value: int, scale: float, zero_point: float) -> float:
        return zero_point + quantized_value * scale
