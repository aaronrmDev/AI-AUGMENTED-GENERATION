import math
from typing import TypedDict, cast

from src.cag.domain.entities import CompressedKV
from src.cag.domain.ports import CrossLayerKVCompressor

_METHOD = "minicache"


class _MiniCachePayload(TypedDict):
    # Same reasoning as KIVI's sibling _KIVIPayload: CompressedKV.payload
    # stays dict[str, Any] at the domain boundary, but within this file a
    # TypedDict gives mypy something real to check a future key rename
    # against, on either side of the compress()/decompress() pair.
    shared_directions: list[list[float]]
    magnitudes_a: list[float]
    magnitudes_b: list[float]

# Below this, sin(theta) is close enough to zero that plain SLERP's division
# would be dividing by a value indistinguishable from zero in floating
# point. Linear interpolation is what SLERP mathematically converges to as
# theta -> 0, so falling back to it here is exact in the limit, not an
# approximation of a different formula.
_NEAR_PARALLEL_SIN_THETA = 1e-6


class MiniCacheCompressor(CrossLayerKVCompressor):
    # Cross-layer compression (CAG.md: "SLERP... between two adjacent
    # layers' similar KV states"). Each token row in each layer is
    # decomposed into a magnitude (Euclidean norm) and a unit direction.
    # The two layers' directions for the same token are SLERP-interpolated
    # at t=0.5 into ONE shared direction; only the cheap per-layer scalar
    # magnitudes stay separate. Reconstructing a layer multiplies its own
    # magnitude back onto the shared direction, which necessarily loses
    # whatever difference existed between the two layers' original
    # directions -- honest lossy compression, not a bug.
    def compress(self, layer_a: list[list[float]], layer_b: list[list[float]]) -> CompressedKV:
        if not layer_a or not layer_b:
            raise ValueError("layer_a and layer_b must be non-empty")
        if len(layer_a) != len(layer_b):
            raise ValueError("layer_a and layer_b must have the same number of tokens")
        num_channels = len(layer_a[0])
        for row in (*layer_a, *layer_b):
            if len(row) != num_channels:
                raise ValueError("all rows in both layers must have the same number of channels")

        shared_directions: list[list[float]] = []
        magnitudes_a: list[float] = []
        magnitudes_b: list[float] = []
        for row_a, row_b in zip(layer_a, layer_b, strict=True):
            magnitude_a = _norm(row_a)
            magnitude_b = _norm(row_b)
            direction_a = _unit(row_a, magnitude_a)
            direction_b = _unit(row_b, magnitude_b)
            shared_directions.append(_slerp(direction_a, direction_b, 0.5))
            magnitudes_a.append(magnitude_a)
            magnitudes_b.append(magnitude_b)

        payload: _MiniCachePayload = {
            "shared_directions": shared_directions,
            "magnitudes_a": magnitudes_a,
            "magnitudes_b": magnitudes_b,
        }
        return CompressedKV(
            method=_METHOD, payload=dict(payload), original_shape=(len(layer_a), num_channels)
        )

    def decompress(self, compressed: CompressedKV) -> tuple[list[list[float]], list[list[float]]]:
        if compressed.method != _METHOD:
            raise ValueError(f"expected a '{_METHOD}' payload, got '{compressed.method}'")
        payload = cast(_MiniCachePayload, compressed.payload)
        shared_directions = payload["shared_directions"]
        magnitudes_a = payload["magnitudes_a"]
        magnitudes_b = payload["magnitudes_b"]
        layer_a = [
            [magnitude * component for component in direction]
            for magnitude, direction in zip(magnitudes_a, shared_directions, strict=True)
        ]
        layer_b = [
            [magnitude * component for component in direction]
            for magnitude, direction in zip(magnitudes_b, shared_directions, strict=True)
        ]
        return layer_a, layer_b


def _norm(row: list[float]) -> float:
    return math.sqrt(sum(value * value for value in row))


def _unit(row: list[float], magnitude: float) -> list[float]:
    # A zero-magnitude row has no defined direction -- return the zero
    # vector itself rather than dividing by zero. This keeps the SLERP
    # dot-product/arccos machinery below well-defined (a dot product with
    # a zero vector is always 0) instead of raising or producing NaN.
    if magnitude == 0.0:
        return [0.0] * len(row)
    return [value / magnitude for value in row]


def _slerp(u: list[float], v: list[float], t: float) -> list[float]:
    dot = sum(a * b for a, b in zip(u, v, strict=True))
    dot = max(-1.0, min(1.0, dot))  # guard acos's domain against float rounding past +-1
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if sin_theta < _NEAR_PARALLEL_SIN_THETA:
        return [(1 - t) * a + t * b for a, b in zip(u, v, strict=True)]
    coefficient_u = math.sin((1 - t) * theta) / sin_theta
    coefficient_v = math.sin(t * theta) / sin_theta
    return [coefficient_u * a + coefficient_v * b for a, b in zip(u, v, strict=True)]
