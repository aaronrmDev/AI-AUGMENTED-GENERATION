from abc import ABC, abstractmethod

from src.cag.domain.entities import CompressedKV


class KVCacheCompressor(ABC):
    # Operates on a single layer's KV tensor (rows = tokens, columns =
    # channels). KIVI, KVQuant, PALU, and ShadowKV all compress one
    # tensor in isolation -- MiniCache does not (see
    # CrossLayerKVCompressor below) and is deliberately not one of this
    # port's implementations.
    @abstractmethod
    def compress(self, kv: list[list[float]]) -> CompressedKV: ...

    @abstractmethod
    def decompress(self, compressed: CompressedKV) -> list[list[float]]: ...


class CrossLayerKVCompressor(ABC):
    # MiniCache's own shape: it merges TWO adjacent layers' KV tensors
    # into one shared representation via SLERP, not a per-layer
    # operation the single-tensor KVCacheCompressor port could express.
    @abstractmethod
    def compress(
        self, layer_a: list[list[float]], layer_b: list[list[float]]
    ) -> CompressedKV: ...

    @abstractmethod
    def decompress(
        self, compressed: CompressedKV
    ) -> tuple[list[list[float]], list[list[float]]]: ...
