from abc import ABC, abstractmethod

from src.cag.domain.entities import CompressedKV, VerificationResult


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


class CandidateGenerator(ABC):
    # Medusa, Lookahead Decoding, and Prompt Lookup Decoding all differ
    # only in WHERE candidate tokens come from -- the propose-verify-
    # accept loop underneath is identical (SpeculativeDecode owns it).
    # Takes both the prompt and the generated-so-far tail since the
    # three variants draw from different slices of that same context
    # (Prompt Lookup: the prompt; Lookahead: the generated tail; Medusa:
    # neither -- its candidates come from the target model's own extra
    # heads, but the uniform signature costs it nothing to ignore what
    # it doesn't need).
    @abstractmethod
    def propose(
        self, prompt_tokens: list[int], generated_tokens: list[int], num_candidates: int
    ) -> list[int]: ...


class TargetModel(ABC):
    # One real forward pass over (tokens + candidates): compares each
    # candidate position's greedy prediction against what the target
    # model itself would have produced there, accepts the prefix up to
    # the first mismatch, and returns it plus the bonus token the
    # target's own distribution at the mismatch point gives for free.
    @abstractmethod
    def verify_candidates(
        self, tokens: list[int], candidates: list[int]
    ) -> VerificationResult: ...
