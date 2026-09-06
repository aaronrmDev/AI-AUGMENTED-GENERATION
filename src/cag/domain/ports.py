from abc import ABC, abstractmethod

from src.cag.domain.entities import CompressedKV, EvictionDecision, VerificationResult


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


class KVCacheEvictor(ABC):
    # H2O, SnapKV, and NACL all answer the same question from the same
    # shape of input: given a per-token importance score (accumulated
    # attention for H2O, windowed-and-pooled for SnapKV, an end-of-input
    # proxy for NACL) and a token budget, which indices survive. What
    # differs is how attention_scores itself got computed upstream and
    # what each does with it below -- not this port's signature.
    @abstractmethod
    def select_keep_indices(
        self, attention_scores: list[float], budget: int
    ) -> EvictionDecision: ...


class RecentPatternEvictor(ABC):
    # MorphKV's own shape: it scores tokens from several recent decoding
    # steps' attention distributions via Sum/Max Fusion, not a single
    # already-accumulated vector -- a genuinely different input shape
    # from KVCacheEvictor above, the same reasoning that keeps MiniCache
    # off the single-tensor KVCacheCompressor port.
    @abstractmethod
    def select_keep_indices(
        self, recent_attention_windows: list[list[float]], budget: int
    ) -> EvictionDecision: ...


class HashBasedEvictor(ABC):
    # HASHEVICT's own shape: it estimates token similarity via
    # locality-sensitive hashing over the raw KV vectors themselves,
    # before any attention computation runs at all -- there is no
    # per-token score to consume here, only vectors.
    @abstractmethod
    def select_keep_indices(
        self, kv_vectors: list[list[float]], budget: int
    ) -> EvictionDecision: ...


class CacheDistiller(ABC):
    # InfiniPot's own shape: rather than a binary per-token keep/evict
    # call, it distills the whole cache down to `budget` representative
    # rows once it overflows -- closer to selective compression of the
    # cache than to per-token selection, so it returns a reduced KV
    # tensor directly instead of an EvictionDecision's index list.
    @abstractmethod
    def distill(self, kv: list[list[float]], budget: int) -> list[list[float]]: ...


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
