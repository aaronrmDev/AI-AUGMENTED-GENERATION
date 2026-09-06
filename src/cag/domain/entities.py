from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompressedKV:
    # payload is method-specific (quantized ints + scales for KIVI/KVQuant,
    # a latent matrix + reconstruction matrix for PALU, an interpolated
    # representation for MiniCache, a composite for ShadowKV) -- a single
    # shared dataclass shape across five genuinely different compressed
    # representations would force an artificial uniformity none of them
    # actually share, the same reasoning that keeps MiniCache off the
    # single-tensor KVCacheCompressor port entirely.
    method: str
    payload: dict[str, Any] = field(default_factory=dict)
    original_shape: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class VerificationResult:
    # What one real target-model forward pass over (tokens + candidates)
    # produces: the accepted prefix (up to the first candidate that
    # didn't match the target's own greedy prediction) plus one bonus
    # token -- the target's own prediction at the mismatch point, free
    # since it came from the same forward pass that verified the
    # candidates. bonus_token is None only when max_new_tokens was
    # already reached before verification (nothing left to bonus into).
    accepted_tokens: list[int] = field(default_factory=list)
    bonus_token: int | None = None


@dataclass(frozen=True)
class EvictionDecision:
    # keep_indices is sorted ascending, a subset of range(original_token_count)
    # -- every one of CAG.md's six eviction algorithms differs only in WHICH
    # indices survive a given budget (attention-accumulated, windowed+pooled,
    # proxy-scored, recent-pattern-fused, hash-bucketed), so the actual KV
    # row selection this decision drives is a plain index operation none of
    # them needs to own individually.
    method: str
    keep_indices: list[int] = field(default_factory=list)
    evicted_count: int = 0


@dataclass(frozen=True)
class SpeculativeDecodingRun:
    # A real, measured record of one full generation run -- everything
    # needed to compute acceptance rate (tokens_accepted_from_candidates
    # / tokens_proposed) and forward-pass reduction (forward_passes vs.
    # len(generated_tokens), the naive-autoregressive baseline) directly,
    # without re-deriving either from the raw token stream.
    generated_tokens: list[int] = field(default_factory=list)
    forward_passes: int = 0
    tokens_accepted_from_candidates: int = 0
    tokens_proposed: int = 0
