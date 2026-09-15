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
class ConversationTurn:
    # One turn of a session. parent_turn_id is what lets a workload be a
    # BRANCHING agent program rather than only a flat conversation --
    # SGLang's whole distinguishing claim is about reuse across a
    # program's control flow, which a linear-only model would be unable
    # to show, and would therefore silently understate.
    turn_id: str
    new_tokens: int
    parent_turn_id: str | None = None


@dataclass(frozen=True)
class SessionRun:
    # per_turn_recompute is kept alongside the total deliberately: the
    # claim under test is that per-turn cost STOPS GROWING, and a total
    # alone cannot distinguish "halved the quadratic" from "made it
    # linear". transfer_tokens is non-zero only for strategies whose
    # cache lives somewhere the compute does not.
    strategy: str
    per_turn_recompute: list[int] = field(default_factory=list)
    tokens_recomputed: int = 0
    peak_tokens_held: int = 0
    transfer_tokens: int = 0


@dataclass(frozen=True)
class OffloadWorkload:
    # One layer-by-layer forward pass over a tiered KV store. Times are
    # in milliseconds and are PARAMETERS of the simulation, not measured
    # hardware constants -- the comparison between strategies is what
    # this model is for, and every report using it says so rather than
    # presenting simulated milliseconds as if they came off a device.
    num_layers: int
    compute_ms_per_layer: float
    warm_fetch_ms_per_layer: float
    cold_fetch_ms_per_layer: float
    gpu_layer_capacity: int


@dataclass(frozen=True)
class OffloadRun:
    # transfer_ms_issued is all the transfer work a strategy asked for;
    # transfer_ms_hidden is however much of it finished underneath
    # compute and so never reached the critical path. Keeping both,
    # rather than only the total, is what makes overlap efficiency
    # measurable instead of inferred.
    strategy: str
    total_ms: float = 0.0
    transfer_ms_issued: float = 0.0
    transfer_ms_hidden: float = 0.0
    gpu_layers_resident: int = 0


@dataclass(frozen=True)
class BatchRequest:
    # A pending request as the scheduling stage sees it: an identity and
    # the token sequence whose leading blocks may or may not be shareable
    # with its neighbours. Deliberately not the prompt string -- prefix
    # sharing happens over tokens at block granularity, and comparing
    # strings would silently disagree with the engine about where a
    # shared prefix actually ends.
    request_id: str
    tokens: tuple[int, ...] = ()


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
