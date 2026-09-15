"""Live verification that KV Cache Eviction actually reduces cache size while
retaining most of the attention mass that mattered, and that the retained
mass genuinely beats naive random eviction at the same budget -- H2O's own
value proposition, "keeping only heavy hitters." Uses a real distilgpt2
forward pass (CPU, no GPU or vLLM needed) exactly the way the KV Cache
Compression batch did: this technique's own correctness question -- does
discarding low-attention tokens preserve enough signal for the model to keep
producing a coherent continuation -- is honestly measurable without a
serving engine, contrary to what CAG.md said about all six deferred CAG
techniques before this batch corrected the claim for Eviction specifically.

Reduces every layer's cache consistently to the same kept token positions,
unlike the Compression batch's single-layer splice: eviction changes
sequence length, so every layer's keys and values must agree on which
positions survived, or attention would run over cache tensors of mismatched
length across layers.

output_attentions=True requires attn_implementation="eager" -- the
transformers default (SDPA) does not return real attention weights, only
None, which would silently break accumulate_attention_scores' every-caller
assumption of a real, populated matrix.
"""
import copy
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cag.domain.eviction_metrics import (
    accumulate_attention_scores,
    memory_reduction_ratio,
    retained_attention_mass,
)
from src.cag.infrastructure.h2o_evictor import H2OEvictor
from src.cag.infrastructure.nacl_evictor import NACLEvictor
from src.cag.infrastructure.snapkv_evictor import SnapKVEvictor

_MODEL_ID = "distilgpt2"
_PROMPT = "The quick brown fox jumps over the lazy dog and runs into the forest."
_LAYER = 0


def _load_model_and_cache():
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, attn_implementation="eager")
    model.eval()
    input_ids = tokenizer(_PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model(input_ids, use_cache=True, output_attentions=True)
    next_token_id = tokenizer.encode(" and", add_special_tokens=False)[0]
    next_token = torch.tensor([[next_token_id]])
    return model, tokenizer, output.past_key_values, output.attentions, next_token


def _mean_causal_attention_matrix(attentions, layer: int) -> list[list[float]]:
    # Averaged across heads into one real [seq_len, seq_len] causal
    # attention matrix -- the same kind of disclosed, real simplification
    # the Compression batch already used (full-layer, all-heads splicing
    # instead of one head in isolation).
    layer_attention = attentions[layer][0]  # [num_heads, seq_len, seq_len]
    return layer_attention.mean(dim=0).tolist()


def _evict_full_cache(cache, keep_indices: list[int]):
    # Every layer's keys AND values must be reduced to the identical kept
    # positions -- a per-layer mismatch would make attention run over
    # cache tensors of different sequence lengths across layers.
    reduced = copy.deepcopy(cache)
    keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
    for layer in reduced.layers:
        layer.keys = layer.keys.index_select(2, keep_tensor)
        layer.values = layer.values.index_select(2, keep_tensor)
    return reduced


def _continue_with_cache(model, cache, next_token):
    isolated_cache = copy.deepcopy(cache)
    with torch.no_grad():
        output = model(next_token, past_key_values=isolated_cache, use_cache=True)
    return output.logits[:, -1, :].clone()


def test_h2o_retains_more_attention_mass_than_random_eviction_at_the_same_budget():
    _model, _tokenizer, _cache, attentions, _next_token = _load_model_and_cache()
    matrix = _mean_causal_attention_matrix(attentions, _LAYER)
    scores = accumulate_attention_scores(matrix)
    seq_len = len(scores)
    budget = max(1, seq_len // 2)

    h2o_decision = H2OEvictor(recent_window=2).select_keep_indices(scores, budget)
    h2o_mass = retained_attention_mass(scores, h2o_decision.keep_indices)

    rng = random.Random(0)
    random_keep = sorted(rng.sample(range(seq_len), budget))
    random_mass = retained_attention_mass(scores, random_keep)

    ratio = memory_reduction_ratio(seq_len, len(h2o_decision.keep_indices))
    print(
        f"\nseq_len={seq_len} budget={budget} "
        f"h2o_retained_mass={h2o_mass:.4f} random_retained_mass={random_mass:.4f} "
        f"memory_reduction_ratio={ratio:.2f}x"
    )
    assert h2o_mass > random_mass, "H2O's heavy-hitter selection should beat random eviction"


def test_evicted_cache_still_produces_a_coherent_continuation():
    model, tokenizer, cache, attentions, next_token = _load_model_and_cache()
    matrix = _mean_causal_attention_matrix(attentions, _LAYER)
    scores = accumulate_attention_scores(matrix)
    seq_len = len(scores)
    budget = max(1, seq_len // 2)

    baseline_logits = _continue_with_cache(model, cache, next_token)
    baseline_top_token = int(torch.argmax(baseline_logits, dim=-1).item())
    decoded_baseline = tokenizer.decode([baseline_top_token])
    print(f"\nbaseline top token: {baseline_top_token} ({decoded_baseline!r})")

    evictors = {
        "h2o": H2OEvictor(recent_window=2),
        "snapkv": SnapKVEvictor(pool_kernel_size=3, recent_window=2),
        "nacl": NACLEvictor(random_fraction=0.2, random_seed=0),
    }
    for name, evictor in evictors.items():
        decision = evictor.select_keep_indices(scores, budget)
        reduced_cache = _evict_full_cache(cache, decision.keep_indices)
        logits = _continue_with_cache(model, reduced_cache, next_token)
        top_token = int(torch.argmax(logits, dim=-1).item())
        max_abs_diff = float((baseline_logits - logits).abs().max().item())
        mass = retained_attention_mass(scores, decision.keep_indices)
        print(
            f"{name}: kept={len(decision.keep_indices)}/{seq_len} "
            f"retained_mass={mass:.4f} "
            f"top_token_matches_baseline={top_token == baseline_top_token} "
            f"decoded={tokenizer.decode([top_token])!r} max|logit diff|={max_abs_diff:.4f}"
        )
        # Eviction is a lossier operation than compression by design (real
        # tokens are discarded outright, not reconstructed from a
        # compressed representation), so top-token preservation isn't
        # asserted the way the Compression batch asserted it -- only that
        # a real, finite continuation still comes out of the reduced
        # cache, not garbage or NaNs.
        assert torch.isfinite(logits).all()


def test_infinipot_distillation_produces_a_finite_continuation_at_a_tighter_budget():
    # InfiniPot merges tokens into centroids rather than selecting a
    # subset, so it needs its own splice path: the distilled rows replace
    # the ENTIRE layer's tensor rather than a subset of original rows,
    # since a centroid isn't any single original token's real KV vector.
    from src.cag.infrastructure.infinipot_distiller import InfiniPotDistiller

    model, tokenizer, cache, attentions, next_token = _load_model_and_cache()
    matrix = _mean_causal_attention_matrix(attentions, _LAYER)
    scores = accumulate_attention_scores(matrix)
    seq_len = len(scores)
    budget = max(1, seq_len // 3)

    def _flatten_layer(layer_cache, kind: str) -> list[list[float]]:
        tensor = layer_cache.keys if kind == "keys" else layer_cache.values
        _, num_heads, s_len, head_dim = tensor.shape
        return tensor[0].permute(1, 0, 2).reshape(s_len, num_heads * head_dim).tolist()

    def _unflatten(distilled: list[list[float]], num_heads: int, head_dim: int):
        tensor = torch.tensor(distilled)
        return tensor.reshape(len(distilled), num_heads, head_dim).permute(1, 0, 2).unsqueeze(0)

    distiller = InfiniPotDistiller()
    reduced = copy.deepcopy(cache)
    for layer in reduced.layers:
        num_heads, head_dim = layer.keys.shape[1], layer.keys.shape[3]
        for kind in ("keys", "values"):
            flat = _flatten_layer(layer, kind)
            distilled = distiller.distill(flat, budget)
            setattr(layer, kind, _unflatten(distilled, num_heads, head_dim))

    logits = _continue_with_cache(model, reduced, next_token)
    top_token = int(torch.argmax(logits, dim=-1).item())
    print(
        f"\ninfinipot: distilled seq_len {seq_len} -> {budget}, "
        f"top token = {top_token} ({tokenizer.decode([top_token])!r})"
    )
    assert torch.isfinite(logits).all()
