"""Live verification of CAG.md's Archetype A, the Long Context Champion:
KV Cache Eviction -> KV Cache Compression -> Hybrid Offloading, composed in
that order over a real distilgpt2 KV tensor and real attention scores.

Runs on CPU with no GPU or vLLM, exactly as the Eviction and Compression
batches did, so unlike the other combination test in this batch it DOES run
in CI.

The claim under test is CAG.md's structural one -- "each step reduces what
the next step has to handle" -- and it is tested two ways, because the
obvious way is not sufficient. A pipeline reducing toward a fixed GPU
capacity makes the naive product telescope: the stage ratios multiply out to
original/capacity no matter how the stages divide the labour, so a synergy
score of 1.0 there says little. The load-bearing test is therefore
interference drift, which compares each stage's ratio inside the pipeline
against the same stage run alone and cannot telescope.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cag.application.long_context_pipeline import LongContextPipeline
from src.cag.domain.combination_metrics import interference_drift
from src.cag.domain.eviction_metrics import accumulate_attention_scores
from src.cag.domain.payload_footprint import dense_bits, payload_bits
from src.cag.infrastructure.h2o_evictor import H2OEvictor
from src.cag.infrastructure.kivi_compressor import KIVICompressor
from src.cag.infrastructure.layerkv_offloader import LayerKVOffloader
from src.cag.infrastructure.palu_compressor import PALUCompressor

_MODEL_ID = "distilgpt2"
_PROMPT = (
    "The following is a detailed technical discussion of distributed systems, "
    "consensus protocols, replication strategies, and fault tolerance as they "
    "appear in large scale production infrastructure over many years. "
)


def _real_cache_and_scores() -> tuple[list[list[float]], list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, attn_implementation="eager")
    model.eval()
    input_ids = tokenizer(_PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model(input_ids, use_cache=True, output_attentions=True)

    keys = output.past_key_values.layers[0].keys
    _, num_heads, seq_len, head_dim = keys.shape
    kv = keys[0].permute(1, 0, 2).reshape(seq_len, num_heads * head_dim).tolist()
    scores = accumulate_attention_scores(output.attentions[0][0].mean(dim=0).tolist())
    return kv, scores


def test_each_stage_hands_the_next_one_less_of_a_real_cache_to_handle():
    kv, scores = _real_cache_and_scores()
    budget = max(4, len(kv) // 2)
    result = LongContextPipeline(
        evictor=H2OEvictor(recent_window=4),
        compressor=KIVICompressor(group_size=8, bits=4),
        offloader=LayerKVOffloader(),
    ).execute(kv=kv, attention_scores=scores, token_budget=budget, gpu_layer_capacity=4)

    print(
        f"\nreal distilgpt2 cache: {result.original_tokens} tokens -> "
        f"{result.tokens_after_eviction} after eviction -> "
        f"{result.slots_after_compression:.2f} slots after compression -> "
        f"{result.gpu_resident_slots:.0f} GPU-resident\n"
        f"stage reductions: {[round(r, 2) for r in result.stage_reductions]}"
    )
    assert result.tokens_after_eviction < result.original_tokens
    assert result.slots_after_compression < result.tokens_after_eviction
    assert result.gpu_resident_slots <= result.slots_after_compression


def test_quantization_barely_notices_that_eviction_ran_before_it():
    # The non-telescoping test. Quantization is per-value, so removing
    # tokens should barely change how well the survivors quantize.
    #
    # On synthetic data whose token count divides evenly into KIVI's
    # groups this measures exactly 0.0% drift, and an earlier version of
    # this test asserted that. Against a real 41-token cache it measures
    # +5.0%, because KIVI keeps a trailing partial group at full
    # precision and eviction changes what fraction of the sequence that
    # residual represents. The drift is a boundary effect of the group
    # size, not a failure of the two stages to compose -- which is why
    # the bound here is loose enough to admit it and still an order of
    # magnitude tighter than the low-rank case below.
    kv, scores = _real_cache_and_scores()
    compressor = KIVICompressor(group_size=8, bits=4)
    channels = len(kv[0])

    def ratio(rows: list[list[float]]) -> float:
        compressed = compressor.compress(rows)
        return dense_bits(len(rows), channels) / payload_bits(compressed.payload, 4)

    standalone = ratio(kv)
    keep = H2OEvictor(recent_window=4).select_keep_indices(scores, len(kv) // 2).keep_indices
    in_pipeline = ratio([kv[index] for index in keep])

    drift = interference_drift(standalone, in_pipeline)
    print(f"KIVI standalone {standalone:.2f}x -> {in_pipeline:.2f}x, drift {drift:+.1%}")
    assert abs(drift) < 0.10


def test_low_rank_compression_measurably_interferes_with_eviction_on_a_real_cache():
    # The counter-case, and the reason a single blanket claim about this
    # archetype would be wrong. PALU's ratio depends on sequence length
    # relative to its rank, so the shorter the sequence eviction leaves,
    # the less it saves -- the two stages genuinely interfere, and the
    # naive product of their standalone ratios overstates the pipeline.
    kv, scores = _real_cache_and_scores()
    compressor = PALUCompressor(rank=4)
    channels = len(kv[0])

    def ratio(rows: list[list[float]]) -> float:
        compressed = compressor.compress(rows)
        return dense_bits(len(rows), channels) / payload_bits(compressed.payload, 4)

    standalone = ratio(kv)
    keep = H2OEvictor(recent_window=4).select_keep_indices(scores, len(kv) // 3).keep_indices
    in_pipeline = ratio([kv[index] for index in keep])

    drift = interference_drift(standalone, in_pipeline)
    print(f"PALU standalone {standalone:.2f}x -> {in_pipeline:.2f}x, drift {drift:+.1%}")
    assert drift < -0.02, "low-rank compression should lose ratio on a shortened sequence"
