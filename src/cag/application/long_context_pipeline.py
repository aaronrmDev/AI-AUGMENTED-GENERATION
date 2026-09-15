from dataclasses import dataclass, field

from src.cag.domain.entities import OffloadWorkload
from src.cag.domain.payload_footprint import dense_bits, payload_bits
from src.cag.domain.ports import KVCacheCompressor, KVCacheEvictor, OffloadingStrategy


@dataclass(frozen=True)
class LongContextResult:
    # Footprint after each stage, in token-equivalents, so the claim that
    # each stage shrinks what the next one handles can be read directly
    # off the sequence rather than inferred from a single total.
    original_tokens: int = 0
    tokens_after_eviction: int = 0
    slots_after_compression: float = 0.0
    gpu_resident_slots: float = 0.0
    stage_reductions: list[float] = field(default_factory=list)
    kept_indices: list[int] = field(default_factory=list)


class LongContextPipeline:
    # CAG.md's Archetype A, composed in the order the source insists on:
    # "eviction first reduces the token count down to essential heavy
    # hitters, compression then shrinks the per-token size of whatever
    # survived eviction, and hybrid offloading moves what's left -- now
    # both fewer tokens and smaller per token -- out to CPU or SSD."
    #
    # The order is the substance of the claim, so this composes the real
    # evictor, compressor and offloader the earlier batches built rather
    # than reimplementing any of them. If the pipeline compounds, it is
    # because those components genuinely feed each other, not because a
    # combination-specific reimplementation was tuned to make it look
    # that way.
    def __init__(
        self,
        evictor: KVCacheEvictor,
        compressor: KVCacheCompressor,
        offloader: OffloadingStrategy,
        quantized_bit_width: int = 4,
    ) -> None:
        self._evictor = evictor
        self._compressor = compressor
        self._offloader = offloader
        self._quantized_bit_width = quantized_bit_width

    def execute(
        self,
        kv: list[list[float]],
        attention_scores: list[float],
        token_budget: int,
        gpu_layer_capacity: int,
    ) -> LongContextResult:
        original_tokens = len(kv)
        if original_tokens == 0:
            raise ValueError("kv must be non-empty")

        # Stage 1 -- eviction reduces how many tokens survive at all.
        decision = self._evictor.select_keep_indices(attention_scores, token_budget)
        surviving = [kv[index] for index in decision.keep_indices]
        eviction_reduction = original_tokens / len(surviving)

        # Stage 2 -- compression shrinks what eviction left, and only
        # what eviction left: the compressor never sees the evicted rows,
        # which is exactly the "reduces what the next step has to handle"
        # relationship being tested.
        compressed = self._compressor.compress(surviving)
        before_bits = dense_bits(len(surviving), len(surviving[0]))
        after_bits = payload_bits(compressed.payload, self._quantized_bit_width)
        compression_reduction = before_bits / after_bits if after_bits else 1.0
        slots_after_compression = len(surviving) / compression_reduction

        # Stage 3 -- offloading moves what remains off the GPU. It is
        # handed the already-reduced footprint, so the layers it has to
        # place are fewer than the raw cache would have needed.
        layers_needed = max(1, int(slots_after_compression))
        run = self._offloader.run(
            OffloadWorkload(
                num_layers=layers_needed,
                compute_ms_per_layer=10.0,
                warm_fetch_ms_per_layer=6.0,
                cold_fetch_ms_per_layer=40.0,
                gpu_layer_capacity=gpu_layer_capacity,
            )
        )
        resident = min(run.gpu_layers_resident, layers_needed)
        offload_reduction = layers_needed / resident if resident else 1.0

        return LongContextResult(
            original_tokens=original_tokens,
            tokens_after_eviction=len(surviving),
            slots_after_compression=slots_after_compression,
            gpu_resident_slots=float(resident),
            stage_reductions=[
                eviction_reduction,
                compression_reduction,
                offload_reduction,
            ],
            kept_indices=list(decision.keep_indices),
        )

