import pytest

from src.cag.application.long_context_pipeline import LongContextPipeline
from src.cag.application.real_time_chat_pipeline import RealTimeChatPipeline
from src.cag.domain.combination_metrics import compounded_reduction, synergy_score
from src.cag.domain.entities import ConversationTurn
from src.cag.infrastructure.h2o_evictor import H2OEvictor
from src.cag.infrastructure.kivi_compressor import KIVICompressor
from src.cag.infrastructure.layerkv_offloader import LayerKVOffloader
from src.cag.infrastructure.naive_recompute_session import IncrementalAppendSession


def _kv(tokens: int, channels: int = 16) -> list[list[float]]:
    return [[float(t * channels + c) for c in range(channels)] for t in range(tokens)]


def _scores(tokens: int) -> list[float]:
    # Heavy hitters early, decaying after -- the shape real accumulated
    # attention takes, so eviction has something meaningful to select.
    return [1.0 / (1 + index) for index in range(tokens)]


def _pipeline() -> LongContextPipeline:
    return LongContextPipeline(
        evictor=H2OEvictor(recent_window=4),
        compressor=KIVICompressor(group_size=4, bits=4),
        offloader=LayerKVOffloader(),
    )


def test_each_stage_hands_the_next_one_strictly_less_to_handle():
    # CAG.md's structural claim for Archetype A, asserted as an ordering
    # rather than inferred from the total.
    result = _pipeline().execute(
        kv=_kv(64), attention_scores=_scores(64), token_budget=16, gpu_layer_capacity=4
    )
    assert result.tokens_after_eviction < result.original_tokens
    assert result.slots_after_compression < result.tokens_after_eviction
    assert result.gpu_resident_slots <= result.slots_after_compression


def test_compression_only_ever_sees_what_eviction_left():
    # The relationship the archetype depends on: the compressor is handed
    # the survivors, never the full cache.
    result = _pipeline().execute(
        kv=_kv(64), attention_scores=_scores(64), token_budget=16, gpu_layer_capacity=4
    )
    assert result.tokens_after_eviction == len(result.kept_indices)
    assert result.tokens_after_eviction == 16


def test_the_composed_reduction_beats_the_strongest_single_stage():
    # The floor a combination must clear before "synergy" is earned.
    result = _pipeline().execute(
        kv=_kv(64), attention_scores=_scores(64), token_budget=16, gpu_layer_capacity=4
    )
    measured = result.original_tokens / result.gpu_resident_slots
    assert measured > max(result.stage_reductions)
    assert synergy_score(measured, result.stage_reductions) > 0.0


def test_the_composed_reduction_approaches_the_full_product_of_its_stages():
    result = _pipeline().execute(
        kv=_kv(64), attention_scores=_scores(64), token_budget=16, gpu_layer_capacity=4
    )
    measured = result.original_tokens / result.gpu_resident_slots
    # Compounding, not merely stacking: within rounding of the product.
    assert measured == pytest.approx(compounded_reduction(result.stage_reductions), rel=0.2)


def test_the_long_context_pipeline_rejects_an_empty_cache():
    with pytest.raises(ValueError):
        _pipeline().execute(
            kv=[], attention_scores=[], token_budget=4, gpu_layer_capacity=2
        )


def _chat_pipeline(budget: int = 64) -> RealTimeChatPipeline:
    return RealTimeChatPipeline(
        session=IncrementalAppendSession(),
        evictor=H2OEvictor(recent_window=8),
        compressor=KIVICompressor(group_size=4, bits=4),
        resident_token_budget=budget,
    )


def _conversation(turns: int, tokens_per_turn: int = 40) -> list[ConversationTurn]:
    return [
        ConversationTurn(
            turn_id=f"t{i}",
            new_tokens=tokens_per_turn,
            parent_turn_id=None if i == 0 else f"t{i - 1}",
        )
        for i in range(turns)
    ]


def test_the_fiftieth_turn_costs_what_the_first_did():
    # CAG.md's Archetype C claim, stated literally: "what let the
    # fiftieth turn of a conversation run as fast as the first."
    result = _chat_pipeline().execute(_conversation(50))
    assert result.per_turn_recompute[-1] == result.per_turn_recompute[0]


def test_memory_stops_growing_even_though_the_conversation_does_not():
    result = _chat_pipeline(budget=64).execute(_conversation(50))
    # Unbounded, the session would hold every token ever spoken.
    assert result.unbounded_resident[-1] > 10 * result.unbounded_resident[0]
    # Bounded by eviction and shrunk by compression, residency plateaus.
    assert max(result.per_turn_resident) <= 64
    assert result.per_turn_resident[-1] < result.unbounded_resident[-1]


def test_the_chat_pipeline_rejects_a_non_positive_budget():
    with pytest.raises(ValueError):
        RealTimeChatPipeline(
            session=IncrementalAppendSession(),
            evictor=H2OEvictor(recent_window=8),
            compressor=KIVICompressor(group_size=4, bits=4),
            resident_token_budget=0,
        )
