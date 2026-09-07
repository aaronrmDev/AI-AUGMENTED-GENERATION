from dataclasses import dataclass, field

from src.cag.domain.entities import ConversationTurn
from src.cag.domain.payload_footprint import dense_bits, payload_bits
from src.cag.domain.ports import KVCacheCompressor, KVCacheEvictor, MultiTurnCacheStrategy


@dataclass(frozen=True)
class RealTimeChatResult:
    # per_turn_resident is what the fiftieth-turn claim is actually
    # about: whether memory stops growing, not whether the total is
    # smaller. Kept per turn for the same reason SessionRun keeps
    # per_turn_recompute.
    per_turn_recompute: list[int] = field(default_factory=list)
    per_turn_resident: list[float] = field(default_factory=list)
    unbounded_resident: list[int] = field(default_factory=list)
    stage_reductions: list[float] = field(default_factory=list)


class RealTimeChatPipeline:
    # CAG.md's Archetype C: "multi-turn caching stores conversation KV
    # incrementally so each new turn doesn't recompute history; left
    # alone, that cache would still grow without bound across a long
    # conversation, so eviction bounds it by dropping low-importance
    # tokens as turns accumulate, and compression fits more of what
    # remains into the same GPU memory."
    #
    # Note this triple is the one CAG.md itself grades below the others:
    # Eviction + Multi-Turn Caching scores A rather than A+. The pipeline
    # is built the same way regardless, so whether that weaker pair shows
    # up in the measurement is a question the numbers answer rather than
    # something assumed either way here.
    def __init__(
        self,
        session: MultiTurnCacheStrategy,
        evictor: KVCacheEvictor,
        compressor: KVCacheCompressor,
        resident_token_budget: int,
        channels_per_token: int = 16,
        quantized_bit_width: int = 4,
    ) -> None:
        if resident_token_budget < 1:
            raise ValueError("resident_token_budget must be at least 1")
        if channels_per_token < 1:
            raise ValueError("channels_per_token must be at least 1")
        self._session = session
        self._evictor = evictor
        self._compressor = compressor
        self._budget = resident_token_budget
        self._channels = channels_per_token
        self._quantized_bit_width = quantized_bit_width

    def execute(self, turns: list[ConversationTurn]) -> RealTimeChatResult:
        run = self._session.run_session(turns)

        unbounded: list[int] = []
        resident: list[float] = []
        accumulated = 0
        compression_reduction = 1.0

        for turn in turns:
            accumulated += turn.new_tokens
            unbounded.append(accumulated)

            # Eviction bounds the growing session cache. Scores stand in
            # for accumulated attention -- older turns decay, which is
            # the ordering eviction acts on in a real conversation.
            scores = [1.0 / (accumulated - position) for position in range(accumulated)]
            decision = self._evictor.select_keep_indices(scores, min(self._budget, accumulated))
            kept = len(decision.keep_indices)

            # Compression then fits more of what survived into the same
            # memory -- acting only on what eviction left, which is the
            # ordering the archetype depends on.
            surviving_rows = [[float(i)] * self._channels for i in range(kept)]
            compressed = self._compressor.compress(surviving_rows)
            after_bits = payload_bits(compressed.payload, self._quantized_bit_width)
            if after_bits:
                compression_reduction = dense_bits(kept, self._channels) / after_bits
            resident.append(kept / compression_reduction)

        eviction_reduction = unbounded[-1] / min(self._budget, unbounded[-1])
        return RealTimeChatResult(
            per_turn_recompute=list(run.per_turn_recompute),
            per_turn_resident=resident,
            unbounded_resident=unbounded,
            stage_reductions=[eviction_reduction, compression_reduction],
        )

