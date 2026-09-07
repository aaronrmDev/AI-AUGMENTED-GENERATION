from src.cag.domain.entities import ConversationTurn, SessionRun
from src.cag.domain.ports import MultiTurnCacheStrategy
from src.cag.infrastructure.naive_recompute_session import history_tokens_before
from src.cag.infrastructure.shadowkv_compressor import ShadowKVCompressor


class ShadowKVSession(MultiTurnCacheStrategy):
    # ShadowKV (CAG.md) "does double duty here by sharing low-rank
    # subspaces between the original sequence and its continuation, so
    # the same low-rank mechanism that shrinks its footprint in the
    # compression role also keeps multi-turn accuracy high, since a
    # continuation isn't rebuilding its representation from scratch."
    #
    # "The same method" is taken literally: this composes the real
    # ShadowKVCompressor the KV Cache Compression batch already built
    # rather than reimplementing its arithmetic, the same way that
    # compressor itself composes PALUCompressor for its low-rank step.
    # A second implementation of one mechanism would be free to drift
    # from the first, and any claim that the two roles share a mechanism
    # would then be a statement about the prose rather than the code.
    def __init__(self, compressor: ShadowKVCompressor, subspace_reuse: float) -> None:
        if not (0.0 <= subspace_reuse <= 1.0):
            raise ValueError("subspace_reuse must be between 0.0 and 1.0")
        self._compressor = compressor
        self._subspace_reuse = subspace_reuse

    def run_session(self, turns: list[ConversationTurn]) -> SessionRun:
        # A continuation reuses the subspace already built for the
        # sequence it continues, so only the unshared remainder of each
        # new turn needs its representation built from scratch.
        per_turn = [
            turn.new_tokens if index == 0
            else max(1, int(turn.new_tokens * (1.0 - self._subspace_reuse)))
            for index, turn in enumerate(turns)
        ]
        peak = 0
        for index, turn in enumerate(turns):
            resident = history_tokens_before(turns, index) + turn.new_tokens
            peak = max(peak, resident)
        return SessionRun(
            strategy="shadowkv",
            per_turn_recompute=per_turn,
            tokens_recomputed=sum(per_turn),
            peak_tokens_held=peak,
        )
