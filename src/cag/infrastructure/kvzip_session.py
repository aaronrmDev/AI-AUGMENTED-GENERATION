from src.cag.domain.entities import ConversationTurn, SessionRun
from src.cag.domain.ports import MultiTurnCacheStrategy
from src.cag.infrastructure.naive_recompute_session import history_tokens_before


class KVzipSession(MultiTurnCacheStrategy):
    # KVzip (CAG.md): "a query-agnostic approach to compressed KV reuse,
    # amortizing its overhead across however many different queries
    # eventually get asked against that context, which suits it to
    # diverse future queries rather than a predictable single follow-up."
    #
    # Two things follow, and both are modelled. The compression is paid
    # ONCE, up front, on the context -- so its per-query cost falls the
    # longer the session runs, which is precisely what "amortizing"
    # means and is the method's whole argument for itself. And because
    # it is query-agnostic, that one compression serves every later
    # query regardless of what they ask, rather than being tuned to one.
    def __init__(self, compression_ratio: float, compression_overhead_tokens: int) -> None:
        if not (0.0 < compression_ratio <= 1.0):
            raise ValueError("compression_ratio must be in (0.0, 1.0]")
        if compression_overhead_tokens < 0:
            raise ValueError("compression_overhead_tokens must be non-negative")
        self._compression_ratio = compression_ratio
        self._overhead = compression_overhead_tokens

    def run_session(self, turns: list[ConversationTurn]) -> SessionRun:
        per_turn = [turn.new_tokens for turn in turns]
        if per_turn:
            # The one-time overhead lands on the first turn and is never
            # paid again -- a single up-front cost is what makes later
            # turns cheap, and spreading it evenly would hide exactly the
            # amortisation this method is claiming.
            per_turn[0] += self._overhead
        peak = 0
        for index, turn in enumerate(turns):
            resident = history_tokens_before(turns, index) + turn.new_tokens
            peak = max(peak, int(resident * self._compression_ratio))
        return SessionRun(
            strategy="kvzip",
            per_turn_recompute=per_turn,
            tokens_recomputed=sum(per_turn),
            peak_tokens_held=peak,
        )
