from src.cag.domain.entities import ConversationTurn, SessionRun
from src.cag.domain.ports import MultiTurnCacheStrategy
from src.cag.infrastructure.naive_recompute_session import history_tokens_before


class RocketKVMTSession(MultiTurnCacheStrategy):
    # RocketKV-MT (CAG.md): "retains all KV tokens for future turns while
    # constraining which tokens get selected within the CURRENT turn --
    # it prioritizes keeping material available for turns not yet asked,
    # at the cost of being more selective about what the present turn
    # contributes."
    #
    # So retention is total (peak memory matches full history) while the
    # working set each turn attends over is capped. The trade this models
    # is therefore memory-for-quality, not memory-for-compute: nothing is
    # discarded, but the present turn sees less than it could.
    def __init__(self, current_turn_budget: int) -> None:
        if current_turn_budget < 1:
            raise ValueError("current_turn_budget must be at least 1")
        self._current_turn_budget = current_turn_budget

    def run_session(self, turns: list[ConversationTurn]) -> SessionRun:
        per_turn = [min(turn.new_tokens, self._current_turn_budget) for turn in turns]
        peak = 0
        for index, turn in enumerate(turns):
            peak = max(peak, history_tokens_before(turns, index) + turn.new_tokens)
        return SessionRun(
            strategy="rocketkv-mt",
            per_turn_recompute=per_turn,
            tokens_recomputed=sum(per_turn),
            # Retains everything: the whole point of the method.
            peak_tokens_held=peak,
        )
