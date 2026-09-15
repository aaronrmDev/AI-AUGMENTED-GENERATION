from src.cag.domain.entities import ConversationTurn, SessionRun
from src.cag.domain.ports import MultiTurnCacheStrategy
from src.cag.infrastructure.naive_recompute_session import history_tokens_before


class SGLangSession(MultiTurnCacheStrategy):
    # SGLang's approach (CAG.md): "folds KV reuse into how it executes
    # structured LM programs -- reuse happens automatically as part of
    # running a program's control flow, which suits it to complex,
    # multi-step agent workflows rather than a single flat conversation."
    #
    # That last clause is the method's whole shape, and it is why this
    # class tracks which prefixes it has already materialised rather than
    # only what the current chain contains. When a program forks -- two
    # turns sharing one parent -- the shared ancestry is materialised
    # once and both branches run off it. On a flat conversation there is
    # nothing to fork, so this degrades exactly to incremental append,
    # which is the honest result rather than a shortfall: CAG.md says
    # plainly that a single flat conversation is not what it suits.
    def run_session(self, turns: list[ConversationTurn]) -> SessionRun:
        materialised: set[str] = set()
        per_turn: list[int] = []
        peak = 0
        for index, turn in enumerate(turns):
            # A turn's own tokens always cost; its ancestry costs only
            # the part no earlier branch has already built.
            cost = turn.new_tokens
            if turn.parent_turn_id is not None and turn.parent_turn_id not in materialised:
                cost += history_tokens_before(turns, index)
            per_turn.append(cost)
            materialised.add(turn.turn_id)
            if turn.parent_turn_id is not None:
                materialised.add(turn.parent_turn_id)
            peak = max(peak, history_tokens_before(turns, index) + turn.new_tokens)
        return SessionRun(
            strategy="sglang",
            per_turn_recompute=per_turn,
            tokens_recomputed=sum(per_turn),
            peak_tokens_held=peak,
        )
