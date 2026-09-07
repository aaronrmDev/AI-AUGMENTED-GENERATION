from src.cag.domain.entities import ConversationTurn, SessionRun
from src.cag.domain.ports import MultiTurnCacheStrategy


def history_tokens_before(turns: list[ConversationTurn], index: int) -> int:
    # Tokens belonging to this turn's ancestry. For a flat conversation
    # that is simply everything before it; for a branching program it is
    # only the chain this turn actually descends from, which is what
    # makes a branch cheaper than a continuation of the whole session.
    by_id = {turn.turn_id: turn for turn in turns}
    current = turns[index].parent_turn_id
    total = 0
    seen: set[str] = set()
    while current is not None and current in by_id and current not in seen:
        seen.add(current)
        total += by_id[current].new_tokens
        current = by_id[current].parent_turn_id
    return total


class NaiveRecomputeSession(MultiTurnCacheStrategy):
    # The behaviour CAG.md compares everything against: "like adding a
    # new chapter to a book and reprinting the whole thing every time."
    # Nothing is kept between turns, so every turn pays for its entire
    # history plus itself -- the quadratic total this technique family
    # exists to escape.
    def run_session(self, turns: list[ConversationTurn]) -> SessionRun:
        per_turn = [
            history_tokens_before(turns, index) + turn.new_tokens
            for index, turn in enumerate(turns)
        ]
        return SessionRun(
            strategy="naive-recompute",
            per_turn_recompute=per_turn,
            tokens_recomputed=sum(per_turn),
            peak_tokens_held=max(per_turn) if per_turn else 0,
        )


class IncrementalAppendSession(MultiTurnCacheStrategy):
    # The general mechanism CAG.md describes before naming any of the
    # five specialisations: keep the session's KV resident and compute
    # only the new turn's tokens. This is the O(1)-per-turn shape the
    # five methods each then specialise, and it is included as its own
    # strategy so their departures from it are measurable rather than
    # assumed.
    #
    # Crucially the cache here is keyed BY SESSION, which is what a
    # conversation-scoped cache actually is. A turn continuing the
    # session directly in front of it reuses everything; a turn that
    # forks from anywhere else begins a new session, whose cache starts
    # empty and must materialise the ancestry it inherited. An earlier
    # version of this class charged only new_tokens for every turn
    # regardless, which quietly modelled an oracle that shares trunks
    # across branches for free -- and since sharing across branches is
    # precisely what SGLang claims as its own contribution, that version
    # handed SGLang's benefit to the baseline and made the two
    # indistinguishable on the one workload built to tell them apart.
    def run_session(self, turns: list[ConversationTurn]) -> SessionRun:
        per_turn: list[int] = []
        peak = 0
        previous_turn_id: str | None = None
        for index, turn in enumerate(turns):
            continues_current_session = turn.parent_turn_id == previous_turn_id
            if continues_current_session:
                per_turn.append(turn.new_tokens)
            else:
                per_turn.append(history_tokens_before(turns, index) + turn.new_tokens)
            peak = max(peak, history_tokens_before(turns, index) + turn.new_tokens)
            previous_turn_id = turn.turn_id
        return SessionRun(
            strategy="incremental-append",
            per_turn_recompute=per_turn,
            tokens_recomputed=sum(per_turn),
            peak_tokens_held=peak,
        )
