from src.cag.domain.entities import ConversationTurn, SessionRun
from src.cag.domain.ports import MultiTurnCacheStrategy
from src.cag.infrastructure.naive_recompute_session import history_tokens_before


class MemServeSession(MultiTurnCacheStrategy):
    # MemServe (CAG.md): "targets disaggregated, cloud-style serving
    # specifically, with an elastic memory pool and context caching built
    # for architectures where serving compute and cache storage don't
    # have to live on the same machine."
    #
    # Disaggregation is the whole mechanism, and it cuts both ways. The
    # pool is elastic, so nothing is ever evicted for lack of local room
    # and no session is capped by one machine's memory -- the reason to
    # want it. But the cache is not where the compute is, so every turn
    # fetches its history across the network, which is why this is the
    # only strategy here reporting non-zero transfer_tokens. Modelling
    # the elasticity without the transfer would describe unlimited free
    # memory rather than a serving architecture.
    def __init__(self, local_resident_tokens: int) -> None:
        if local_resident_tokens < 0:
            raise ValueError("local_resident_tokens must be non-negative")
        self._local_resident_tokens = local_resident_tokens

    def run_session(self, turns: list[ConversationTurn]) -> SessionRun:
        per_turn = [turn.new_tokens for turn in turns]
        transfer = 0
        peak_local = 0
        for index, turn in enumerate(turns):
            history = history_tokens_before(turns, index)
            # Whatever does not fit locally is fetched from the pool.
            transfer += max(0, history - self._local_resident_tokens)
            peak_local = max(
                peak_local, min(history, self._local_resident_tokens) + turn.new_tokens
            )
        return SessionRun(
            strategy="memserve",
            per_turn_recompute=per_turn,
            tokens_recomputed=sum(per_turn),
            peak_tokens_held=peak_local,
            transfer_tokens=transfer,
        )
