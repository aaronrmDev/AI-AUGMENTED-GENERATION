from src.cag.domain.entities import EvictionDecision
from src.cag.domain.ports import KVCacheEvictor

_METHOD = "h2o"


class H2OEvictor(KVCacheEvictor):
    # Heavy-Hitter Oracle: keeps a fixed recent window unconditionally,
    # then fills whatever budget remains with the highest-accumulated-
    # attention "heavy hitter" tokens among everything older than that
    # window (CAG.md: "keeping only heavy hitters... plus a recent
    # window").
    def __init__(self, recent_window: int) -> None:
        if recent_window < 0:
            raise ValueError("recent_window must be non-negative")
        self._recent_window = recent_window

    def select_keep_indices(
        self, attention_scores: list[float], budget: int
    ) -> EvictionDecision:
        if not attention_scores:
            raise ValueError("attention_scores must be non-empty")
        if budget < 1:
            raise ValueError("budget must be at least 1")
        seq_len = len(attention_scores)

        if budget >= seq_len:
            return EvictionDecision(
                method=_METHOD, keep_indices=list(range(seq_len)), evicted_count=0
            )

        recent_start = max(0, seq_len - self._recent_window)
        recent_indices = set(range(recent_start, seq_len))

        if budget <= len(recent_indices):
            # Budget too tight even for the recent window alone -- keep
            # only the most recent `budget` positions.
            keep = sorted(range(seq_len - budget, seq_len))
            return EvictionDecision(
                method=_METHOD, keep_indices=keep, evicted_count=seq_len - len(keep)
            )

        heavy_hitter_budget = budget - len(recent_indices)
        candidates = [i for i in range(seq_len) if i not in recent_indices]
        heavy_hitters = sorted(candidates, key=lambda i: attention_scores[i], reverse=True)[
            :heavy_hitter_budget
        ]
        keep = sorted(recent_indices | set(heavy_hitters))
        return EvictionDecision(
            method=_METHOD, keep_indices=keep, evicted_count=seq_len - len(keep)
        )
