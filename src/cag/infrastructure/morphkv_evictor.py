from src.cag.domain.entities import EvictionDecision
from src.cag.domain.ports import RecentPatternEvictor

_METHOD = "morphkv"


class MorphKVEvictor(RecentPatternEvictor):
    # MorphKV's own shape (CAG.md): decisions come from recent attention
    # patterns rather than the full accumulated history, via "Sum/Max
    # Fusion" over those patterns -- disclosed here as fused_score[i] =
    # sum_i + max_i across the supplied recent-window rows, a direct
    # reading of the name with no further reweighting invented on top.
    # Because the scoring basis IS already "recent," there's no separate
    # reserved recent-window carve-out the way H2O/SnapKV need one -- the
    # top-`budget` fused scores are the whole selection.
    def select_keep_indices(
        self, recent_attention_windows: list[list[float]], budget: int
    ) -> EvictionDecision:
        if not recent_attention_windows:
            raise ValueError("recent_attention_windows must be non-empty")
        seq_len = len(recent_attention_windows[0])
        if seq_len == 0:
            raise ValueError("each attention window must be non-empty")
        for window in recent_attention_windows:
            if len(window) != seq_len:
                raise ValueError("every attention window must have the same length")
        if budget < 1:
            raise ValueError("budget must be at least 1")

        if budget >= seq_len:
            return EvictionDecision(
                method=_METHOD, keep_indices=list(range(seq_len)), evicted_count=0
            )

        sum_scores = [sum(window[i] for window in recent_attention_windows) for i in range(seq_len)]
        max_scores = [max(window[i] for window in recent_attention_windows) for i in range(seq_len)]
        fused = [sum_scores[i] + max_scores[i] for i in range(seq_len)]

        keep = sorted(
            sorted(range(seq_len), key=lambda i: fused[i], reverse=True)[:budget]
        )
        return EvictionDecision(
            method=_METHOD, keep_indices=keep, evicted_count=seq_len - len(keep)
        )
