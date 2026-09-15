from src.cag.domain.entities import EvictionDecision
from src.cag.domain.ports import KVCacheEvictor

_METHOD = "snapkv"


class SnapKVEvictor(KVCacheEvictor):
    # SnapKV's own departure from H2O (CAG.md): scores come from an
    # observation window "plus 1D pooling" rather than a running
    # accumulated score, so a token sitting next to an important one
    # inherits some of its importance instead of needing its own high
    # score -- "keeps the important prefix tokens, the context
    # immediately around them, and a recent window." The 1D max-pool
    # over attention_scores is the mechanism that spreads importance to
    # neighbors; the recent-window-plus-top-up selection beneath it is
    # otherwise the same shape H2O uses.
    def __init__(self, pool_kernel_size: int, recent_window: int) -> None:
        if pool_kernel_size < 1:
            raise ValueError("pool_kernel_size must be at least 1")
        if recent_window < 0:
            raise ValueError("recent_window must be non-negative")
        self._pool_kernel_size = pool_kernel_size
        self._recent_window = recent_window

    def _pooled(self, scores: list[float]) -> list[float]:
        half = self._pool_kernel_size // 2
        seq_len = len(scores)
        return [
            max(scores[max(0, i - half) : min(seq_len, i + half + 1)]) for i in range(seq_len)
        ]

    def select_keep_indices(
        self, attention_scores: list[float], budget: int
    ) -> EvictionDecision:
        if not attention_scores:
            raise ValueError("attention_scores must be non-empty")
        if budget < 1:
            raise ValueError("budget must be at least 1")
        seq_len = len(attention_scores)
        pooled = self._pooled(attention_scores)

        if budget >= seq_len:
            return EvictionDecision(
                method=_METHOD, keep_indices=list(range(seq_len)), evicted_count=0
            )

        recent_start = max(0, seq_len - self._recent_window)
        recent_indices = set(range(recent_start, seq_len))

        if budget <= len(recent_indices):
            keep = sorted(range(seq_len - budget, seq_len))
            return EvictionDecision(
                method=_METHOD, keep_indices=keep, evicted_count=seq_len - len(keep)
            )

        top_up_budget = budget - len(recent_indices)
        candidates = [i for i in range(seq_len) if i not in recent_indices]
        top_up = sorted(candidates, key=lambda i: pooled[i], reverse=True)[:top_up_budget]
        keep = sorted(recent_indices | set(top_up))
        return EvictionDecision(
            method=_METHOD, keep_indices=keep, evicted_count=seq_len - len(keep)
        )
