import random

from src.cag.domain.entities import EvictionDecision
from src.cag.domain.ports import KVCacheEvictor

_METHOD = "nacl"


class NACLEvictor(KVCacheEvictor):
    # NACL's own departure from H2O/SnapKV (CAG.md): single-shot rather
    # than continuous -- one pass, no recent-window carve-out -- scoring
    # by an end-of-input proxy (the caller supplies attention_scores
    # already computed that way) "plus some random eviction." Modeled
    # here as: most of the budget goes to the top-scoring tokens
    # deterministically, but a `random_fraction` slice of the budget is
    # instead filled by a random draw from the remaining pool, so the
    # exact cutoff boundary isn't rigidly the same set every time a
    # similar-scored batch is evicted -- with random_fraction=0.0 this
    # degrades to plain top-k, disclosed and tested as the check that
    # the randomness is additive, not a replacement for scoring.
    def __init__(self, random_fraction: float, random_seed: int | None = None) -> None:
        if not (0.0 <= random_fraction <= 1.0):
            raise ValueError("random_fraction must be between 0.0 and 1.0")
        self._random_fraction = random_fraction
        self._rng = random.Random(random_seed)

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

        random_budget = round(budget * self._random_fraction)
        deterministic_budget = budget - random_budget

        ranked = sorted(range(seq_len), key=lambda i: attention_scores[i], reverse=True)
        deterministic_keep = ranked[:deterministic_budget]
        remaining_pool = ranked[deterministic_budget:]
        random_keep = self._rng.sample(
            remaining_pool, k=min(random_budget, len(remaining_pool))
        )

        keep = sorted(set(deterministic_keep) | set(random_keep))
        return EvictionDecision(
            method=_METHOD, keep_indices=keep, evicted_count=seq_len - len(keep)
        )
