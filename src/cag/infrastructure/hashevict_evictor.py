import random

from src.cag.domain.entities import EvictionDecision
from src.cag.domain.ports import HashBasedEvictor

_METHOD = "hashevict"


class HASHEVICTEvictor(HashBasedEvictor):
    # HASHEVICT's own departure from every attention-score-based method
    # above (CAG.md): it estimates which tokens are similar to each
    # other via locality-sensitive hashing (SimHash) BEFORE any attention
    # computation runs, which is what makes it lighter-weight than H2O
    # or SnapKV. Real SimHash: a fixed set of random hyperplanes (seeded
    # for determinism) turns each raw KV vector into a bit-string bucket
    # id; vectors landing in the same bucket are estimated-similar, and
    # only one representative per bucket (the most recent) survives as
    # non-redundant. If deduplication alone doesn't reach the budget, the
    # most recent survivors are kept up to it.
    def __init__(self, num_hash_bits: int, random_seed: int) -> None:
        if num_hash_bits < 1:
            raise ValueError("num_hash_bits must be at least 1")
        self._num_hash_bits = num_hash_bits
        self._random_seed = random_seed

    def _hyperplanes(self, dim: int) -> list[list[float]]:
        rng = random.Random(self._random_seed)
        return [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(self._num_hash_bits)]

    def _bucket(self, vector: list[float], hyperplanes: list[list[float]]) -> tuple[int, ...]:
        return tuple(
            1 if sum(v * h for v, h in zip(vector, hyperplane, strict=True)) >= 0 else 0
            for hyperplane in hyperplanes
        )

    def select_keep_indices(
        self, kv_vectors: list[list[float]], budget: int
    ) -> EvictionDecision:
        if not kv_vectors:
            raise ValueError("kv_vectors must be non-empty")
        if budget < 1:
            raise ValueError("budget must be at least 1")
        seq_len = len(kv_vectors)

        if budget >= seq_len:
            return EvictionDecision(
                method=_METHOD, keep_indices=list(range(seq_len)), evicted_count=0
            )

        hyperplanes = self._hyperplanes(len(kv_vectors[0]))
        last_seen_by_bucket: dict[tuple[int, ...], int] = {}
        for index, vector in enumerate(kv_vectors):
            bucket = self._bucket(vector, hyperplanes)
            # Most recent index in a bucket wins -- the most recent
            # occurrence of a redundant pattern is the one most likely to
            # still be relevant to what comes next.
            last_seen_by_bucket[bucket] = index

        deduped = sorted(last_seen_by_bucket.values())
        if len(deduped) > budget:
            deduped = deduped[-budget:]

        return EvictionDecision(
            method=_METHOD, keep_indices=deduped, evicted_count=seq_len - len(deduped)
        )
