import math
from dataclasses import replace

from src.mag.application.gating._scoring import safe_score
from src.mag.domain.entities import GatingCandidate


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    # Same defensive shape as tests/unit/mag_fakes.py's _cosine_similarity --
    # kept real here (dot product over the product of L2 norms), not a
    # stand-in constant, so re-ranked order reflects an actual similarity a
    # real backend would produce. norm-is-zero (an all-zero embedding) has no
    # direction to compare, so it returns 0.0 rather than dividing by zero.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    result = dot / (norm_a * norm_b)
    # A NaN or Inf component anywhere in either embedding (a corrupted
    # upstream write, not a case this project's own writers should ever
    # produce, but nothing downstream validates embeddings on the way in)
    # makes `result` non-finite. Every sort in this package already guards
    # AGAINST a non-finite score via safe_score(), but that only protects
    # comparisons -- it doesn't stop the raw NaN from being written into
    # the returned candidate's own .score field, where a future consumer
    # (JSON serialization, a sum/average for a relevance display) could be
    # silently poisoned with no exception raised. Same "no honest signal"
    # fallback as the zero-norm case above closes that gap at its source.
    return result if math.isfinite(result) else 0.0


class DynamicReranking:
    # Re-ranks the retrieved set against the CURRENT query rather than
    # trusting the original retrieval score (issue #60) -- the recomputed
    # cosine similarity REPLACES candidate.score outright, it does not blend
    # or average with whatever score the candidate arrived with. Only
    # candidates with a real embedding can be re-ranked this way: embedding
    # is [] for Postgres-only and graph-node candidates (this project's
    # established convention -- see GatingCandidate's docstring), and there's
    # no honest similarity to compute against nothing, so those pass through
    # with their original score untouched rather than being scored 0.0 (which
    # would falsely claim "definitely irrelevant").
    async def execute(
        self, candidates: list[GatingCandidate], query_embedding: list[float]
    ) -> list[GatingCandidate]:
        reranked = [
            replace(candidate, score=_cosine_similarity(query_embedding, candidate.embedding))
            if candidate.embedding
            else candidate
            for candidate in candidates
        ]
        return sorted(reranked, key=lambda c: safe_score(c.score), reverse=True)
