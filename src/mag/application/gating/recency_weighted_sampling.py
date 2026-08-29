import math
from dataclasses import replace
from datetime import UTC, datetime

from src.mag.application.gating._scoring import safe_score
from src.mag.domain.entities import GatingCandidate


class RecencyWeightedSampling:
    # Re-weights, rather than selects: every candidate comes back, just with
    # its score decayed by age so recency doesn't wash out old-but-important
    # material outright -- a caller composes this with a selection strategy
    # (e.g. TopKSelection) to actually narrow the pool. Same half-life decay
    # formula as RecencyDecayFusionRetrieval's `_recency_decay`
    # (src/mag/application/queries/retrieve_with_recency_decay_fusion.py),
    # applied here to gating candidates instead of fused retrieval scores.
    async def execute(
        self,
        candidates: list[GatingCandidate],
        half_life_hours: float = 24.0,
        now: datetime | None = None,
    ) -> list[GatingCandidate]:
        if half_life_hours <= 0:
            # A zero half-life divides by zero in the decay formula; a
            # negative one would make the score GROW with age instead of
            # decaying, inverting the whole point of "recency-weighted."
            # Rejected here, at the entry point, rather than left to fail
            # cryptically deep in the math. This raises rather than
            # returning [] like the selection strategies' degenerate cases
            # (TopKSelection's k<=0, TokenBudgetAllocation's negative
            # budget) -- deliberately: those are SELECTION strategies,
            # where "nothing selected" is itself a valid, meaningful
            # result. This is a RE-SCORING strategy that returns every
            # candidate it's given; there's no analogous "valid empty
            # result" to fall back to, so silently swallowing a
            # nonsensical half-life would just hide a caller bug instead
            # of surfacing it.
            raise ValueError("half_life_hours must be positive")
        now = now or datetime.now(UTC)

        reweighted = []
        for candidate in candidates:
            if candidate.timestamp is None:
                # Facts and most graph nodes carry no timestamp -- there's no
                # age to compute, and the point of this strategy is to stop
                # over-penalizing age a timestamp-less candidate doesn't even
                # have, so it passes through with its score untouched.
                reweighted.append(candidate)
                continue
            age_hours = max((now - candidate.timestamp).total_seconds() / 3600.0, 0.0)
            decay = math.exp(-math.log(2) * age_hours / half_life_hours)
            # score * decay only "fades toward less relevant" when score is
            # >= 0 -- decay is in (0, 1], so it shrinks a positive score
            # toward zero as age grows, correctly. A NEGATIVE score (e.g.
            # DynamicReranking's cosine similarity, which ranges [-1, 1])
            # shrinks the same way, but shrinking a negative number toward
            # zero makes it LESS negative, i.e. climb toward "more
            # relevant" the older and less certain it gets -- exactly
            # backwards. Scaling by (2 - decay) instead (which is 1 at
            # age=0 and grows toward 2 as decay falls toward 0) pushes a
            # negative score further from zero -- more negative, i.e.
            # worse -- as it ages, mirroring how a positive score is
            # pushed toward zero. Both directions converge on the same
            # place at age=0 (decay=1, multiplier=1, unchanged either way).
            multiplier = decay if candidate.score >= 0 else (2 - decay)
            reweighted.append(replace(candidate, score=candidate.score * multiplier))

        return sorted(reweighted, key=lambda c: safe_score(c.score), reverse=True)
