import math
from dataclasses import replace
from datetime import UTC, datetime

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
            # cryptically deep in the math.
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
            reweighted.append(replace(candidate, score=candidate.score * decay))

        return sorted(reweighted, key=lambda c: c.score, reverse=True)
