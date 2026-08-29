from src.mag.application.gating._scoring import safe_score
from src.mag.domain.entities import GatingCandidate


class TopKSelection:
    # The simplest and fastest gating option -- the right default when speed
    # matters more than nuance and the scoring function is already
    # trustworthy (issue #53). No re-ranking, no diversity, no salience
    # weighting: just the top k candidates by score, in score order.
    async def execute(
        self, candidates: list[GatingCandidate], k: int
    ) -> list[GatingCandidate]:
        if k <= 0:
            return []
        ranked = sorted(candidates, key=lambda c: safe_score(c.score), reverse=True)
        return ranked[:k]
