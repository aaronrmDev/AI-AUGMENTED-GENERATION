from evaluation.domain.entities import JudgeScores
from evaluation.domain.ports import Judge


class FakeJudge(Judge):
    def __init__(self, scores: JudgeScores | None = None) -> None:
        self._scores = scores or JudgeScores(
            coherence=4, relevance=4, completeness=4, groundedness=4
        )
        self.calls: list[tuple[str, str, str, str, str]] = []

    async def score(
        self, query: str, response_a: str, response_b: str, context_a: str, context_b: str
    ) -> tuple[JudgeScores, JudgeScores]:
        self.calls.append((query, response_a, response_b, context_a, context_b))
        return self._scores, self._scores
