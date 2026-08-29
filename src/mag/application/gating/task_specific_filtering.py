from src.mag.domain.entities import GatingCandidate


class TaskSpecificFiltering:
    # The strategy to reach for when the noise problem isn't too many
    # candidates but too many KINDS of candidate (issue #58) -- a pure
    # type-membership filter, not a re-score or a re-rank. Preserves
    # input order rather than imposing one, since ordering by relevance
    # is a different gating strategy's job (see TopKSelection).
    async def execute(
        self, candidates: list[GatingCandidate], allowed_source_types: set[str]
    ) -> list[GatingCandidate]:
        return [c for c in candidates if c.source_type in allowed_source_types]
