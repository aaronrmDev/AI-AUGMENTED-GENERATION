from src.mag.application.gating._scoring import safe_score
from src.mag.domain.entities import GatingCandidate
from src.shared.tokenization import count_tokens


class TokenBudgetAllocation:
    # Maximizes information density rather than memory count (#54) -- a
    # better fit than TopKSelection when candidates vary wildly in length.
    # Sorted by score like TopKSelection, but walks the WHOLE sorted list
    # instead of stopping at k: a later, smaller candidate can still fit
    # under budget even after an earlier, larger one didn't, so one
    # oversized miss must not halt the walk. Same skip-and-continue shape
    # CompressingRetriever already established for the analogous sentence
    # budget-fill problem (src/rag/infrastructure/compressing_retriever.py).
    async def execute(
        self, candidates: list[GatingCandidate], token_budget: int
    ) -> list[GatingCandidate]:
        # Only a genuinely invalid (negative) budget short-circuits here.
        # token_budget == 0 must still walk the list below: a zero-cost
        # candidate (empty content_text) legitimately fits in zero tokens,
        # and the per-candidate check two lines down already handles that
        # correctly -- returning [] early was excluding it for no reason.
        if token_budget < 0:
            return []
        ranked = sorted(candidates, key=lambda c: safe_score(c.score), reverse=True)
        selected: list[GatingCandidate] = []
        running_total = 0
        for candidate in ranked:
            candidate_tokens = count_tokens(candidate.content_text)
            if running_total + candidate_tokens > token_budget:
                continue
            running_total += candidate_tokens
            selected.append(candidate)
        return selected
