import re

from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm
from src.orchestration.domain.ports import QueryClassifier

# Cue phrases drawn from how Concept 1 describes each routing dimension
# (unified_rag_cag_mag_architecture.md): data freshness and live data point
# to RAG, prior-conversation dependency points to MAG, static and
# pre-loaded reference knowledge points to CAG.
DEFAULT_CUES: dict[Paradigm, tuple[str, ...]] = {
    Paradigm.CAG: (
        "policy", "policies", "manual", "guide", "handbook", "documentation",
        "procedure", "how do i", "how to", "explain", "reference", "code file", "faq",
    ),
    Paradigm.MAG: (
        "remember", "remind me", "we discussed", "we talked", "left off", "continue",
        "earlier", "last time", "previously", "i told you", "i asked", "i mentioned",
        "my preference", "you said", "our conversation",
    ),
    Paradigm.RAG: (
        "today", "latest", "current", "currently", "right now", "this week",
        "this month", "changed", "recent", "recently", "new", "news", "update",
        "updated", "live", "price", "prices", "stock", "compare",
    ),
}

# One cue scores 0.7, two 0.91, three 0.973: each further cue closes 70% of
# the remaining distance to certainty.
_CUE_WEIGHT = 0.7


class LexicalQueryClassifier(QueryClassifier):
    """The cheapest classifier and the floor the other two must beat."""

    def __init__(self, cues: dict[Paradigm, tuple[str, ...]] | None = None) -> None:
        source = cues if cues is not None else DEFAULT_CUES
        self._patterns = {
            paradigm: [
                re.compile(rf"\b{re.escape(cue)}\b", re.IGNORECASE)
                for cue in source.get(paradigm, ())
            ]
            for paradigm in PARADIGM_ORDER
        }

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        scores: dict[Paradigm, float] = {}
        for paradigm, patterns in self._patterns.items():
            matches = sum(1 for pattern in patterns if pattern.search(query))
            scores[paradigm] = 1.0 - (1.0 - _CUE_WEIGHT) ** matches
        return scores
