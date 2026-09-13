from dataclasses import dataclass

from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    BudgetAllocation,
    ContextItem,
    Paradigm,
)
from src.shared.tokenization import count_tokens

_SECTION_TITLES = {
    Paradigm.CAG: "Pre-loaded reference knowledge (CAG)",
    Paradigm.MAG: "What is known about this user (MAG)",
    Paradigm.RAG: "Retrieved documents (RAG)",
}


@dataclass(frozen=True)
class AssembledContext:
    text: str
    included: list[ContextItem]
    dropped: dict[Paradigm, int]
    tokens_used: dict[Paradigm, int]


def assemble_context(items: list[ContextItem], allocation: BudgetAllocation) -> AssembledContext:
    """Pack each paradigm's items into its own slice, best score first.

    The walk skips an item that doesn't fit and keeps going, the same
    shape TokenBudgetAllocation and CompressingRetriever already use, so
    one oversized item never shuts out a smaller one. Dropped counts are
    how the source's "state grows too large -> eviction" rule becomes
    visible per turn. Section headers are not charged to any slice; the
    reserve absorbs those few tokens.
    """
    sections: list[str] = []
    included: list[ContextItem] = []
    dropped = dict.fromkeys(PARADIGM_ORDER, 0)
    tokens_used = dict.fromkeys(PARADIGM_ORDER, 0)
    for paradigm in PARADIGM_ORDER:
        budget = allocation.for_paradigm(paradigm)
        ranked = sorted(
            (item for item in items if item.paradigm is paradigm),
            key=lambda item: item.score,
            reverse=True,
        )
        kept: list[ContextItem] = []
        for item in ranked:
            cost = count_tokens(item.content)
            if tokens_used[paradigm] + cost > budget:
                dropped[paradigm] += 1
                continue
            tokens_used[paradigm] += cost
            kept.append(item)
        if kept:
            body = "\n\n".join(item.content for item in kept)
            sections.append(f"## {_SECTION_TITLES[paradigm]}\n{body}")
            included.extend(kept)
    return AssembledContext("\n\n".join(sections), included, dropped, tokens_used)
