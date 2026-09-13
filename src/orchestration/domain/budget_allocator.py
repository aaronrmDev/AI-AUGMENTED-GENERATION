import math
from collections.abc import Iterable

from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    BudgetAllocation,
    BudgetShares,
    Paradigm,
)

DEFAULT_SHARES = BudgetShares()


def _recipients(idle: Paradigm, active: frozenset[Paradigm]) -> tuple[Paradigm, ...]:
    # Each branch is one of Concept 3's reallocation rules, read literally.
    if idle is Paradigm.RAG:  # "if query is simple (no RAG needed) -> MAG slice expands"
        return next(((p,) for p in (Paradigm.MAG, Paradigm.CAG) if p in active), ())
    if idle is Paradigm.CAG:  # "if CAG cache misses -> RAG slice temporarily expands"
        return next(((p,) for p in (Paradigm.RAG, Paradigm.MAG) if p in active), ())
    # "if session is new (no MAG state) -> CAG or RAG slice expands": the
    # source doesn't choose, so both share it in proportion to base share.
    return tuple(p for p in (Paradigm.CAG, Paradigm.RAG) if p in active)


def allocate(
    total_tokens: int, contributing: Iterable[Paradigm], shares: BudgetShares = DEFAULT_SHARES
) -> BudgetAllocation:
    """Slice the context window for one turn.

    Every idle paradigm donates its whole BASE slice exactly once, so no
    donation is ever re-donated and the result doesn't depend on the order
    rules are applied in. Anything with no eligible recipient, plus every
    flooring remainder, lands in reserve, so the slices always sum to
    total_tokens. The source's fourth rule ("MAG state grows too large ->
    compression or eviction") never expands a slice; assemble_context
    enforces it by dropping lowest-scoring items.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    active = frozenset(contributing)
    base = {p: math.floor(total_tokens * shares.for_paradigm(p)) for p in PARADIGM_ORDER}
    query = math.floor(total_tokens * shares.query)
    reserve = total_tokens - sum(base.values()) - query
    slices = dict(base)

    for idle in PARADIGM_ORDER:
        if idle in active:
            continue
        donation = base[idle]
        slices[idle] -= donation
        recipients = _recipients(idle, active)
        if not recipients:
            reserve += donation
            continue
        if len(recipients) == 1:
            slices[recipients[0]] += donation
            continue
        weight_total = sum(shares.for_paradigm(p) for p in recipients)
        if weight_total <= 0.0:
            reserve += donation
            continue
        given = 0
        for recipient in recipients:
            portion = math.floor(donation * shares.for_paradigm(recipient) / weight_total)
            slices[recipient] += portion
            given += portion
        reserve += donation - given

    return BudgetAllocation(
        cag=slices[Paradigm.CAG],
        mag=slices[Paradigm.MAG],
        rag=slices[Paradigm.RAG],
        query=query,
        reserve=reserve,
    )
