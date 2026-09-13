import uuid

import pytest

from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    BudgetAllocation,
    BudgetShares,
    CascadeResult,
    ContextItem,
    Paradigm,
    TierOutcome,
    TierResult,
)


def test_paradigm_order_is_cheapest_first():
    assert PARADIGM_ORDER == (Paradigm.CAG, Paradigm.MAG, Paradigm.RAG)


@pytest.mark.parametrize("outcome", [TierOutcome.TIMEOUT, TierOutcome.ERROR])
def test_a_tier_cannot_report_an_outcome_only_the_cascade_can_observe(outcome):
    with pytest.raises(ValueError):
        TierResult(outcome)


def test_a_miss_carries_no_items():
    with pytest.raises(ValueError):
        TierResult(TierOutcome.MISS, [ContextItem(Paradigm.CAG, "x", 0.1)])


def test_a_hit_carries_its_items():
    item = ContextItem(Paradigm.RAG, "x", 0.9, uuid.uuid4())
    assert TierResult(TierOutcome.HIT, [item]).items == [item]


def test_contributing_is_the_set_of_paradigms_that_returned_items():
    result = CascadeResult(
        items=[ContextItem(Paradigm.CAG, "a", 0.5), ContextItem(Paradigm.RAG, "b", 0.9)],
        attempts=[],
        satisfied=frozenset({Paradigm.RAG}),
        degraded=False,
    )
    assert result.contributing == frozenset({Paradigm.CAG, Paradigm.RAG})


def test_default_shares_are_the_source_split():
    shares = BudgetShares()
    assert (shares.cag, shares.mag, shares.rag, shares.query, shares.reserve) == (
        0.40, 0.25, 0.20, 0.10, 0.05,
    )
    assert shares.for_paradigm(Paradigm.MAG) == 0.25


def test_shares_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(ValueError):
        BudgetShares(cag=0.5, mag=0.5, rag=0.5, query=0.0, reserve=0.0)


def test_negative_shares_are_rejected():
    with pytest.raises(ValueError):
        BudgetShares(cag=-0.1, mag=0.35, rag=0.5, query=0.2, reserve=0.05)


def test_allocation_total_and_per_paradigm_lookup():
    allocation = BudgetAllocation(cag=4, mag=3, rag=2, query=1, reserve=1)
    assert allocation.total == 11
    assert allocation.for_paradigm(Paradigm.RAG) == 2
