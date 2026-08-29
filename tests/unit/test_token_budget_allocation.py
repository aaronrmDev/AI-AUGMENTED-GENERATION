import uuid

from src.mag.application.gating.token_budget_allocation import TokenBudgetAllocation
from src.mag.domain.entities import GatingCandidate, SemanticMemory
from src.shared.tokenization import count_tokens


def _candidate(text: str, score: float) -> GatingCandidate:
    origin = SemanticMemory(
        id=uuid.uuid4(), user_id=uuid.uuid4(), fact_key="k", fact_value=text, embedding=[]
    )
    return GatingCandidate(
        content_text=text,
        score=score,
        salience=0.0,
        timestamp=None,
        source_type="fact",
        origin=origin,
        embedding=[],
    )


async def test_selects_candidates_in_score_descending_order_up_to_budget():
    low = _candidate("alpha", score=1.0)
    high = _candidate("bravo", score=3.0)
    mid = _candidate("charlie", score=2.0)
    budget = (
        count_tokens(low.content_text) + count_tokens(high.content_text)
        + count_tokens(mid.content_text)
    )

    result = await TokenBudgetAllocation().execute([low, high, mid], budget)

    assert result == [high, mid, low]


async def test_lower_scored_short_candidate_included_after_longer_higher_scored_one_skipped():
    long_text = "word " * 200  # many tokens -- deliberately too big for the budget below
    short_text = "hi"
    long_candidate = _candidate(long_text, score=10.0)  # scores highest, sorted first
    short_candidate = _candidate(short_text, score=1.0)  # scores lowest, sorted last
    short_tokens = count_tokens(short_text)
    long_tokens = count_tokens(long_text)
    assert long_tokens > short_tokens  # sanity check the fixture is shaped as intended
    budget = short_tokens  # too small for long_candidate, exactly enough for short_candidate

    result = await TokenBudgetAllocation().execute([long_candidate, short_candidate], budget)

    # The walk must not stop at the first oversized candidate: the long one
    # (checked first, since it scores highest) is skipped, but the smaller
    # one after it is still checked and fits.
    assert long_candidate not in result
    assert short_candidate in result
    assert result == [short_candidate]


async def test_running_total_of_selected_candidates_never_exceeds_budget():
    candidates = [
        _candidate("short one", score=5.0),
        _candidate("a somewhat longer piece of text here", score=4.0),
        _candidate("x", score=3.0),
        _candidate("another moderately sized chunk of content", score=2.0),
        _candidate("y" * 5, score=1.0),
    ]
    budget = count_tokens(candidates[0].content_text) + count_tokens(candidates[2].content_text)

    result = await TokenBudgetAllocation().execute(candidates, budget)

    total_tokens = sum(count_tokens(c.content_text) for c in result)
    assert total_tokens <= budget


async def test_zero_budget_returns_empty_list():
    candidates = [_candidate("hello", score=1.0)]

    result = await TokenBudgetAllocation().execute(candidates, 0)

    assert result == []


async def test_negative_budget_returns_empty_list():
    candidates = [_candidate("hello", score=1.0)]

    result = await TokenBudgetAllocation().execute(candidates, -5)

    assert result == []


async def test_empty_candidate_list_returns_empty_list():
    result = await TokenBudgetAllocation().execute([], 1000)

    assert result == []


async def test_single_candidate_exceeding_budget_is_excluded_not_raised():
    text = "word " * 200
    candidate = _candidate(text, score=1.0)
    budget = count_tokens(text) - 1  # one token too small to fit

    result = await TokenBudgetAllocation().execute([candidate], budget)

    assert result == []
