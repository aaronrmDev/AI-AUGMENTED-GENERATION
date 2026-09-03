from src.mag.application.gating.top_k_selection import TopKSelection
from src.mag.domain.entities import GatingCandidate


def _candidate(score: float, content_text: str = "x") -> GatingCandidate:
    return GatingCandidate(
        content_text=content_text,
        score=score,
        salience=0.0,
        timestamp=None,
        source_type="episode",
        origin=None,  # type: ignore[arg-type]
        embedding=[],
    )


async def test_returns_candidates_in_descending_score_order():
    candidates = [_candidate(0.2), _candidate(0.9), _candidate(0.5)]

    result = await TopKSelection().execute(candidates, k=3)

    assert [c.score for c in result] == [0.9, 0.5, 0.2]


async def test_returns_exactly_k_when_more_candidates_than_k():
    candidates = [_candidate(0.1), _candidate(0.9), _candidate(0.5), _candidate(0.7)]

    result = await TopKSelection().execute(candidates, k=2)

    assert [c.score for c in result] == [0.9, 0.7]


async def test_returns_all_candidates_when_k_exceeds_candidate_count():
    candidates = [_candidate(0.1), _candidate(0.9)]

    result = await TopKSelection().execute(candidates, k=10)

    assert len(result) == 2
    assert [c.score for c in result] == [0.9, 0.1]


async def test_empty_candidates_returns_empty_list():
    result = await TopKSelection().execute([], k=5)

    assert result == []


async def test_k_zero_returns_empty_list():
    candidates = [_candidate(0.1), _candidate(0.9)]

    result = await TopKSelection().execute(candidates, k=0)

    assert result == []


async def test_tie_in_score_preserves_original_relative_order():
    # Distinguished by content_text, not just score -- three candidates
    # with genuinely IDENTICAL fields would make dataclass equality accept
    # ANY permutation as "correct" (since first == second == third would
    # already hold), proving nothing about actual ordering. This is the
    # real regression check: is sorted() actually stable here.
    first = _candidate(0.5, content_text="first")
    second = _candidate(0.5, content_text="second")
    third = _candidate(0.5, content_text="third")

    result = await TopKSelection().execute([first, second, third], k=3)

    assert [c.content_text for c in result] == ["first", "second", "third"]


async def test_nan_score_sorts_last_without_corrupting_well_defined_scores():
    # NaN fails every comparison, so sorting on the raw score can scramble
    # OTHER, well-defined scores around a single corrupted candidate, not
    # just misplace the NaN one itself. This strategy can't assume every
    # candidate it's handed came through a source that sanitizes its own
    # output, so it must defend itself here regardless.
    high = _candidate(0.9, content_text="high")
    corrupted = _candidate(float("nan"), content_text="corrupted")
    low = _candidate(0.1, content_text="low")

    result = await TopKSelection().execute([corrupted, high, low], k=3)

    assert [c.content_text for c in result] == ["high", "low", "corrupted"]
