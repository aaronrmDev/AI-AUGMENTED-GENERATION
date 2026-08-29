import math

import pytest

from src.mag.application.gating.dynamic_reranking import DynamicReranking
from src.mag.domain.entities import GatingCandidate


def _candidate(score: float, embedding: list[float]) -> GatingCandidate:
    return GatingCandidate(
        content_text="x",
        score=score,
        salience=0.0,
        timestamp=None,
        source_type="episode",
        origin=None,  # type: ignore[arg-type]
        embedding=embedding,
    )


async def test_candidate_matching_the_query_embedding_is_rescored_to_near_one():
    candidate = _candidate(score=0.2, embedding=[1.0, 0.0])

    result = await DynamicReranking().execute([candidate], query_embedding=[1.0, 0.0])

    assert result[0].score == pytest.approx(1.0)


async def test_orthogonal_embedding_is_rescored_to_near_zero():
    candidate = _candidate(score=0.9, embedding=[0.0, 1.0])

    result = await DynamicReranking().execute([candidate], query_embedding=[1.0, 0.0])

    assert result[0].score == pytest.approx(0.0)


async def test_opposite_direction_embedding_is_rescored_to_near_negative_one():
    candidate = _candidate(score=0.9, embedding=[-1.0, 0.0])

    result = await DynamicReranking().execute([candidate], query_embedding=[1.0, 0.0])

    assert result[0].score == pytest.approx(-1.0)


async def test_candidate_with_no_embedding_keeps_its_exact_original_score():
    candidate = _candidate(score=0.42, embedding=[])

    result = await DynamicReranking().execute([candidate], query_embedding=[1.0, 0.0])

    assert result[0].score == 0.42


async def test_zero_vector_candidate_embedding_resolves_to_zero_without_raising():
    candidate = _candidate(score=0.5, embedding=[0.0, 0.0])

    result = await DynamicReranking().execute([candidate], query_embedding=[1.0, 0.0])

    assert result[0].score == 0.0


async def test_zero_vector_query_embedding_resolves_to_zero_without_raising():
    candidate = _candidate(score=0.5, embedding=[1.0, 0.0])

    result = await DynamicReranking().execute([candidate], query_embedding=[0.0, 0.0])

    assert result[0].score == 0.0


async def test_result_is_sorted_by_resulting_score_descending():
    matches_query = _candidate(score=0.1, embedding=[1.0, 0.0])
    orthogonal = _candidate(score=0.9, embedding=[0.0, 1.0])
    no_embedding = _candidate(score=0.5, embedding=[])

    result = await DynamicReranking().execute(
        [orthogonal, no_embedding, matches_query], query_embedding=[1.0, 0.0]
    )

    scores = [c.score for c in result]
    assert scores == sorted(scores, reverse=True)


async def test_returns_the_same_count_as_the_input():
    candidates = [
        _candidate(score=0.1, embedding=[1.0, 0.0]),
        _candidate(score=0.2, embedding=[]),
        _candidate(score=0.3, embedding=[0.0, 1.0]),
    ]

    result = await DynamicReranking().execute(candidates, query_embedding=[1.0, 0.0])

    assert len(result) == len(candidates)


async def test_candidate_with_a_nan_embedding_component_sorts_last_without_corrupting_others():
    # A NaN component anywhere in an embedding (a corrupted upstream write,
    # for example) makes the dot-product sum NaN, and NaN fails every
    # comparison -- sorting on the raw score directly can scramble the
    # OTHER, well-defined scores around it, not just misplace this one.
    matches_query = _candidate(score=0.1, embedding=[1.0, 0.0])
    corrupted = _candidate(score=0.1, embedding=[float("nan"), 0.0])
    orthogonal = _candidate(score=0.9, embedding=[0.0, 1.0])

    result = await DynamicReranking().execute(
        [corrupted, orthogonal, matches_query], query_embedding=[1.0, 0.0]
    )

    assert result[0].score == pytest.approx(1.0)  # matches_query
    assert result[1].score == pytest.approx(0.0)  # orthogonal
    assert math.isnan(result[2].score)  # corrupted, sorted last


async def test_empty_candidate_list_returns_empty_list():
    result = await DynamicReranking().execute([], query_embedding=[1.0, 0.0])

    assert result == []
