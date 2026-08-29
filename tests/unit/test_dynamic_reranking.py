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


async def test_candidate_with_a_nan_embedding_component_resolves_to_zero_not_nan():
    # A NaN component anywhere in an embedding (a corrupted upstream write,
    # for example) makes the raw dot-product/norm computation NaN.
    # _cosine_similarity now sanitizes any non-finite result to 0.0 --
    # same "no honest signal" fallback as the existing zero-norm case --
    # so a NaN never reaches the candidate's own .score field, where a
    # future consumer (JSON serialization, a sum/average for a relevance
    # display) could otherwise be silently poisoned with no exception
    # raised.
    candidate = _candidate(score=0.5, embedding=[float("nan"), 0.0])

    result = await DynamicReranking().execute([candidate], query_embedding=[1.0, 0.0])

    assert result[0].score == 0.0
    assert not math.isnan(result[0].score)


async def test_a_candidate_that_arrives_already_nan_scored_sorts_last_without_corrupting_others():
    # DynamicReranking's own computation can no longer produce a NaN score
    # (see the test above), but a candidate could in principle still
    # arrive already NaN-scored from some future upstream source -- e.g.
    # one with no embedding, which this stage passes through untouched
    # rather than re-scoring. safe_score() in the final sort must still
    # guard against that without corrupting the other, well-defined
    # scores around it.
    matches_query = _candidate(score=0.1, embedding=[1.0, 0.0])
    already_corrupted = _candidate(score=float("nan"), embedding=[])
    orthogonal = _candidate(score=0.9, embedding=[0.0, 1.0])

    result = await DynamicReranking().execute(
        [already_corrupted, orthogonal, matches_query], query_embedding=[1.0, 0.0]
    )

    assert result[0].score == pytest.approx(1.0)  # matches_query
    assert result[1].score == pytest.approx(0.0)  # orthogonal
    assert math.isnan(result[2].score)  # already_corrupted, passed through, sorted last


async def test_empty_candidate_list_returns_empty_list():
    result = await DynamicReranking().execute([], query_embedding=[1.0, 0.0])

    assert result == []
