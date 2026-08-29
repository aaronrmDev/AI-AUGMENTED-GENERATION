from datetime import UTC, datetime, timedelta

import pytest

from src.mag.application.gating.recency_weighted_sampling import RecencyWeightedSampling
from src.mag.domain.entities import GatingCandidate

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _candidate(score: float, timestamp: datetime | None) -> GatingCandidate:
    return GatingCandidate(
        content_text="x",
        score=score,
        salience=0.0,
        timestamp=timestamp,
        source_type="episode",
        origin=None,  # type: ignore[arg-type]
        embedding=[],
    )


async def test_older_candidate_decays_relative_to_a_newer_one():
    older = _candidate(score=0.8, timestamp=NOW - timedelta(hours=48))
    newer = _candidate(score=0.8, timestamp=NOW - timedelta(hours=1))

    result = await RecencyWeightedSampling().execute(
        [older, newer], half_life_hours=24.0, now=NOW
    )

    newer_result = next(c for c in result if c.timestamp == newer.timestamp)
    older_result = next(c for c in result if c.timestamp == older.timestamp)
    assert newer_result.score > older_result.score


async def test_candidate_with_no_timestamp_keeps_its_exact_original_score():
    candidate = _candidate(score=0.42, timestamp=None)

    result = await RecencyWeightedSampling().execute([candidate], now=NOW)

    assert result[0].score == 0.42


async def test_result_is_sorted_by_resulting_score_descending():
    low = _candidate(score=0.1, timestamp=NOW - timedelta(hours=1))
    high_but_old = _candidate(score=0.9, timestamp=NOW - timedelta(hours=1000))
    no_timestamp = _candidate(score=0.5, timestamp=None)

    result = await RecencyWeightedSampling().execute(
        [low, high_but_old, no_timestamp], half_life_hours=24.0, now=NOW
    )

    scores = [c.score for c in result]
    assert scores == sorted(scores, reverse=True)


async def test_returns_the_same_count_as_the_input():
    candidates = [
        _candidate(score=0.1, timestamp=NOW),
        _candidate(score=0.2, timestamp=None),
        _candidate(score=0.3, timestamp=NOW - timedelta(hours=5)),
    ]

    result = await RecencyWeightedSampling().execute(candidates, now=NOW)

    assert len(result) == len(candidates)


async def test_zero_half_life_raises_value_error():
    with pytest.raises(ValueError, match="half_life_hours must be positive"):
        await RecencyWeightedSampling().execute(
            [_candidate(score=0.5, timestamp=NOW)], half_life_hours=0.0, now=NOW
        )


async def test_negative_half_life_raises_value_error():
    with pytest.raises(ValueError, match="half_life_hours must be positive"):
        await RecencyWeightedSampling().execute(
            [_candidate(score=0.5, timestamp=NOW)], half_life_hours=-1.0, now=NOW
        )


async def test_empty_candidate_list_returns_empty_list():
    result = await RecencyWeightedSampling().execute([], now=NOW)

    assert result == []


async def test_runs_without_a_supplied_now():
    # timestamp=None takes the pass-through branch, so this never touches
    # the unpredictable real wall-clock value -- it only exercises the
    # "now defaults to datetime.now(UTC)" path without crashing.
    candidate = _candidate(score=0.7, timestamp=None)

    result = await RecencyWeightedSampling().execute([candidate])

    assert result[0].score == 0.7
