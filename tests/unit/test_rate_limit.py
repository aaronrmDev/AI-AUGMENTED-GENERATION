from datetime import UTC, datetime

import pytest
from fastapi import Request, Response

from src.api.rate_limit import (
    RateLimitExceeded,
    chat_rate_limit,
    enforce_rate_limit,
    global_chat_limit,
    stream_concurrency_limit,
    stream_deadline_seconds,
    stream_slot_ttl_seconds,
)
from src.identity.domain.ports import RateLimiter
from src.orchestration.application.unified_answer_question import DEFAULT_STREAM_DEADLINE

_RESET = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)


class _FakeLimiter(RateLimiter):
    def __init__(self, allowed: bool, remaining: int) -> None:
        self._result = (allowed, remaining, _RESET)
        self.calls: list[tuple[str, int, int]] = []

    async def check(self, key: str, limit: int, window_seconds: int):
        self.calls.append((key, limit, window_seconds))
        return self._result


def _request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/", "headers": []})


async def test_an_allowed_request_carries_its_headers_on_the_response_and_request_state():
    request, response, limiter = _request(), Response(), _FakeLimiter(True, 7)

    await enforce_rate_limit(request, response, limiter=limiter, key="chat:u", limit=10)

    expected = {
        "X-RateLimit-Limit": "10",
        "X-RateLimit-Remaining": "7",
        "X-RateLimit-Reset": _RESET.isoformat(),
    }
    assert {k: response.headers[k] for k in expected} == expected
    assert request.state.rate_limit_headers == expected
    assert limiter.calls == [("chat:u", 10, 60)]


async def test_a_refused_request_raises_with_the_limit_and_the_reset_time():
    with pytest.raises(RateLimitExceeded) as refused:
        await enforce_rate_limit(
            _request(), Response(), limiter=_FakeLimiter(False, 0), key="k", limit=3
        )

    assert (refused.value.limit, refused.value.remaining, refused.value.reset_at) == (3, 0, _RESET)


def test_the_chat_limit_defaults_to_100_and_is_read_from_the_environment_per_call(monkeypatch):
    monkeypatch.delenv("CHAT_RATE_LIMIT_PER_MINUTE", raising=False)
    assert chat_rate_limit() == 100
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "2")
    assert chat_rate_limit() == 2


def test_a_chat_limit_below_1_is_refused(monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "0")
    with pytest.raises(ValueError):
        chat_rate_limit()


def test_the_global_chat_limit_defaults_to_10000_and_is_read_from_the_environment_per_call(
    monkeypatch,
):
    monkeypatch.delenv("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", raising=False)
    assert global_chat_limit() == 10000
    monkeypatch.setenv("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", "2")
    assert global_chat_limit() == 2


def test_a_global_chat_limit_below_1_is_refused(monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", "0")
    with pytest.raises(ValueError):
        global_chat_limit()


def test_the_stream_concurrency_limit_defaults_to_5_and_is_read_from_the_environment_per_call(
    monkeypatch,
):
    monkeypatch.delenv("STREAM_CONCURRENCY_LIMIT_PER_USER", raising=False)
    assert stream_concurrency_limit() == 5
    monkeypatch.setenv("STREAM_CONCURRENCY_LIMIT_PER_USER", "2")
    assert stream_concurrency_limit() == 2


def test_a_stream_concurrency_limit_below_1_is_refused(monkeypatch):
    monkeypatch.setenv("STREAM_CONCURRENCY_LIMIT_PER_USER", "0")
    with pytest.raises(ValueError):
        stream_concurrency_limit()


def test_the_stream_deadline_defaults_to_the_use_cases_own_constant_and_is_read_per_call(
    monkeypatch,
):
    monkeypatch.delenv("STREAM_DEADLINE_SECONDS", raising=False)
    assert stream_deadline_seconds() == DEFAULT_STREAM_DEADLINE
    monkeypatch.setenv("STREAM_DEADLINE_SECONDS", "30")
    assert stream_deadline_seconds() == 30.0


def test_a_non_positive_stream_deadline_is_refused(monkeypatch):
    monkeypatch.setenv("STREAM_DEADLINE_SECONDS", "0")
    with pytest.raises(ValueError):
        stream_deadline_seconds()


def test_the_stream_slot_ttl_tracks_the_deadline_plus_its_own_buffer(monkeypatch):
    monkeypatch.setenv("STREAM_DEADLINE_SECONDS", "30")
    assert stream_slot_ttl_seconds() == 40.0
