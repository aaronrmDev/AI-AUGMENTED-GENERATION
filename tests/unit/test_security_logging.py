import uuid

import pytest
import structlog

from src.api.caller import Caller
from src.api.exception_handlers import rate_limit_exceeded_handler, session_not_found_handler
from src.api.rate_limit import RateLimitExceeded
from src.api.security_logging import configure_security_logging
from src.orchestration.domain.errors import SessionNotFound


@pytest.fixture(autouse=True)
def _restore_structlog_defaults():
    # configure_security_logging() below reconfigures structlog process-wide
    # (PrintLoggerFactory, JSON rendering). Left in place, that configuration
    # would leak into every test that runs afterward in this same process,
    # regardless of module -- reset it back to structlog's own defaults once
    # each test here is done.
    yield
    structlog.reset_defaults()


def _request(*, caller: Caller | None = None):
    from starlette.requests import Request

    request = Request(
        {"type": "http", "method": "POST", "path": "/sessions/x/answers", "headers": []}
    )
    if caller is not None:
        request.state.caller = caller
    return request


async def test_a_session_not_found_response_logs_the_callers_identity_and_denied_id():
    configure_security_logging()
    caller = Caller(tenant_id=uuid.uuid4(), user_id=uuid.uuid4())
    session_id = uuid.uuid4()

    with structlog.testing.capture_logs() as captured:
        response = await session_not_found_handler(
            _request(caller=caller), SessionNotFound(session_id)
        )

    assert response.status_code == 404
    [event] = captured
    assert event["event"] == "authz_denied"
    assert event["tenant_id"] == str(caller.tenant_id)
    assert event["user_id"] == str(caller.user_id)
    assert event["session_id"] == str(session_id)
    assert event["path"] == "/sessions/x/answers"
    assert "question" not in event and "answer" not in event and "authorization" not in event


async def test_a_rate_limit_exceeded_response_logs_which_key_tripped():
    configure_security_logging()
    from datetime import UTC, datetime

    exc = RateLimitExceeded(
        limit=5, remaining=0, reset_at=datetime.now(UTC), key="register:203.0.113.7"
    )

    with structlog.testing.capture_logs() as captured:
        response = await rate_limit_exceeded_handler(_request(), exc)

    assert response.status_code == 429
    [event] = captured
    assert event["event"] == "rate_limit_exceeded"
    assert event["key"] == "register:203.0.113.7"
    assert event["limit"] == 5
