import os
from collections.abc import Awaitable, Callable
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.identity.domain.ports import RateLimiter

WINDOW_SECONDS = 60
AUTH_LIMIT = 5  # per client IP per window, on /auth/register and /auth/login
SESSION_CREATE_LIMIT = 20  # per user per window, on POST /sessions
_DEFAULT_CHAT_LIMIT = 100


def chat_rate_limit() -> int:
    """Requests per user per window, shared by POST /chat and answering in a session.

    Read per call, like COOKIE_SECURE, so an operator or a test can change it without
    controlling module import order.
    """
    limit = int(os.environ.get("CHAT_RATE_LIMIT_PER_MINUTE", str(_DEFAULT_CHAT_LIMIT)))
    if limit < 1:
        raise ValueError("CHAT_RATE_LIMIT_PER_MINUTE must be at least 1")
    return limit


class RateLimitExceeded(Exception):
    def __init__(self, limit: int, remaining: int, reset_at: datetime) -> None:
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """Reapplies the X-RateLimit-* headers onto whatever response the app ends up returning.

    `enforce_rate_limit` below sets these same headers directly on its injected
    `response` for the common case, but that only reaches the client when the route
    returns normally. When the rate limiter allows the request and the route then
    raises a domain exception anyway (e.g. login's `AuthenticateUser.execute()`
    raising `InvalidCredentials` for a wrong password), FastAPI's exception-handling
    path builds an entirely new Response from the registered handler — one that never
    saw the injected `response` and so never inherits headers written to it. Starlette
    dispatches registered exception handlers inside `ExceptionMiddleware`, which sits
    *below* any middleware added via `add_middleware`, so `call_next` here always
    hands back the final response — success or handled-exception alike — letting this
    middleware attach the headers stashed on `request.state` regardless of which path
    produced that response.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        headers = getattr(request.state, "rate_limit_headers", None)
        if headers is not None:
            response.headers.update(headers)
        return response


async def enforce_rate_limit(
    request: Request,
    response: Response,
    *,
    limiter: RateLimiter,
    key: str,
    limit: int,
    window_seconds: int = WINDOW_SECONDS,
) -> None:
    allowed, remaining, reset_at = await limiter.check(
        key=key, limit=limit, window_seconds=window_seconds
    )
    if not allowed:
        # Raising here means the route's own successful-response path never
        # runs, so headers set on the injected `response` below would never
        # reach the client — FastAPI's exception handler builds an entirely
        # new JSONResponse for the 429 and does not inherit them. Carry the
        # values on the exception itself instead, and let the handler set
        # them on the response it actually returns.
        raise RateLimitExceeded(limit=limit, remaining=0, reset_at=reset_at)

    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": reset_at.isoformat(),
    }
    # Written to both places: directly on `response` covers the normal
    # successful-response path with no extra hop through the middleware, and
    # stashed on `request.state` is what lets RateLimitHeadersMiddleware recover
    # these same values if the route raises a domain exception afterward.
    response.headers.update(headers)
    request.state.rate_limit_headers = headers
