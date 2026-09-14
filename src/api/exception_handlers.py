from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.rate_limit import RateLimitExceeded
from src.identity.domain.errors import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    TokenAlreadyUsed,
    TokenExpired,
)
from src.rag.domain.errors import UnsupportedFileType


async def invalid_credentials_handler(request: Request, exc: InvalidCredentials) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})


async def email_already_registered_handler(
    request: Request, exc: EmailAlreadyRegistered
) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": "Email already registered"})


async def token_expired_handler(request: Request, exc: TokenExpired) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "Token expired"})


async def token_already_used_handler(request: Request, exc: TokenAlreadyUsed) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})


async def unsupported_file_type_handler(request: Request, exc: UnsupportedFileType) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # Reads limit/remaining/reset_at off the exception rather than the response:
    # the raise in enforce_rate_limit happens before any headers are written to
    # the route's injected Response, and FastAPI's exception-handling path builds
    # an entirely new response object for a raised exception, which doesn't
    # inherit anything set on that never-returned Response.
    response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    response.headers["X-RateLimit-Limit"] = str(exc.limit)
    response.headers["X-RateLimit-Remaining"] = str(exc.remaining)
    response.headers["X-RateLimit-Reset"] = exc.reset_at.isoformat()
    return response


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(InvalidCredentials, invalid_credentials_handler)  # type: ignore[arg-type]
    app.add_exception_handler(EmailAlreadyRegistered, email_already_registered_handler)  # type: ignore[arg-type]
    app.add_exception_handler(TokenExpired, token_expired_handler)  # type: ignore[arg-type]
    app.add_exception_handler(TokenAlreadyUsed, token_already_used_handler)  # type: ignore[arg-type]
    app.add_exception_handler(UnsupportedFileType, unsupported_file_type_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]
