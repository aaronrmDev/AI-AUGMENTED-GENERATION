import os
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.dependencies import (
    get_rate_limiter,
    get_raw_db_session,
    get_refresh_token_store,
    get_token_issuer,
)
from src.api.schemas.auth import LoginRequest, RegisterRequest, RegisterResponse, TokenResponse
from src.identity.application.authenticate_user import AuthenticateUser
from src.identity.application.refresh_access_token import RefreshAccessToken
from src.identity.application.register_user import RegisterUser
from src.identity.application.revoke_refresh_token import RevokeRefreshToken
from src.identity.domain.errors import TokenAlreadyUsed
from src.identity.infrastructure.argon2_password_hasher import Argon2PasswordHasher
from src.identity.infrastructure.postgres_user_repository import PostgresUserRepository

router = APIRouter(prefix="/auth", tags=["auth"])

_RATE_LIMIT = 5
_RATE_LIMIT_WINDOW_SECONDS = 60
_REFRESH_COOKIE_NAME = "refresh_token"
_REFRESH_COOKIE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
# Scoped to /auth rather than "/": the refresh cookie is only ever read by
# /auth/refresh and /auth/logout, so there is no reason to attach it to every
# request the browser makes to this origin once other routers exist.
_REFRESH_COOKIE_PATH = "/auth"


def _cookie_secure() -> bool:
    """Whether to set the Secure flag on the refresh cookie. Defaults to on.

    Secure keeps the browser from ever sending the refresh token over plain
    HTTP, which is the whole point of putting it in an httpOnly cookie in the
    first place. The default is therefore True, and local HTTP development has
    to opt *out* explicitly by setting COOKIE_SECURE=false — the polarity
    matters: an opt-in flag left unset in production silently ships the
    insecure behavior, while an opt-out flag left unset ships the safe one.

    Read per call rather than at import time so a test (or a container's
    entrypoint) can set it without having to control module import order.
    """
    return os.environ.get("COOKIE_SECURE", "true").lower() != "false"


def _set_refresh_cookie(response: Response, token_id: uuid.UUID) -> None:
    response.set_cookie(
        _REFRESH_COOKIE_NAME,
        str(token_id),
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        path=_REFRESH_COOKIE_PATH,
        max_age=_REFRESH_COOKIE_MAX_AGE_SECONDS,
    )


def _parse_refresh_cookie(token_id_str: str) -> uuid.UUID:
    """Parse the refresh cookie's value, treating an unparseable one as a bad token.

    uuid.UUID() raises ValueError on anything that isn't a UUID, and a cookie
    value is entirely client-controlled — stale from a previous deployment,
    truncated, or simply crafted. Letting that ValueError escape produces an
    unhandled 500 where the domain already has a precise answer for "this
    refresh token is not usable", so it is translated to TokenAlreadyUsed and
    surfaces as the same generic 401 an expired or replayed token gets. Note
    that this deliberately does not distinguish "malformed" from "unknown":
    telling the two apart would leak whether a given token id was ever real.
    """
    try:
        return uuid.UUID(token_id_str)
    except ValueError as exc:
        raise TokenAlreadyUsed() from exc


class _RateLimitExceeded(Exception):
    def __init__(self, limit: int, remaining: int, reset_at: datetime) -> None:
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """Reapplies the X-RateLimit-* headers onto whatever response the app ends up returning.

    `_enforce_rate_limit` below sets these same headers directly on its injected
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


async def _enforce_rate_limit(request: Request, response: Response, route_name: str) -> None:
    limiter = get_rate_limiter()
    client_ip = request.client.host if request.client else "unknown"
    allowed, remaining, reset_at = await limiter.check(
        key=f"{route_name}:{client_ip}",
        limit=_RATE_LIMIT,
        window_seconds=_RATE_LIMIT_WINDOW_SECONDS,
    )
    if not allowed:
        # Raising here means the route's own successful-response path never
        # runs, so headers set on the injected `response` below would never
        # reach the client — FastAPI's exception handler builds an entirely
        # new JSONResponse for the 429 and does not inherit them. Carry the
        # values on the exception itself instead, and let the handler set
        # them on the response it actually returns.
        raise _RateLimitExceeded(limit=_RATE_LIMIT, remaining=0, reset_at=reset_at)

    headers = {
        "X-RateLimit-Limit": str(_RATE_LIMIT),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": reset_at.isoformat(),
    }
    # Written to both places: directly on `response` covers the normal
    # successful-response path with no extra hop through the middleware, and
    # stashed on `request.state` is what lets RateLimitHeadersMiddleware recover
    # these same values if the route raises a domain exception afterward.
    response.headers.update(headers)
    request.state.rate_limit_headers = headers


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_raw_db_session),
) -> RegisterResponse:
    await _enforce_rate_limit(request, response, "register")
    use_case = RegisterUser(
        user_repository=PostgresUserRepository(session), password_hasher=Argon2PasswordHasher()
    )
    user = await use_case.execute(email=payload.email, plain_password=payload.password)
    await session.commit()
    return RegisterResponse(id=user.id, email=user.email)


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_raw_db_session),
) -> TokenResponse:
    await _enforce_rate_limit(request, response, "login")
    use_case = AuthenticateUser(
        user_repository=PostgresUserRepository(session),
        password_hasher=Argon2PasswordHasher(),
        token_issuer=get_token_issuer(),
    )
    user, pair = await use_case.execute(email=payload.email, plain_password=payload.password)
    await get_refresh_token_store().save(pair.refresh_token, user_id=user.id)
    _set_refresh_cookie(response, pair.refresh_token.token_id)
    return TokenResponse(access_token=pair.access_token.value)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: Request, response: Response, session: AsyncSession = Depends(get_raw_db_session)
) -> TokenResponse:
    token_id_str = request.cookies.get(_REFRESH_COOKIE_NAME)
    if not token_id_str:
        raise TokenAlreadyUsed()
    token_id = _parse_refresh_cookie(token_id_str)

    store = get_refresh_token_store()
    user_id = await store.get_user_id(token_id)
    if user_id is None:
        raise TokenAlreadyUsed()

    # tenant_id isn't stored alongside the refresh token in this sub-project's schema
    # (deliberately, per the spec — refresh tokens carry no tenant claim); the new
    # access token is issued against the user's own tenant, looked up fresh via the
    # same injected session register/login already use — no ad-hoc engine here.
    user = await PostgresUserRepository(session).find_by_id(user_id)
    if user is None:
        raise TokenAlreadyUsed()

    use_case = RefreshAccessToken(refresh_token_store=store, token_issuer=get_token_issuer())
    new_pair = await use_case.execute(refresh_token_id=token_id, tenant_id=user.tenant_id)
    _set_refresh_cookie(response, new_pair.refresh_token.token_id)
    return TokenResponse(access_token=new_pair.access_token.value)


@router.post("/logout", status_code=204)
async def logout(request: Request, response: Response) -> None:
    token_id_str = request.cookies.get(_REFRESH_COOKIE_NAME)
    if token_id_str:
        # Deliberately more forgiving than /auth/refresh: logout is idempotent
        # and must not be failable. A user holding a cookie this endpoint can't
        # parse is exactly the user most in need of clearing it, and a 401 or a
        # 500 here would leave that unparseable value sitting in their browser
        # with no way to get rid of it through the API. So a bad cookie skips
        # the revocation it could never have accomplished anyway — there is no
        # Redis key under an id that isn't a UUID — and falls through to
        # delete_cookie below, which is the part the caller actually needs.
        try:
            token_id = uuid.UUID(token_id_str)
        except ValueError:
            token_id = None
        if token_id is not None:
            await RevokeRefreshToken(refresh_token_store=get_refresh_token_store()).execute(
                refresh_token_id=token_id
            )
    # Path must match the one set_cookie used, or the browser treats it as a
    # different cookie and the original survives the deletion.
    response.delete_cookie(_REFRESH_COOKIE_NAME, path=_REFRESH_COOKIE_PATH)
