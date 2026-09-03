# tests/integration/test_auth_endpoints.py
import os

import pytest
from httpx import ASGITransport, AsyncClient

# src.api.dependencies builds its Postgres engine (and asyncpg connection pool)
# once, at first import, and every test in this module reuses that same cached
# `src.api.main` module (see `_client` below) rather than re-importing it fresh.
# pytest-asyncio's default is a brand-new event loop per test function, but an
# asyncpg connection pool is bound to whichever loop was running when its
# connections were opened -- reusing pooled connections from a prior test's
# (now-closed) loop raises "Event loop is closed" / "'NoneType' object has no
# attribute 'send'" deep in asyncpg's Windows ProactorEventLoop transport, since
# the pool's connections outlive the loop they were created on. Pinning every
# test in this module to one shared event loop matches the engine's actual
# lifetime (created once, reused for the module) and avoids that mismatch --
# mirroring how a real uvicorn process runs on a single, persistent event loop.
pytestmark = pytest.mark.asyncio(loop_scope="module")

# Note: Redis is flushed before every test in this file by the autouse
# _clean_redis_between_all_integration_tests fixture in
# tests/integration/conftest.py, which covers every integration test file for
# the same reason this module originally needed its own local version of it
# (see that fixture's docstring for why) -- no local fixture needed here.


@pytest.fixture(autouse=True)
def _default_cookie_secure_setting():
    # The refresh cookie ships Secure by default; COOKIE_SECURE=false is the
    # documented opt-out for local HTTP development. Tests run against the
    # default unless one deliberately overrides it, and this restores whatever
    # was there afterward so an override can't leak into a later test.
    previous = os.environ.pop("COOKIE_SECURE", None)
    yield
    if previous is None:
        os.environ.pop("COOKIE_SECURE", None)
    else:
        os.environ["COOKIE_SECURE"] = previous


async def _client(app_database_url, redis_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from src.api.main import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_register_then_login_returns_a_working_access_token(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        register_response = await client.post(
            "/auth/register", json={"email": "flow@example.com", "password": "hunter2hunter2"}
        )
        assert register_response.status_code == 201

        login_response = await client.post(
            "/auth/login", json={"email": "flow@example.com", "password": "hunter2hunter2"}
        )
        assert login_response.status_code == 200
        assert "access_token" in login_response.json()
        assert "refresh_token" in login_response.cookies


async def test_login_sets_the_refresh_cookie_httponly_secure_and_scoped_to_auth(
    app_database_url, redis_url
):
    """The refresh cookie's flags are the control, not the cookie's existence.

    HttpOnly keeps it out of reach of page JavaScript (and therefore of an XSS
    payload); Secure keeps the browser from ever putting it on a plaintext
    request, which is what makes HttpOnly worth anything against a network
    attacker; Path=/auth stops it being attached to every unrelated request to
    this origin once other routers exist. Asserting on the raw Set-Cookie
    header rather than a parsed jar is deliberate — the flags are exactly what
    a cookie jar normalizes away.
    """
    async with await _client(app_database_url, redis_url) as client:
        await client.post(
            "/auth/register", json={"email": "flags@example.com", "password": "hunter2hunter2"}
        )
        login_response = await client.post(
            "/auth/login", json={"email": "flags@example.com", "password": "hunter2hunter2"}
        )

        set_cookie = login_response.headers["set-cookie"]
        assert "HttpOnly" in set_cookie
        assert "Secure" in set_cookie
        assert "Path=/auth" in set_cookie


async def test_cookie_secure_can_be_turned_off_for_local_http_development(
    app_database_url, redis_url
):
    """COOKIE_SECURE=false is the opt-out; anything else (including unset) stays secure.

    The polarity is the point: a flag that must be set to *enable* security is
    one a production deploy can silently forget. This asserts the off switch
    works so nobody is tempted to invert the default to make local dev easy.
    """
    async with await _client(app_database_url, redis_url) as client:
        await client.post(
            "/auth/register", json={"email": "insecure@example.com", "password": "hunter2hunter2"}
        )
        os.environ["COOKIE_SECURE"] = "false"
        login_response = await client.post(
            "/auth/login", json={"email": "insecure@example.com", "password": "hunter2hunter2"}
        )

        set_cookie = login_response.headers["set-cookie"]
        assert "Secure" not in set_cookie
        # HttpOnly is not part of the opt-out — it costs nothing over HTTP and
        # protects against the same XSS read either way.
        assert "HttpOnly" in set_cookie


async def test_login_with_wrong_password_returns_generic_401(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        await client.post(
            "/auth/register", json={"email": "wp@example.com", "password": "hunter2hunter2"}
        )
        response = await client.post(
            "/auth/login", json={"email": "wp@example.com", "password": "wrongwrong"}
        )
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid credentials"}


async def test_login_with_unknown_email_returns_the_same_generic_401(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        response = await client.post(
            "/auth/login", json={"email": "ghost@example.com", "password": "whatever1"}
        )
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid credentials"}


async def test_register_rejects_an_invalid_payload(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        response = await client.post(
            "/auth/register", json={"email": "not-an-email", "password": "short"}
        )
        assert response.status_code == 422


async def test_refresh_rotates_the_token_and_the_old_one_cannot_be_replayed(
    app_database_url, redis_url
):
    async with await _client(app_database_url, redis_url) as client:
        await client.post(
            "/auth/register", json={"email": "rot@example.com", "password": "hunter2hunter2"}
        )
        login_response = await client.post(
            "/auth/login", json={"email": "rot@example.com", "password": "hunter2hunter2"}
        )
        old_cookie = login_response.cookies.get("refresh_token")

        refresh_response = await client.post(
            "/auth/refresh", cookies={"refresh_token": old_cookie}
        )
        assert refresh_response.status_code == 200

        replay_response = await client.post("/auth/refresh", cookies={"refresh_token": old_cookie})
        assert replay_response.status_code == 401


@pytest.mark.parametrize(
    "bad_cookie",
    [
        "not-a-uuid",
        "",
        "12345",
        "deadbeef-dead-beef-dead-beefdeadbee",  # one hex digit short of a UUID
        "'; DROP TABLE users; --",
    ],
)
async def test_refresh_with_a_malformed_cookie_returns_401_not_500(
    app_database_url, redis_url, bad_cookie
):
    """A cookie value is client-controlled, so it must not be able to crash the endpoint.

    uuid.UUID() raises ValueError on anything that isn't a UUID. Before this
    was handled, a stale cookie left over from an earlier deployment — or a
    crafted one — produced an unhandled 500 rather than the 401 the domain
    already defines for an unusable refresh token. A 500 is both a worse
    answer and a louder one: it distinguishes "malformed" from "unknown",
    which a uniform 401 deliberately does not.
    """
    async with await _client(app_database_url, redis_url) as client:
        response = await client.post("/auth/refresh", cookies={"refresh_token": bad_cookie})

        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid credentials"}


@pytest.mark.parametrize(
    "bad_cookie",
    [
        "not-a-uuid",
        "",
        "12345",
        "deadbeef-dead-beef-dead-beefdeadbee",
        "'; DROP TABLE users; --",
    ],
)
async def test_logout_with_a_malformed_cookie_still_succeeds_and_clears_the_cookie(
    app_database_url, redis_url, bad_cookie
):
    """Logout has to be unfailable, which is a stronger requirement than refresh's.

    A user whose cookie this endpoint cannot parse is precisely the user who
    most needs it cleared, and returning 401 or 500 here would strand that
    unparseable value in their browser with no API path to get rid of it. So
    logout is deliberately more forgiving than /auth/refresh: it skips the
    revocation it could never have performed (no Redis key can exist under an
    id that is not a UUID) and still returns 204 with the cookie deleted.
    """
    async with await _client(app_database_url, redis_url) as client:
        response = await client.post("/auth/logout", cookies={"refresh_token": bad_cookie})

        assert response.status_code == 204
        # An expiring Set-Cookie for the same name is how a deletion is
        # expressed over HTTP — the path must match the one login set, or the
        # browser treats it as a different cookie and the original survives.
        set_cookie = response.headers["set-cookie"]
        assert "refresh_token=" in set_cookie
        assert "Path=/auth" in set_cookie


async def test_logout_with_a_valid_cookie_revokes_the_token(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        await client.post(
            "/auth/register", json={"email": "bye@example.com", "password": "hunter2hunter2"}
        )
        login_response = await client.post(
            "/auth/login", json={"email": "bye@example.com", "password": "hunter2hunter2"}
        )
        cookie = login_response.cookies.get("refresh_token")

        logout_response = await client.post("/auth/logout", cookies={"refresh_token": cookie})
        assert logout_response.status_code == 204

        # The revocation is what makes logout mean anything: the same cookie
        # must no longer buy a new access token.
        refresh_response = await client.post("/auth/refresh", cookies={"refresh_token": cookie})
        assert refresh_response.status_code == 401


async def test_logout_with_no_cookie_at_all_is_a_no_op_204(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        response = await client.post("/auth/logout")
        assert response.status_code == 204


async def test_sixth_request_in_a_window_is_rate_limited(app_database_url, redis_url):
    async with await _client(app_database_url, redis_url) as client:
        for i in range(5):
            response = await client.post(
                "/auth/login", json={"email": f"rl{i}@example.com", "password": "wrongwrong"}
            )
            assert "X-RateLimit-Remaining" in response.headers
        sixth = await client.post(
            "/auth/login", json={"email": "rl-sixth@example.com", "password": "wrongwrong"}
        )
        assert sixth.status_code == 429
        assert sixth.headers["X-RateLimit-Remaining"] == "0"
