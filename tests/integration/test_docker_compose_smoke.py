# tests/integration/test_docker_compose_smoke.py
"""Manual end-to-end smoke test for the real Docker Compose stack.

This is NOT part of the `pytest tests/` suite CI runs: the CI sub-project
doesn't stand up a Docker Compose stack, and this script talks to a real
server on localhost rather than an app instance or TestContainers-managed
dependency. The file keeps its `test_` prefix so a human scanning
`tests/integration/` finds it, but it deliberately defines no `test_*`
function — pytest's default collection walks this module looking for
functions named `test_*` and finds none, so `pytest tests/` neither runs
nor reports anything from this file. Run it directly instead, after the
stack is up.

Setup (see docker/docker-compose.yml and Task 14 of the auth-foundation
plan):

    JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") \\
    APP_DB_PASSWORD=$(python3 -c "import secrets; print(secrets.token_hex(16))") \\
    docker compose -f docker/docker-compose.yml up --build -d

Then confirm all three services are healthy:

    docker compose -f docker/docker-compose.yml ps

The refresh-token cookie defaults to `Secure`, which this local stack can't
satisfy since it publishes the API on plain HTTP — docker-compose.yml sets
`COOKIE_SECURE: "false"` for exactly that reason, so no extra setup is
needed here, but never carry that override into a real deployment sitting
behind TLS.

The equivalent smoke test as raw curl, for anyone who wants to run it by
hand instead of via this script:

    curl -s -X POST http://localhost:8000/auth/register -H "Content-Type: application/json" \\
      -d '{"email":"smoke@example.com","password":"hunter2hunter2"}'
    curl -s -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" \\
      -d '{"email":"smoke@example.com","password":"hunter2hunter2"}' -c /tmp/cookies.txt
    curl -s -X POST http://localhost:8000/auth/refresh -b /tmp/cookies.txt
    curl -s -X POST http://localhost:8000/auth/logout -b /tmp/cookies.txt \\
      -o /dev/null -w "%{http_code}\\n"

Expected: register returns 201 with an id/email; login returns 200 with an
access_token and sets a refresh_token cookie; refresh returns 200 with a new
access_token; logout returns 204.

Tear down afterward to leave a clean state:

    docker compose -f docker/docker-compose.yml down -v

Run this script directly (from the repository root, with the stack up):

    uv run python tests/integration/test_docker_compose_smoke.py
"""

import sys
import uuid

import httpx

BASE_URL = "http://localhost:8000"


def _run() -> None:
    # A fresh email each run -- the stack's Postgres volume persists between
    # runs of this script (only `docker compose down -v` clears it), and
    # /auth/register on a duplicate email would fail the smoke test for a
    # reason that has nothing to do with the stack being broken.
    email = f"smoke-{uuid.uuid4().hex[:8]}@example.com"
    password = "hunter2hunter2"

    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        print("=== POST /auth/register ===")
        register_response = client.post(
            "/auth/register", json={"email": email, "password": password}
        )
        print(register_response.status_code, register_response.text)
        assert register_response.status_code == 201, "register did not return 201"

        print("=== POST /auth/login ===")
        login_response = client.post("/auth/login", json={"email": email, "password": password})
        print(login_response.status_code, login_response.text)
        assert login_response.status_code == 200, "login did not return 200"
        assert "access_token" in login_response.json(), "login response missing access_token"
        assert "refresh_token" in login_response.cookies, "login did not set a refresh_token cookie"

        print("=== POST /auth/refresh ===")
        refresh_response = client.post("/auth/refresh")
        print(refresh_response.status_code, refresh_response.text)
        assert refresh_response.status_code == 200, "refresh did not return 200"
        assert "access_token" in refresh_response.json(), "refresh response missing access_token"

        print("=== POST /auth/logout ===")
        logout_response = client.post("/auth/logout")
        print(logout_response.status_code)
        assert logout_response.status_code == 204, "logout did not return 204"

    print("\nSmoke test passed: register -> login -> refresh -> logout all behaved as expected.")


if __name__ == "__main__":
    try:
        _run()
    except httpx.ConnectError:
        print(
            f"Could not connect to {BASE_URL} -- is the stack up? "
            "See the module docstring for the docker compose command.",
            file=sys.stderr,
        )
        sys.exit(1)
