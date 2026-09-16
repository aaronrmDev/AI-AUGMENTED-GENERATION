"""Full-stack proof: a real Celery worker (Celery's own start_worker test
helper), consuming from the real TestContainers Redis instance, actually runs
IngestDataSource against real Postgres and Qdrant -- reached entirely through
HTTP, the same way every other route in this API is proven."""
import os
import time
import uuid

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="module")
def celery_worker_for_ingestion(redis_url, app_database_url, qdrant_url, neo4j_url):
    os.environ["REDIS_URL"] = redis_url
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["QDRANT_URL"] = qdrant_url
    url, username, password = neo4j_url
    os.environ["NEO4J_URL"] = url
    os.environ["NEO4J_USERNAME"] = username
    os.environ["NEO4J_PASSWORD"] = password
    # get_token_issuer() (src/api/dependencies.py) reads this lazily, per call,
    # the same way REDIS_URL/APP_DATABASE_URL/QDRANT_URL above are read lazily
    # by their own dependency factories -- every other integration test file
    # that reaches /auth/register or /auth/login (test_sessions_endpoints.py's
    # _client() among them) sets this itself for exactly the same reason.
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-ingestion-endpoint-tests"

    # Deferred import, not module-top-level: src.workers.celery_app reads
    # REDIS_URL at import time (see the identical note on Task 3's dispatcher
    # fixture), and this module is also imported transitively by src.api.main
    # (via src.api.dependencies, Task 6's own change) -- a top-level import
    # here would run at collection time, before any env var above is set.
    from celery.contrib.testing.worker import start_worker

    from src.workers.celery_app import celery_app as _celery_app

    with start_worker(_celery_app, pool="solo", perform_ping_check=False):
        yield


async def _register_and_login(client: AsyncClient) -> str:
    email = f"ingest-{uuid.uuid4().hex[:8]}@example.com"
    password = "hunter2hunter2"
    await client.post("/auth/register", json={"email": email, "password": password})
    login = await client.post("/auth/login", json={"email": email, "password": password})
    return str(login.json()["access_token"])


@pytest.mark.asyncio
async def test_post_then_poll_reaches_success(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}

        post_response = await client.post(
            "/data-sources",
            json={
                "source_key": "return-policy",
                "scope": "tenant",
                "expected_change_interval_seconds": 90 * 24 * 60 * 60,
                "content": "Our return policy allows returns within 90 days.",
            },
            headers=headers,
        )
        assert post_response.status_code == 202
        task_id = post_response.json()["task_id"]

        deadline = time.monotonic() + 30
        status_body = None
        while time.monotonic() < deadline:
            status_response = await client.get(f"/data-sources/jobs/{task_id}", headers=headers)
            assert status_response.status_code == 200
            status_body = status_response.json()
            if status_body["state"] != "pending":
                break
            time.sleep(0.5)
        assert status_body is not None
        assert status_body["state"] == "success"
        assert status_body["result"]["changed"] is True


@pytest.mark.asyncio
async def test_a_client_cannot_declare_its_own_tenant_id(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.post(
            "/data-sources",
            json={
                "source_key": "return-policy",
                "scope": "tenant",
                "expected_change_interval_seconds": 3600,
                "content": "text",
                "tenant_id": str(uuid.uuid4()),
            },
            headers=headers,
        )
        assert response.status_code == 422  # extra="forbid" rejects the field outright


@pytest.mark.asyncio
async def test_checking_another_users_job_returns_404(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token_a = await _register_and_login(client)
        token_b = await _register_and_login(client)

        post_response = await client.post(
            "/data-sources",
            json={
                "source_key": "return-policy",
                "scope": "tenant",
                "expected_change_interval_seconds": 3600,
                "content": "text",
            },
            headers={"Authorization": f"Bearer {token_a}"},
        )
        task_id = post_response.json()["task_id"]

        cross_response = await client.get(
            f"/data-sources/jobs/{task_id}", headers={"Authorization": f"Bearer {token_b}"}
        )
        assert cross_response.status_code == 404


@pytest.mark.asyncio
async def test_checking_an_unknown_job_id_returns_404(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await _register_and_login(client)
        response = await client.get(
            f"/data-sources/jobs/{uuid.uuid4()}", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_ingestion_is_rate_limited_at_ten_per_minute(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "source_key": "return-policy",
            "scope": "tenant",
            "expected_change_interval_seconds": 3600,
            "content": "text",
        }
        responses = [
            await client.post("/data-sources", json=payload, headers=headers) for _ in range(11)
        ]
        assert responses[-1].status_code == 429


@pytest.mark.asyncio
async def test_an_oversized_body_is_rejected_before_parsing(celery_worker_for_ingestion):
    from src.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}", "Content-Length": str(12 * 1024 * 1024)}
        oversized = "x" * (12 * 1024 * 1024)
        response = await client.post("/data-sources", content=oversized, headers=headers)
        assert response.status_code == 413
