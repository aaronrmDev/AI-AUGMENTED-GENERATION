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
# See the identical rationale in tests/integration/test_auth_endpoints.py.
pytestmark = pytest.mark.asyncio(loop_scope="module")


async def _client(app_database_url, redis_url, qdrant_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from src.api.dependencies import get_vector_store
    from src.api.main import app

    # httpx's ASGITransport does not invoke the app's startup lifespan, so
    # main.py's own ensure_qdrant_collection() startup hook never runs here —
    # call it directly rather than relying on another test file happening to
    # run first and create the collection as a side effect.
    await get_vector_store().ensure_collection()

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def _register_and_login(client) -> str:
    await client.post("/auth/register", json={"email": "rag@example.com", "password": "hunter2hunter2"})
    response = await client.post("/auth/login", json={"email": "rag@example.com", "password": "hunter2hunter2"})
    return response.json()["access_token"]


async def test_upload_then_search_finds_the_uploaded_content(app_database_url, redis_url, qdrant_url):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        token = await _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}

        upload_response = await client.post(
            "/documents",
            headers=headers,
            files={"file": ("notes.txt", b"FastAPI is a modern Python web framework.", "text/plain")},
        )
        assert upload_response.status_code == 201
        assert upload_response.json()["chunk_count"] >= 1

        search_response = await client.post(
            "/documents/search", headers=headers, json={"query": "Python web framework", "top_k": 5}
        )
        assert search_response.status_code == 200
        results = search_response.json()["results"]
        assert any("FastAPI" in r["content"] for r in results)


async def test_upload_without_a_token_returns_401(app_database_url, redis_url, qdrant_url):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        response = await client.post("/documents", files={"file": ("notes.txt", b"content", "text/plain")})
        assert response.status_code == 401


async def test_search_without_a_token_returns_401(app_database_url, redis_url, qdrant_url):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        response = await client.post("/documents/search", json={"query": "anything", "top_k": 5})
        assert response.status_code == 401


async def test_upload_rejects_an_unsupported_file_type(app_database_url, redis_url, qdrant_url):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        token = await _register_and_login(client)
        response = await client.post(
            "/documents",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("archive.docx", b"content", "application/octet-stream")},
        )
        assert response.status_code == 422
