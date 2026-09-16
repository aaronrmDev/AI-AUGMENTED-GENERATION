import os

from httpx import ASGITransport, AsyncClient


async def _client(app_database_url, redis_url, qdrant_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from src.api.dependencies import get_vector_store
    from src.api.main import app

    await get_vector_store().ensure_collection()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_every_response_carries_the_security_header_baseline(
    app_database_url, redis_url, qdrant_url
):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        response = await client.get("/health")

    assert response.headers["Strict-Transport-Security"] == "max-age=63072000; includeSubDomains"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
