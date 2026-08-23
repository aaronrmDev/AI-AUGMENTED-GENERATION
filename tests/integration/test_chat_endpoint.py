import os

from httpx import ASGITransport, AsyncClient


async def _client(app_database_url, redis_url, qdrant_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from src.api.dependencies import get_vector_store
    from src.api.main import app

    # See the matching comment in test_documents_endpoints.py's _client(): the
    # ASGITransport path never runs the app's startup lifespan.
    await get_vector_store().ensure_collection()

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_chat_without_a_token_returns_401(app_database_url, redis_url, qdrant_url):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        response = await client.post("/chat", json={"question": "What is FastAPI?"})
        assert response.status_code == 401
