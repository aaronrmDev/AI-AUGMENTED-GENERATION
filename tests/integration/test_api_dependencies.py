import os

import pytest

# One event loop for the module, matching the app's module-level engine (see
# test_documents_endpoints.py).
pytestmark = pytest.mark.asyncio(loop_scope="module")


def _dependencies(app_database_url, redis_url, qdrant_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from src.api import dependencies

    return dependencies


async def test_the_redis_clients_are_built_once_per_process(
    app_database_url, redis_url, qdrant_url
):
    # A client per request opened a fresh connection pool on every rate-limited request,
    # and nothing ever closed it. Every answer in a session now passes through the limiter.
    dependencies = _dependencies(app_database_url, redis_url, qdrant_url)

    assert dependencies.get_rate_limiter() is dependencies.get_rate_limiter()
    assert dependencies.get_refresh_token_store() is dependencies.get_refresh_token_store()


async def test_closing_the_shared_clients_lets_the_next_call_build_fresh_ones(
    app_database_url, redis_url, qdrant_url
):
    dependencies = _dependencies(app_database_url, redis_url, qdrant_url)
    limiter = dependencies.get_rate_limiter()
    store = dependencies.get_refresh_token_store()

    await dependencies.close_redis_clients()

    assert dependencies.get_rate_limiter() is not limiter
    assert dependencies.get_refresh_token_store() is not store
    allowed, _, _ = await dependencies.get_rate_limiter().check("deps-key", 5, 60)
    assert allowed is True


async def test_a_client_that_fails_to_close_is_still_forgotten(
    app_database_url, redis_url, qdrant_url, monkeypatch
):
    # A failed close must not leave a dead client cached, where every later request
    # would pick it up.
    dependencies = _dependencies(app_database_url, redis_url, qdrant_url)
    limiter = dependencies.get_rate_limiter()
    store = dependencies.get_refresh_token_store()

    async def refuse_to_close() -> None:
        raise RuntimeError("close failed")

    monkeypatch.setattr(limiter, "aclose", refuse_to_close)

    with pytest.raises(RuntimeError, match="close failed"):
        await dependencies.close_redis_clients()

    assert dependencies.get_rate_limiter() is not limiter
    assert dependencies.get_refresh_token_store() is not store
