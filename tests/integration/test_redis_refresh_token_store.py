import uuid
from datetime import UTC, datetime, timedelta

import pytest_asyncio

from src.identity.domain.entities import RefreshToken
from src.identity.infrastructure.redis_refresh_token_store import RedisRefreshTokenStore


@pytest_asyncio.fixture
async def store(redis_url):
    s = RedisRefreshTokenStore(redis_url)
    yield s
    await s._client.aclose()


async def test_save_then_get_user_id_returns_the_owning_user(store):
    token_id = uuid.uuid4()
    user_id = uuid.uuid4()
    token = RefreshToken(
        token_id=token_id, value="opaque", expires_at=datetime.now(UTC) + timedelta(days=7)
    )

    await store.save(token, user_id)
    assert await store.get_user_id(token_id) == user_id


async def test_get_user_id_returns_none_for_an_unknown_token(store):
    assert await store.get_user_id(uuid.uuid4()) is None


async def test_delete_makes_the_token_unfindable(store):
    token_id = uuid.uuid4()
    user_id = uuid.uuid4()
    token = RefreshToken(
        token_id=token_id, value="opaque", expires_at=datetime.now(UTC) + timedelta(days=7)
    )

    await store.save(token, user_id)
    await store.delete(token_id)
    assert await store.get_user_id(token_id) is None


async def test_save_sets_a_ttl_matching_the_token_lifetime(redis_url):
    import redis.asyncio as redis

    store = RedisRefreshTokenStore(redis_url)
    token_id = uuid.uuid4()
    expires_at = datetime.now(UTC) + timedelta(days=7)
    token = RefreshToken(token_id=token_id, value="opaque", expires_at=expires_at)
    await store.save(token, uuid.uuid4())

    client = redis.from_url(redis_url)
    ttl = await client.ttl(f"identity:refresh:{token_id}")
    await client.aclose()
    await store._client.aclose()

    seven_days_seconds = 7 * 24 * 60 * 60
    assert seven_days_seconds - 60 < ttl <= seven_days_seconds
