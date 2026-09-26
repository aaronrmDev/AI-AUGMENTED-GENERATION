import asyncio

import redis.asyncio as redis

from src.identity.infrastructure.redis_stream_concurrency_limiter import (
    RedisStreamConcurrencyLimiter,
)


async def test_allows_acquires_up_to_the_limit_then_blocks(redis_url):
    limiter = RedisStreamConcurrencyLimiter(redis_url)
    tokens = [await limiter.acquire("user-a", limit=3, ttl_seconds=60) for _ in range(3)]
    assert all(token is not None for token in tokens)

    blocked = await limiter.acquire("user-a", limit=3, ttl_seconds=60)

    assert blocked is None


async def test_a_released_slot_can_be_reacquired(redis_url):
    limiter = RedisStreamConcurrencyLimiter(redis_url)
    first = await limiter.acquire("user-b", limit=1, ttl_seconds=60)
    assert first is not None
    assert await limiter.acquire("user-b", limit=1, ttl_seconds=60) is None

    await limiter.release("user-b", first)

    second = await limiter.acquire("user-b", limit=1, ttl_seconds=60)
    assert second is not None


async def test_different_keys_have_independent_limits(redis_url):
    limiter = RedisStreamConcurrencyLimiter(redis_url)
    await limiter.acquire("user-c", limit=1, ttl_seconds=60)

    other = await limiter.acquire("user-d", limit=1, ttl_seconds=60)

    assert other is not None


async def test_a_rejected_acquire_does_not_leave_its_own_slot_behind(redis_url):
    # If the optimistic add-then-undo in acquire() ever forgot the undo half,
    # a rejected call would still leave its own uuid counted against the key --
    # eventually starving it out even though every acquire past the limit was
    # correctly rejected.
    limiter = RedisStreamConcurrencyLimiter(redis_url)
    await limiter.acquire("user-e", limit=1, ttl_seconds=60)
    for _ in range(5):
        assert await limiter.acquire("user-e", limit=1, ttl_seconds=60) is None

    client = redis.from_url(redis_url, decode_responses=True)
    try:
        assert await client.zcard("identity:streamslots:user-e") == 1
    finally:
        await client.aclose()


async def test_a_slot_that_outlives_its_ttl_is_not_counted_towards_the_limit(redis_url):
    limiter = RedisStreamConcurrencyLimiter(redis_url)
    expiring = await limiter.acquire("user-f", limit=1, ttl_seconds=0.05)
    assert expiring is not None

    await asyncio.sleep(0.15)

    # release() is deliberately never called for the expired slot: the whole
    # point is that its own score-as-expiry self-heals the count without it.
    fresh = await limiter.acquire("user-f", limit=1, ttl_seconds=60)
    assert fresh is not None


async def test_the_key_itself_carries_a_backstop_expiry(redis_url):
    limiter = RedisStreamConcurrencyLimiter(redis_url)
    await limiter.acquire("user-g", limit=5, ttl_seconds=30)

    client = redis.from_url(redis_url, decode_responses=True)
    try:
        ttl = await client.ttl("identity:streamslots:user-g")
        assert 0 < ttl <= 40
    finally:
        await client.aclose()


async def test_release_of_an_unknown_token_is_a_no_op(redis_url):
    limiter = RedisStreamConcurrencyLimiter(redis_url)
    # Doesn't raise, and doesn't affect a real slot for the same key.
    await limiter.release("user-h", "not-a-real-token")
    token = await limiter.acquire("user-h", limit=1, ttl_seconds=60)
    assert token is not None
