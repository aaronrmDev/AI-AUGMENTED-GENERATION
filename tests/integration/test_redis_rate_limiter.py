import redis.asyncio as redis

from src.identity.infrastructure.redis_rate_limiter import RedisRateLimiter


async def test_allows_requests_under_the_limit(redis_url):
    limiter = RedisRateLimiter(redis_url)
    for _ in range(5):
        allowed, remaining, _ = await limiter.check("test-key-a", limit=5, window_seconds=60)
        assert allowed is True
    assert remaining == 0


async def test_blocks_the_request_that_exceeds_the_limit(redis_url):
    limiter = RedisRateLimiter(redis_url)
    for _ in range(5):
        await limiter.check("test-key-b", limit=5, window_seconds=60)
    allowed, remaining, _ = await limiter.check("test-key-b", limit=5, window_seconds=60)
    assert allowed is False
    assert remaining == 0


async def test_different_keys_have_independent_limits(redis_url):
    limiter = RedisRateLimiter(redis_url)
    for _ in range(5):
        await limiter.check("test-key-c", limit=5, window_seconds=60)
    allowed, _, _ = await limiter.check("test-key-d", limit=5, window_seconds=60)
    assert allowed is True


async def test_a_counter_left_without_an_expiry_gets_one_on_the_next_check(redis_url):
    # A process that dies between counting a request and setting the window's expiry
    # leaves a counter with no TTL. Unless the next check repairs it, the counter never
    # resets, and its owner stays rate limited for good once it passes the limit.
    client = redis.from_url(redis_url, decode_responses=True)
    await client.set("identity:ratelimit:test-key-e", 3)
    limiter = RedisRateLimiter(redis_url)
    try:
        await limiter.check("test-key-e", limit=5, window_seconds=60)
        assert 0 < await client.ttl("identity:ratelimit:test-key-e") <= 60
    finally:
        await limiter.aclose()
        await client.aclose()
