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
