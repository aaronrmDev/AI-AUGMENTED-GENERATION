from datetime import UTC, datetime, timedelta

import redis.asyncio as redis

from src.identity.domain.ports import RateLimiter

_KEY_PREFIX = "identity:ratelimit:"


class RedisRateLimiter(RateLimiter):
    def __init__(self, redis_url: str) -> None:
        self._client = redis.from_url(redis_url, decode_responses=True)

    async def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int, datetime]:
        redis_key = f"{_KEY_PREFIX}{key}"
        count = await self._client.incr(redis_key)
        if count == 1:
            await self._client.expire(redis_key, window_seconds)

        ttl = await self._client.ttl(redis_key)
        reset_at = datetime.now(UTC) + timedelta(seconds=max(ttl, 0))

        if count > limit:
            return False, 0, reset_at
        return True, limit - count, reset_at
