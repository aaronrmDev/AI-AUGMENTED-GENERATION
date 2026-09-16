from datetime import UTC, datetime, timedelta

import redis.asyncio as redis

from src.identity.domain.ports import RateLimiter

_KEY_PREFIX = "identity:ratelimit:"


class RedisRateLimiter(RateLimiter):
    def __init__(self, redis_url: str) -> None:
        # redis.asyncio.utils.from_url ships with zero annotations in the redis 6.x
        # line (verified against the installed 6.4.0 wheel), unlike the 8.1.0 this
        # project ran on before celery[redis]>=5.4 (added for the ingestion job
        # dispatcher) pulled in kombu[redis], which caps redis at <6.5.
        self._client = redis.from_url(redis_url, decode_responses=True)  # type: ignore[no-untyped-call]

    async def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int, datetime]:
        redis_key = f"{_KEY_PREFIX}{key}"
        # One MULTI/EXEC, so the count and the window's expiry land together: a process
        # that dies mid-check can't leave a counter without a TTL, which would never reset
        # and would lock its owner out for good. EXPIRE NX sets the expiry only when the key
        # has none, so it also repairs a counter an interrupted check left behind.
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.incr(redis_key)
            pipe.expire(redis_key, window_seconds, nx=True)
            pipe.ttl(redis_key)
            count, _, ttl = await pipe.execute()

        reset_at = datetime.now(UTC) + timedelta(seconds=max(ttl, 0))
        if count > limit:
            return False, 0, reset_at
        return True, limit - count, reset_at

    async def aclose(self) -> None:
        await self._client.aclose()
