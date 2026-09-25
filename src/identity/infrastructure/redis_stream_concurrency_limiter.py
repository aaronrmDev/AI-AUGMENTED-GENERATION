import time
import uuid

import redis.asyncio as redis

from src.identity.domain.ports import StreamConcurrencyLimiter

_KEY_PREFIX = "identity:streamslots:"
_KEY_EXPIRY_BUFFER_SECONDS = 5


class RedisStreamConcurrencyLimiter(StreamConcurrencyLimiter):
    def __init__(self, redis_url: str) -> None:
        # redis.asyncio.utils.from_url ships with zero annotations in the redis 6.x
        # line this project runs on -- see RedisRateLimiter's identical note.
        self._client = redis.from_url(redis_url, decode_responses=True)  # type: ignore[no-untyped-call]

    async def acquire(self, key: str, limit: int, ttl_seconds: float) -> str | None:
        redis_key = f"{_KEY_PREFIX}{key}"
        slot_token = uuid.uuid4().hex
        now = time.time()
        # Mirrors RedisRateLimiter.check(): add this acquire's own slot
        # optimistically in the same MULTI/EXEC as the expired-slot cleanup and
        # the count read, then undo it below if that count is over limit --
        # rather than a read-then-write race. Self-healing two ways: each
        # slot's score is its own expiry (cleared by ZREMRANGEBYSCORE on the
        # very next acquire from anyone), and the key-level EXPIRE clears the
        # whole thing even if nobody ever calls acquire() again for this key.
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(redis_key, "-inf", now)
            pipe.zadd(redis_key, {slot_token: now + ttl_seconds})
            pipe.zcard(redis_key)
            pipe.expire(redis_key, int(ttl_seconds) + _KEY_EXPIRY_BUFFER_SECONDS)
            _, _, count, _ = await pipe.execute()

        if count > limit:
            await self._client.zrem(redis_key, slot_token)
            return None
        return slot_token

    async def release(self, key: str, slot_token: str) -> None:
        await self._client.zrem(f"{_KEY_PREFIX}{key}", slot_token)

    async def aclose(self) -> None:
        await self._client.aclose()
