import uuid
from datetime import UTC, datetime

import redis.asyncio as redis

from src.identity.domain.entities import RefreshToken
from src.identity.domain.ports import RefreshTokenStore

_KEY_PREFIX = "identity:refresh:"


class RedisRefreshTokenStore(RefreshTokenStore):
    def __init__(self, redis_url: str) -> None:
        self._client = redis.from_url(redis_url, decode_responses=True)

    async def save(self, refresh_token: RefreshToken, user_id: uuid.UUID) -> None:
        ttl_seconds = int((refresh_token.expires_at - datetime.now(UTC)).total_seconds())
        await self._client.set(
            f"{_KEY_PREFIX}{refresh_token.token_id}",
            str(user_id),
            ex=max(ttl_seconds, 1),
        )

    async def get_user_id(self, token_id: uuid.UUID) -> uuid.UUID | None:
        value = await self._client.get(f"{_KEY_PREFIX}{token_id}")
        if not value:
            return None
        # redis-py types get() as returning `bytes | str` because the return
        # type depends on the `decode_responses` flag passed to from_url(),
        # which the type system can't see. This client sets it True, so the
        # str branch is the one that actually runs — but decoding the bytes
        # branch rather than asserting it away keeps this correct even if that
        # constructor flag is ever changed, and costs one line.
        if isinstance(value, bytes):
            value = value.decode()
        return uuid.UUID(value)

    async def delete(self, token_id: uuid.UUID) -> None:
        await self._client.delete(f"{_KEY_PREFIX}{token_id}")
