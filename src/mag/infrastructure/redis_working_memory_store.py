import json
import uuid
from datetime import datetime

import redis.asyncio as redis

from src.mag.domain.entities import WorkingMemoryTurn
from src.mag.domain.ports import WorkingMemoryStore

# docs/architecture/MAG.md's tier table scopes the working-memory tier at
# "seconds-minutes" to "hours-days" -- 24 hours is a concrete point in that
# range, refreshed on every push so an active session never expires mid-use.
_TTL_SECONDS = 86400


class RedisWorkingMemoryStore(WorkingMemoryStore):
    def __init__(self, redis_url: str) -> None:
        self._client = redis.from_url(redis_url, decode_responses=True)

    async def push_turn(self, session_id: uuid.UUID, turn: WorkingMemoryTurn) -> None:
        key = self._key(session_id)
        payload = json.dumps(
            {
                "role": turn.role,
                "content": turn.content,
                "recorded_at": turn.recorded_at.isoformat(),
                "metadata": turn.metadata,
            }
        )
        # Pipelined so RPUSH and EXPIRE reach Redis as one round trip and one
        # unit of work -- two separate awaits left a window where a process
        # crash between them left a newly created key with no TTL, leaking it
        # forever in the tier whose entire point is bounded lifetime.
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, payload)
            pipe.expire(key, _TTL_SECONDS)
            await pipe.execute()

    async def get_recent_turns(
        self, session_id: uuid.UUID, limit: int
    ) -> list[WorkingMemoryTurn]:
        # limit <= 0 must return nothing, not everything: Redis's LRANGE
        # treats a start index of -0 as plain 0, its idiom for "the whole
        # list" -- so a naive -limit here silently ignores a caller that
        # asked for zero turns and hands back the entire session instead.
        if limit <= 0:
            return []
        raw_entries = await self._client.lrange(self._key(session_id), -limit, -1)
        return [self._deserialize(entry) for entry in raw_entries]

    @staticmethod
    def _key(session_id: uuid.UUID) -> str:
        return f"session:{session_id}:working_memory"

    @staticmethod
    def _deserialize(entry: bytes | str) -> WorkingMemoryTurn:
        # redis-py types list elements as `bytes | str` because the actual
        # return type depends on the `decode_responses` flag passed to
        # from_url(), which it can't see -- this client sets it True, so the
        # str branch is the one that actually runs (see the identical
        # reasoning in RedisRefreshTokenStore.get_user_id).
        if isinstance(entry, bytes):
            entry = entry.decode()
        data = json.loads(entry)
        return WorkingMemoryTurn(
            role=data["role"],
            content=data["content"],
            recorded_at=datetime.fromisoformat(data["recorded_at"]),
            metadata=data["metadata"],
        )
