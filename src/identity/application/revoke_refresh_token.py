import uuid

from src.identity.domain.ports import RefreshTokenStore


class RevokeRefreshToken:
    def __init__(self, refresh_token_store: RefreshTokenStore) -> None:
        self._store = refresh_token_store

    async def execute(self, refresh_token_id: uuid.UUID) -> None:
        await self._store.delete(refresh_token_id)
