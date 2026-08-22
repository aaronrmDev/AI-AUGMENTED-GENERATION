import uuid

from src.identity.domain.entities import TokenPair
from src.identity.domain.errors import TokenAlreadyUsed
from src.identity.domain.ports import RefreshTokenStore, TokenIssuer


class RefreshAccessToken:
    def __init__(self, refresh_token_store: RefreshTokenStore, token_issuer: TokenIssuer) -> None:
        self._store = refresh_token_store
        self._tokens = token_issuer

    async def execute(self, refresh_token_id: uuid.UUID, tenant_id: uuid.UUID) -> TokenPair:
        user_id = await self._store.get_user_id(refresh_token_id)
        if user_id is None:
            raise TokenAlreadyUsed()

        await self._store.delete(refresh_token_id)
        new_pair = self._tokens.issue_pair(user_id=user_id, tenant_id=tenant_id)
        await self._store.save(new_pair.refresh_token, user_id)
        return new_pair
