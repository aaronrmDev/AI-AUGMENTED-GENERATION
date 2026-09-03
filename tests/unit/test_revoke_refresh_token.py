import uuid

from src.identity.application.revoke_refresh_token import RevokeRefreshToken
from tests.unit.fakes import FakeRefreshTokenStore, FakeTokenIssuer


async def test_revoke_deletes_the_token_so_it_can_no_longer_be_used():
    store = FakeRefreshTokenStore()
    issuer = FakeTokenIssuer()
    user_id = uuid.uuid4()
    pair = issuer.issue_pair(user_id, uuid.uuid4())
    await store.save(pair.refresh_token, user_id)

    use_case = RevokeRefreshToken(refresh_token_store=store)
    await use_case.execute(refresh_token_id=pair.refresh_token.token_id)

    assert await store.get_user_id(pair.refresh_token.token_id) is None
