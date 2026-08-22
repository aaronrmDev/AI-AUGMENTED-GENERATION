import uuid

import pytest

from src.identity.application.refresh_access_token import RefreshAccessToken
from src.identity.domain.errors import TokenAlreadyUsed
from tests.unit.fakes import FakeRefreshTokenStore, FakeTokenIssuer


async def test_refresh_issues_a_new_pair_and_invalidates_the_old_token():
    store = FakeRefreshTokenStore()
    issuer = FakeTokenIssuer()
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    original_pair = issuer.issue_pair(user_id, tenant_id)
    await store.save(original_pair.refresh_token, user_id)

    use_case = RefreshAccessToken(refresh_token_store=store, token_issuer=issuer)
    new_pair = await use_case.execute(
        refresh_token_id=original_pair.refresh_token.token_id, tenant_id=tenant_id
    )

    assert new_pair.access_token.value.startswith("access-")
    assert await store.get_user_id(original_pair.refresh_token.token_id) is None


async def test_refresh_issues_the_new_access_token_against_the_tenant_it_was_given():
    """The tenant argument has to reach the issued token, not just be accepted.

    RefreshAccessToken takes tenant_id as a separate parameter because a
    refresh token deliberately carries no tenant claim of its own — the router
    looks the user's tenant up fresh and passes it in. Nothing else in the
    suite would notice if that argument were dropped, defaulted, or crossed
    with another user's, so this asserts the value actually lands in the new
    access token rather than merely that some token came back.

    Scope note: the tenant-scoped DB session wiring (get_db_session /
    set_tenant_context in src/api/dependencies.py) has no consumer yet — this
    sub-project ships no protected routes. The next sub-project's first
    protected route is where tenant propagation gets exercised end-to-end,
    against real RLS; this test covers the use-case half of that path now.
    """
    store = FakeRefreshTokenStore()
    issuer = FakeTokenIssuer()
    user_id = uuid.uuid4()
    original_tenant_id = uuid.uuid4()
    original_pair = issuer.issue_pair(user_id, original_tenant_id)
    await store.save(original_pair.refresh_token, user_id)

    use_case = RefreshAccessToken(refresh_token_store=store, token_issuer=issuer)
    new_pair = await use_case.execute(
        refresh_token_id=original_pair.refresh_token.token_id, tenant_id=original_tenant_id
    )

    # FakeTokenIssuer encodes the pair it was called with as
    # "access-{user_id}-{tenant_id}", so the value is a direct readout of the
    # arguments the use case actually passed down to the issuer.
    assert new_pair.access_token.value == f"access-{user_id}-{original_tenant_id}"


async def test_refresh_does_not_reuse_the_tenant_from_the_previous_token():
    """A different tenant on the way in must produce a different tenant on the way out.

    The positive test above would still pass if the issuer somehow echoed the
    original pair's tenant back; this pins the value to the argument of *this*
    call by passing a tenant that differs from the one the original token was
    minted with.
    """
    store = FakeRefreshTokenStore()
    issuer = FakeTokenIssuer()
    user_id = uuid.uuid4()
    original_tenant_id = uuid.uuid4()
    other_tenant_id = uuid.uuid4()
    original_pair = issuer.issue_pair(user_id, original_tenant_id)
    await store.save(original_pair.refresh_token, user_id)

    use_case = RefreshAccessToken(refresh_token_store=store, token_issuer=issuer)
    new_pair = await use_case.execute(
        refresh_token_id=original_pair.refresh_token.token_id, tenant_id=other_tenant_id
    )

    assert new_pair.access_token.value == f"access-{user_id}-{other_tenant_id}"
    assert str(original_tenant_id) not in new_pair.access_token.value


async def test_refresh_rejects_an_already_used_token():
    store = FakeRefreshTokenStore()
    issuer = FakeTokenIssuer()
    use_case = RefreshAccessToken(refresh_token_store=store, token_issuer=issuer)

    with pytest.raises(TokenAlreadyUsed):
        await use_case.execute(refresh_token_id=uuid.uuid4(), tenant_id=uuid.uuid4())


async def test_refresh_rejects_the_same_token_replayed_a_second_time():
    store = FakeRefreshTokenStore()
    issuer = FakeTokenIssuer()
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    original_pair = issuer.issue_pair(user_id, tenant_id)
    await store.save(original_pair.refresh_token, user_id)
    use_case = RefreshAccessToken(refresh_token_store=store, token_issuer=issuer)

    await use_case.execute(
        refresh_token_id=original_pair.refresh_token.token_id, tenant_id=tenant_id
    )

    with pytest.raises(TokenAlreadyUsed):
        await use_case.execute(
            refresh_token_id=original_pair.refresh_token.token_id, tenant_id=tenant_id
        )
