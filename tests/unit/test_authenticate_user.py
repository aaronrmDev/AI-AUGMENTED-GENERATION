import pytest

from src.identity.application.authenticate_user import AuthenticateUser
from src.identity.application.register_user import RegisterUser
from src.identity.domain.errors import InvalidCredentials
from tests.unit.fakes import FakePasswordHasher, FakeTokenIssuer, FakeUserRepository


async def test_authenticate_succeeds_with_correct_credentials_and_returns_the_user_and_tokens():
    repo = FakeUserRepository()
    hasher = FakePasswordHasher()
    registered = await RegisterUser(user_repository=repo, password_hasher=hasher).execute(
        email="a@example.com", plain_password="hunter2"
    )
    use_case = AuthenticateUser(
        user_repository=repo, password_hasher=hasher, token_issuer=FakeTokenIssuer()
    )

    user, pair = await use_case.execute(email="a@example.com", plain_password="hunter2")

    assert user.id == registered.id
    assert pair.access_token.value.startswith("access-")


async def test_authenticate_rejects_a_wrong_password():
    repo = FakeUserRepository()
    hasher = FakePasswordHasher()
    await RegisterUser(user_repository=repo, password_hasher=hasher).execute(
        email="a@example.com", plain_password="hunter2"
    )
    use_case = AuthenticateUser(
        user_repository=repo, password_hasher=hasher, token_issuer=FakeTokenIssuer()
    )

    with pytest.raises(InvalidCredentials):
        await use_case.execute(email="a@example.com", plain_password="wrong")


async def test_authenticate_rejects_an_unknown_email_with_the_same_error_as_a_wrong_password():
    repo = FakeUserRepository()
    hasher = FakePasswordHasher()
    use_case = AuthenticateUser(
        user_repository=repo, password_hasher=hasher, token_issuer=FakeTokenIssuer()
    )

    with pytest.raises(InvalidCredentials) as unknown_email_exc:
        await use_case.execute(email="nobody@example.com", plain_password="whatever")

    assert str(unknown_email_exc.value) == "Invalid credentials"


async def test_authenticate_hashes_exactly_once_for_an_unknown_email_just_like_a_wrong_password():
    """The unknown-email and wrong-password paths must cost the same, not just look the same.

    Returning an identical 401 body from both branches only closes half the
    disclosure: if the unknown-email branch skips password verification, it
    returns in microseconds while the wrong-password branch pays Argon2id's
    full ~35ms, and that difference is measurable over a network — turning the
    endpoint into a working "is this email registered?" oracle.

    Counting verify() calls, rather than timing them, is what makes this
    assertion deterministic: with a real hasher the two branches would differ
    by scheduler noise, but "one hash computed either way" is the invariant
    that actually produces equal cost, and it holds exactly.
    """
    repo = FakeUserRepository()
    hasher = FakePasswordHasher()
    await RegisterUser(user_repository=repo, password_hasher=hasher).execute(
        email="known@example.com", plain_password="hunter2"
    )
    # RegisterUser hashes on the way in; only what happens after this point is
    # what the two authenticate branches are being compared on.
    hasher.verify_call_count = 0
    use_case = AuthenticateUser(
        user_repository=repo, password_hasher=hasher, token_issuer=FakeTokenIssuer()
    )

    with pytest.raises(InvalidCredentials):
        await use_case.execute(email="known@example.com", plain_password="wrong")
    wrong_password_verifies = hasher.verify_call_count

    hasher.verify_call_count = 0
    with pytest.raises(InvalidCredentials):
        await use_case.execute(email="nobody@example.com", plain_password="wrong")
    unknown_email_verifies = hasher.verify_call_count

    assert wrong_password_verifies == 1
    assert unknown_email_verifies == 1
    assert unknown_email_verifies == wrong_password_verifies
