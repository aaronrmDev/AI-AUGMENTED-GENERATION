import pytest

from src.identity.application.register_user import RegisterUser
from src.identity.domain.errors import EmailAlreadyRegistered
from tests.unit.fakes import FakePasswordHasher, FakeUserRepository


async def test_register_creates_a_user_with_a_hashed_password():
    repo = FakeUserRepository()
    use_case = RegisterUser(user_repository=repo, password_hasher=FakePasswordHasher())

    user = await use_case.execute(email="new@example.com", plain_password="hunter2")

    stored = await repo.find_by_email("new@example.com")
    assert stored is not None
    assert stored.id == user.id
    assert str(stored.hashed_password) != "hunter2"


async def test_register_rejects_a_duplicate_email():
    repo = FakeUserRepository()
    use_case = RegisterUser(user_repository=repo, password_hasher=FakePasswordHasher())
    await use_case.execute(email="dup@example.com", plain_password="hunter2")

    with pytest.raises(EmailAlreadyRegistered):
        await use_case.execute(email="dup@example.com", plain_password="different")


async def test_register_gives_each_new_user_their_own_tenant():
    repo = FakeUserRepository()
    use_case = RegisterUser(user_repository=repo, password_hasher=FakePasswordHasher())

    first = await use_case.execute(email="a@example.com", plain_password="hunter2")
    second = await use_case.execute(email="b@example.com", plain_password="hunter2")

    assert first.tenant_id != second.tenant_id
