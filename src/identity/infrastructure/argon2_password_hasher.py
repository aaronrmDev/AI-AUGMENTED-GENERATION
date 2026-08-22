from argon2 import PasswordHasher as _Argon2PasswordHasher
from argon2.exceptions import VerifyMismatchError

from src.identity.domain.entities import PasswordHash
from src.identity.domain.ports import PasswordHasher


class Argon2PasswordHasher(PasswordHasher):
    def __init__(self) -> None:
        self._hasher = _Argon2PasswordHasher()

    def hash(self, plain_password: str) -> PasswordHash:
        return PasswordHash(self._hasher.hash(plain_password))

    def verify(self, plain_password: str, hashed: PasswordHash) -> bool:
        try:
            return self._hasher.verify(str(hashed), plain_password)
        except VerifyMismatchError:
            return False
