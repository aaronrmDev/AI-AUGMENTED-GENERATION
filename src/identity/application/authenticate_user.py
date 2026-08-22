from src.identity.domain.entities import PasswordHash, TokenPair, User
from src.identity.domain.errors import InvalidCredentials
from src.identity.domain.ports import PasswordHasher, TokenIssuer, UserRepository

# A syntactically valid Argon2id hash that no password can ever match: its
# encoded digest is the literal bytes "hashvalue", not a digest of anything.
# Its cost parameters (m=65536, t=3, p=4) are argon2-cffi's own defaults, the
# same ones Argon2PasswordHasher uses, so verifying against it costs the same
# as verifying against a real stored hash.
#
# Why this exists: the unknown-email branch below used to short-circuit before
# any hashing happened, so a request for an unregistered address returned in
# microseconds while a request for a registered one paid Argon2id's full
# ~35ms. That gap is trivially measurable over a network, and it turns a
# deliberately generic 401 into a working "is this email registered?" oracle —
# defeating the same disclosure this endpoint's uniform response shape exists
# to prevent. Running the hasher against this dummy makes both branches pay
# the same cost, so only the response (identical in both cases) is observable.
_DUMMY_HASH = PasswordHash("$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl")


class AuthenticateUser:
    def __init__(
        self,
        user_repository: UserRepository,
        password_hasher: PasswordHasher,
        token_issuer: TokenIssuer,
    ) -> None:
        self._users = user_repository
        self._hasher = password_hasher
        self._tokens = token_issuer

    async def execute(self, email: str, plain_password: str) -> tuple[User, TokenPair]:
        user = await self._users.find_by_email(email)
        if user is None:
            # Deliberately not short-circuited — see _DUMMY_HASH above. The
            # return value is discarded because it is always False; the point
            # is the elapsed time, not the answer.
            self._hasher.verify(plain_password, _DUMMY_HASH)
            raise InvalidCredentials()
        if not self._hasher.verify(plain_password, user.hashed_password):
            raise InvalidCredentials()
        pair = self._tokens.issue_pair(user_id=user.id, tenant_id=user.tenant_id)
        return user, pair
