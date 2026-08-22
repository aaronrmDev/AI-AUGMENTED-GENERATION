from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from jwt import InvalidTokenError

from src.identity.domain.entities import AccessToken, RefreshToken, TokenPair
from src.identity.domain.errors import TokenExpired
from src.identity.domain.ports import TokenIssuer

_ACCESS_TOKEN_LIFETIME = timedelta(minutes=15)
_REFRESH_TOKEN_LIFETIME = timedelta(days=7)
_ALGORITHM = "HS256"


def _default_clock() -> datetime:
    return datetime.now(UTC)


class JWTTokenIssuer(TokenIssuer):
    def __init__(self, secret_key: str, clock: Callable[[], datetime] = _default_clock) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def issue_pair(self, user_id: uuid.UUID, tenant_id: uuid.UUID) -> TokenPair:
        now = self._clock()
        access_expires_at = now + _ACCESS_TOKEN_LIFETIME
        access_claims = {
            "sub": str(user_id),
            "tenant_id": str(tenant_id),
            "exp": int(access_expires_at.timestamp()),
            "iat": int(now.timestamp()),
            "type": "access",
        }
        access_value = jwt.encode(access_claims, self._secret_key, algorithm=_ALGORITHM)

        refresh_expires_at = now + _REFRESH_TOKEN_LIFETIME
        token_id = uuid.uuid4()
        refresh_claims = {
            "sub": str(user_id),
            "jti": str(token_id),
            "exp": int(refresh_expires_at.timestamp()),
            "iat": int(now.timestamp()),
            "type": "refresh",
        }
        refresh_value = jwt.encode(refresh_claims, self._secret_key, algorithm=_ALGORITHM)

        return TokenPair(
            access_token=AccessToken(value=access_value, expires_at=access_expires_at),
            refresh_token=RefreshToken(
                token_id=token_id, value=refresh_value, expires_at=refresh_expires_at
            ),
        )

    def verify_access_token(self, token: str) -> dict[str, Any]:
        try:
            now = self._clock()
            claims: dict[str, Any] = jwt.decode(
                token,
                self._secret_key,
                algorithms=[_ALGORITHM],
                options={"verify_exp": False},
            )
            # Manually verify expiry using the injected clock
            if "exp" in claims:
                exp_time = claims["exp"]
                if now.timestamp() > exp_time:
                    raise jwt.ExpiredSignatureError("Token has expired")
            # Verify token type is access token, not refresh token
            if claims.get("type") != "access":
                raise jwt.InvalidTokenError("Invalid token type")
            return claims
        except InvalidTokenError as exc:
            raise TokenExpired() from exc
