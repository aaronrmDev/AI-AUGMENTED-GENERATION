import uuid
from dataclasses import dataclass
from typing import Any

from src.identity.domain.errors import TokenExpired


@dataclass(frozen=True)
class Caller:
    tenant_id: uuid.UUID
    user_id: uuid.UUID


def caller_from_claims(claims: dict[str, Any]) -> Caller:
    """The caller is the verified token's subject, in the token's tenant, and nothing a
    client sends besides the token. Claims this issuer would never mint surface as the
    same invalid-token error a bad signature does."""
    try:
        return Caller(tenant_id=uuid.UUID(claims["tenant_id"]), user_id=uuid.UUID(claims["sub"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise TokenExpired() from exc
