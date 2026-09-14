import uuid

import pytest

from src.api.caller import Caller, caller_from_claims
from src.identity.domain.errors import TokenExpired


def test_the_caller_is_the_tokens_subject_in_the_tokens_tenant():
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    claims = {"sub": str(user_id), "tenant_id": str(tenant_id), "type": "access"}
    assert caller_from_claims(claims) == Caller(tenant_id=tenant_id, user_id=user_id)


@pytest.mark.parametrize(
    "claims", [{"tenant_id": str(uuid.uuid4())}, {"sub": "not-a-uuid", "tenant_id": "x"}]
)
def test_claims_without_a_usable_subject_or_tenant_are_an_invalid_token(claims):
    with pytest.raises(TokenExpired):
        caller_from_claims(claims)
