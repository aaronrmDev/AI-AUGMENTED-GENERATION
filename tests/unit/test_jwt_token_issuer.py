import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.identity.domain.errors import TokenExpired
from src.identity.infrastructure.jwt_token_issuer import JWTTokenIssuer

SECRET = "test-secret-key-not-for-production"


def test_issue_pair_returns_an_access_token_expiring_in_15_minutes():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    issuer = JWTTokenIssuer(secret_key=SECRET, clock=lambda: now)
    pair = issuer.issue_pair(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())
    assert pair.access_token.expires_at == now + timedelta(minutes=15)


def test_issue_pair_returns_a_refresh_token_expiring_in_7_days():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    issuer = JWTTokenIssuer(secret_key=SECRET, clock=lambda: now)
    pair = issuer.issue_pair(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())
    assert pair.refresh_token.expires_at == now + timedelta(days=7)


def test_verify_access_token_returns_the_claims_within_expiry():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    issuer = JWTTokenIssuer(secret_key=SECRET, clock=lambda: now)
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    pair = issuer.issue_pair(user_id=user_id, tenant_id=tenant_id)
    claims = issuer.verify_access_token(pair.access_token.value)
    assert claims["sub"] == str(user_id)
    assert claims["tenant_id"] == str(tenant_id)


def test_verify_access_token_rejects_a_token_issued_16_minutes_ago():
    issued_at = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    issuer_at_issue_time = JWTTokenIssuer(secret_key=SECRET, clock=lambda: issued_at)
    pair = issuer_at_issue_time.issue_pair(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())

    later = issued_at + timedelta(minutes=16)
    issuer_at_verify_time = JWTTokenIssuer(secret_key=SECRET, clock=lambda: later)
    with pytest.raises(TokenExpired):
        issuer_at_verify_time.verify_access_token(pair.access_token.value)


def test_verify_access_token_rejects_a_tampered_token():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    issuer = JWTTokenIssuer(secret_key=SECRET, clock=lambda: now)
    pair = issuer.issue_pair(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())

    header, payload, signature = pair.access_token.value.split(".")
    tampered_signature = ("A" if signature[0] != "A" else "B") + signature[1:]
    tampered = f"{header}.{payload}.{tampered_signature}"

    with pytest.raises(TokenExpired):
        issuer.verify_access_token(tampered)


def test_verify_access_token_rejects_a_refresh_token():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    issuer = JWTTokenIssuer(secret_key=SECRET, clock=lambda: now)
    pair = issuer.issue_pair(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())
    with pytest.raises(TokenExpired):
        issuer.verify_access_token(pair.refresh_token.value)
