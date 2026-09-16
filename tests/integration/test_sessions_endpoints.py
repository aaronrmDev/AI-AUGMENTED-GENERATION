import json
import os
import uuid
from datetime import UTC, datetime

import pytest
import redis.asyncio as redis
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from src.api.rate_limit import GLOBAL_CHAT_KEY
from src.identity.infrastructure.db import set_tenant_context
from src.identity.infrastructure.jwt_token_issuer import JWTTokenIssuer
from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.rag.domain.entities import Chunk
from tests.integration.orchestration_env import VALID_HASH, ContextEchoChatModel

# One event loop for the module, matching the app's module-level engine (see
# test_documents_endpoints.py).
pytestmark = pytest.mark.asyncio(loop_scope="module")

_SECRET = "test-secret-key"
_POLICY = "Our return policy allows returns of unopened items within forty-five days."
_NOT_FOUND = {"detail": "Session not found"}
# Mirrors RedisRateLimiter's own private _KEY_PREFIX -- there's no public
# constant for it, and reading the counter directly is the only way to prove
# a rejected request never incremented it.
_GLOBAL_CHAT_REDIS_KEY = f"identity:ratelimit:{GLOBAL_CHAT_KEY}"


@pytest.fixture(autouse=True)
def _default_chat_rate_limit():
    previous = os.environ.pop("CHAT_RATE_LIMIT_PER_MINUTE", None)
    yield
    os.environ.pop("CHAT_RATE_LIMIT_PER_MINUTE", None)
    if previous is not None:
        os.environ["CHAT_RATE_LIMIT_PER_MINUTE"] = previous


@pytest.fixture(autouse=True)
def _default_global_chat_rate_limit():
    previous = os.environ.pop("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", None)
    yield
    os.environ.pop("CHAT_RATE_LIMIT_GLOBAL_PER_HOUR", None)
    if previous is not None:
        os.environ["CHAT_RATE_LIMIT_GLOBAL_PER_HOUR"] = previous


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    import sys

    main = sys.modules.get("src.api.main")
    if main is not None:
        main.app.dependency_overrides.clear()


async def _client(app_database_url, redis_url, qdrant_url, embedding_model):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["JWT_SECRET_KEY"] = _SECRET
    from src.api import dependencies
    from src.api.main import app
    from src.api.unified_pipeline import build_unified_pipeline

    await dependencies.get_vector_store().ensure_collection()
    # The real composition with a context-echo model: every store, the router, and the
    # budget recorder are real; only generation is replaced, so the answer shows the
    # context it was given.
    pipeline = build_unified_pipeline(
        sessionmaker=dependencies._sessionmaker,
        sessions=dependencies.get_chat_session_repository(),
        embedding_model=embedding_model,
        vector_store=dependencies.get_vector_store(),
        chat_model=ContextEchoChatModel(),
    )
    app.dependency_overrides[dependencies.get_answer_in_session] = (
        lambda: pipeline.answer_in_session
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _user(db_session, tenant_id: uuid.UUID) -> uuid.UUID:
    now, user_id = datetime.now(UTC), uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id, "created_at": now, "updated_at": now,
        },
    )
    await db_session.commit()
    return user_id


def _auth(user_id: uuid.UUID, tenant_id: uuid.UUID) -> dict[str, str]:
    token = JWTTokenIssuer(secret_key=_SECRET).issue_pair(user_id, tenant_id).access_token.value
    return {"Authorization": f"Bearer {token}"}


async def _stored_budget(db_session, tenant_id: uuid.UUID, session_id: str):
    await set_tenant_context(db_session, tenant_id)
    value = (
        await db_session.execute(
            text("SELECT context_budget FROM sessions WHERE id = :id"),
            {"id": uuid.UUID(session_id)},
        )
    ).scalar_one()
    await db_session.commit()
    return json.loads(value) if isinstance(value, str) else value


async def _seed_policy(tenant_id: uuid.UUID, embedding_model) -> None:
    from src.api.dependencies import get_vector_store

    chunk = Chunk(uuid.uuid4(), uuid.uuid4(), _POLICY, embedding_model.embed(_POLICY))
    await get_vector_store().upsert(chunk, tenant_id)


async def test_every_sessions_route_refuses_a_request_without_a_token(
    app_database_url, redis_url, qdrant_url, embedding_model
):
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        responses = [
            await client.post("/sessions", json={}),
            await client.get("/sessions"),
            await client.post(f"/sessions/{uuid.uuid4()}/answers", json={"question": "q"}),
        ]
    assert [r.status_code for r in responses] == [401, 401, 401]


async def test_created_sessions_are_listed_for_their_owner_newest_first(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        first = await client.post("/sessions", json={"title": "first"}, headers=headers)
        second = await client.post("/sessions", json={"title": "  second  "}, headers=headers)
        listed = await client.get("/sessions", headers=headers)

    assert (first.status_code, second.status_code) == (201, 201)
    assert first.headers["X-RateLimit-Limit"] == "20"
    sessions = listed.json()["sessions"]
    assert [s["id"] for s in sessions] == [second.json()["id"], first.json()["id"]]
    assert [s["title"] for s in sessions] == ["second", "first"]
    assert all(s["context_budget"] is None for s in sessions)


async def test_answering_in_an_owned_session_routes_cascades_and_records_the_budget(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    user_id = await _user(db_session, tenant_id)
    preference = "The user prefers expedited two-day shipping."
    await set_tenant_context(db_session, tenant_id)
    await PostgresSemanticMemoryRepository(db_session).save(
        SemanticMemory(
            id=uuid.uuid4(),
            user_id=user_id,
            fact_key="preferred_shipping_speed",
            fact_value=preference,
            embedding=embedding_model.embed(preference),
        ),
        tenant_id,
    )
    await db_session.commit()
    await _seed_policy(tenant_id, embedding_model)
    headers = _auth(user_id, tenant_id)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        response = await client.post(
            f"/sessions/{session_id}/answers",
            json={"question": "What is the return policy for unopened items?"},
            headers=headers,
        )
        listed = await client.get("/sessions", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert "forty-five days" in body["answer"]
    assert any(s["paradigm"] == "rag" and "forty-five" in s["content"] for s in body["sources"])
    assert body["routing"]["mode"] in {"cascade", "parallel"}
    attempted = {a["paradigm"] for a in body["attempts"]}
    assert "rag" in attempted
    assert "cag" not in attempted
    assert response.headers["X-RateLimit-Limit"] == "100"
    assert (await _stored_budget(db_session, tenant_id, session_id))["total"] == 128_000
    assert listed.json()["sessions"][0]["context_budget"]["total"] == 128_000


async def test_another_users_session_in_the_same_tenant_is_not_found_and_left_untouched(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    owner, intruder = await _user(db_session, tenant_id), await _user(db_session, tenant_id)
    await _seed_policy(tenant_id, embedding_model)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (
            await client.post("/sessions", json={}, headers=_auth(owner, tenant_id))
        ).json()["id"]
        response = await client.post(
            f"/sessions/{session_id}/answers",
            json={"question": "What is the return policy?"},
            headers=_auth(intruder, tenant_id),
        )

    assert (response.status_code, response.json()) == (404, _NOT_FOUND)
    assert await _stored_budget(db_session, tenant_id, session_id) is None


async def test_another_tenant_and_an_unknown_id_get_the_identical_404(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id, other_tenant = uuid.uuid4(), uuid.uuid4()
    owner = await _user(db_session, tenant_id)
    stranger = await _user(db_session, other_tenant)
    question = {"question": "What is the return policy?"}
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (
            await client.post("/sessions", json={}, headers=_auth(owner, tenant_id))
        ).json()["id"]
        foreign = await client.post(
            f"/sessions/{session_id}/answers",
            json=question,
            headers=_auth(stranger, other_tenant),
        )
        unknown = await client.post(
            f"/sessions/{uuid.uuid4()}/answers", json=question, headers=_auth(owner, tenant_id)
        )

    assert (foreign.status_code, foreign.content) == (unknown.status_code, unknown.content)
    assert (foreign.status_code, foreign.json()) == (404, _NOT_FOUND)
    assert await _stored_budget(db_session, tenant_id, session_id) is None


async def test_listing_never_shows_another_users_or_another_tenants_sessions(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id, other_tenant = uuid.uuid4(), uuid.uuid4()
    owner, colleague = await _user(db_session, tenant_id), await _user(db_session, tenant_id)
    stranger = await _user(db_session, other_tenant)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        mine = await client.post(
            "/sessions", json={"title": "mine"}, headers=_auth(owner, tenant_id)
        )
        await client.post("/sessions", json={}, headers=_auth(colleague, tenant_id))
        await client.post("/sessions", json={}, headers=_auth(stranger, other_tenant))
        listed = await client.get("/sessions", headers=_auth(owner, tenant_id))

    assert [s["id"] for s in listed.json()["sessions"]] == [mine.json()["id"]]


async def test_the_chat_limit_is_shared_between_answering_and_post_chat(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    os.environ["CHAT_RATE_LIMIT_PER_MINUTE"] = "2"
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    await _seed_policy(tenant_id, embedding_model)
    question = {"question": "What is the return policy?"}
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        answers = [
            await client.post(f"/sessions/{session_id}/answers", json=question, headers=headers)
            for _ in range(3)
        ]
        # Refused before its RAG-only pipeline runs, so no chat model is needed.
        chat = await client.post("/chat", json=question, headers=headers)

    assert [a.status_code for a in answers] == [200, 200, 429]
    assert answers[2].headers["X-RateLimit-Remaining"] == "0"
    assert chat.status_code == 429


async def test_the_global_chat_quota_is_shared_across_every_account(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    os.environ["CHAT_RATE_LIMIT_GLOBAL_PER_HOUR"] = "2"
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    headers_a = _auth(await _user(db_session, tenant_a), tenant_a)
    headers_b = _auth(await _user(db_session, tenant_b), tenant_b)
    await _seed_policy(tenant_a, embedding_model)
    await _seed_policy(tenant_b, embedding_model)
    question = {"question": "What is the return policy?"}
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_a = (await client.post("/sessions", json={}, headers=headers_a)).json()["id"]
        session_b = (await client.post("/sessions", json={}, headers=headers_b)).json()["id"]
        # Two different accounts, well under either one's own 100/minute limit,
        # exhaust the shared global quota of 2 between them.
        first = await client.post(
            f"/sessions/{session_a}/answers", json=question, headers=headers_a
        )
        second = await client.post(
            f"/sessions/{session_b}/answers", json=question, headers=headers_b
        )
        # Account B again, not a third account: its own per-user limit is
        # nowhere close to tripped, yet this request is still refused because
        # the two prior requests (one from each account) already exhausted
        # the quota of 2 that's shared across every account.
        third = await client.post(
            f"/sessions/{session_b}/answers", json=question, headers=headers_b
        )

    assert [first.status_code, second.status_code, third.status_code] == [200, 200, 429]


async def test_an_account_already_over_its_own_limit_never_touches_the_global_counter(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    # The per-user check must run -- and reject -- before the global one ever
    # executes. Checked the other way around, every request from an account
    # already over its own limit would still increment the shared global
    # counter on its way to being rejected, letting one abusive account burn
    # down the global quota and deny service to every other tenant.
    os.environ["CHAT_RATE_LIMIT_PER_MINUTE"] = "1"
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    await _seed_policy(tenant_id, embedding_model)
    question = {"question": "What is the return policy?"}
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        first = await client.post(
            f"/sessions/{session_id}/answers", json=question, headers=headers
        )
        # Both refused by the per-user check alone (already at its limit of
        # 1) -- neither should ever reach the global check.
        second = await client.post(
            f"/sessions/{session_id}/answers", json=question, headers=headers
        )
        third = await client.post(
            f"/sessions/{session_id}/answers", json=question, headers=headers
        )

    assert [first.status_code, second.status_code, third.status_code] == [200, 429, 429]
    redis_client = redis.from_url(redis_url, decode_responses=True)
    global_count = await redis_client.get(_GLOBAL_CHAT_REDIS_KEY)
    await redis_client.aclose()
    # Only the one request that passed its own per-user check incremented
    # the shared global counter; the two the per-user limit rejected did not.
    assert global_count == "1"


async def test_creating_sessions_is_limited_to_20_a_minute_per_user(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        statuses = [
            (await client.post("/sessions", json={}, headers=headers)).status_code
            for _ in range(21)
        ]
    assert statuses == [201] * 20 + [429]


async def test_invalid_requests_are_refused_with_422(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    answers_path = "/sessions/{}/answers"
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        responses = [
            await client.post(
                answers_path.format(session_id), json={"question": "   "}, headers=headers
            ),
            await client.post(
                answers_path.format(session_id), json={"question": "x" * 4001}, headers=headers
            ),
            await client.post("/sessions", json={"title": "x" * 201}, headers=headers),
            await client.get("/sessions?limit=0", headers=headers),
            await client.get("/sessions?limit=101", headers=headers),
            await client.post(
                answers_path.format("not-a-uuid"), json={"question": "q"}, headers=headers
            ),
        ]
    assert [r.status_code for r in responses] == [422] * 6


async def test_an_oversized_request_body_is_rejected_before_validation_runs(
    app_database_url, redis_url, qdrant_url, embedding_model
):
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        response = await client.post(
            "/sessions", content=json.dumps({"title": "x" * 20_000}).encode(), headers={
                "Content-Type": "application/json", "Authorization": "Bearer not-even-checked",
            }
        )
    assert response.status_code == 413
