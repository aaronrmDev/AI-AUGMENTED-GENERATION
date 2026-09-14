import json
import os
import uuid
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

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


@pytest.fixture(autouse=True)
def _default_chat_rate_limit():
    previous = os.environ.pop("CHAT_RATE_LIMIT_PER_MINUTE", None)
    yield
    os.environ.pop("CHAT_RATE_LIMIT_PER_MINUTE", None)
    if previous is not None:
        os.environ["CHAT_RATE_LIMIT_PER_MINUTE"] = previous


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
