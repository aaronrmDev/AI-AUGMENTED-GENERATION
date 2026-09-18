import json
import os
import uuid
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from src.identity.infrastructure.jwt_token_issuer import JWTTokenIssuer
from src.rag.domain.entities import Chunk
from tests.integration.orchestration_env import VALID_HASH, ContextEchoChatModel

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


async def _seed_policy(tenant_id: uuid.UUID, embedding_model) -> None:
    from src.api.dependencies import get_vector_store

    chunk = Chunk(uuid.uuid4(), uuid.uuid4(), _POLICY, embedding_model.embed(_POLICY))
    await get_vector_store().upsert(chunk, tenant_id)


def _parse_sse(raw: str) -> list[tuple[str | None, dict]]:
    events: list[tuple[str | None, dict]] = []
    for block in raw.strip("\n").split("\n\n"):
        if not block:
            continue
        name: str | None = None
        data: dict = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        events.append((name, data))
    return events


async def _post_stream(client, url, *, json_body, headers):
    async with client.stream("POST", url, json=json_body, headers=headers) as response:
        await response.aread()
    return response


async def test_a_streamed_answer_reassembles_the_same_stages_and_text_the_json_endpoint_would_give(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    user_id = await _user(db_session, tenant_id)
    headers = _auth(user_id, tenant_id)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        # After _client(), never before it: _client() is what sets
        # APP_DATABASE_URL/QDRANT_URL and imports src.api.dependencies, whose
        # module-level engine construction reads those env vars at import
        # time. _seed_policy() imports that same module -- calling it first
        # (as this test would were it the very first test to run in the
        # process) hits a bare KeyError on APP_DATABASE_URL.
        await _seed_policy(tenant_id, embedding_model)
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        response = await _post_stream(
            client,
            f"/sessions/{session_id}/answers/stream",
            json_body={"question": "What is the return policy for unopened items?"},
            headers=headers,
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "routing"
    assert names[1] == "retrieval"
    assert names[2] == "budget"
    assert names[-1] == "done"
    assert names[3:-1] and all(name == "chunk" for name in names[3:-1])
    full_answer = "".join(data["text"] for name, data in events if name == "chunk")
    assert "forty-five days" in full_answer
    budget_sources = next(data for name, data in events if name == "budget")["sources"]
    assert any(s["paradigm"] == "rag" and "forty-five" in s["content"] for s in budget_sources)


async def test_streaming_another_users_session_in_the_same_tenant_returns_a_plain_404_with_no_stream(  # noqa: E501
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    tenant_id = uuid.uuid4()
    owner, intruder = await _user(db_session, tenant_id), await _user(db_session, tenant_id)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        # See the comment in the happy-path test above: this must run after
        # _client(), not before it.
        await _seed_policy(tenant_id, embedding_model)
        session_id = (
            await client.post("/sessions", json={}, headers=_auth(owner, tenant_id))
        ).json()["id"]
        response = await _post_stream(
            client,
            f"/sessions/{session_id}/answers/stream",
            json_body={"question": "What is the return policy?"},
            headers=_auth(intruder, tenant_id),
        )

    assert (response.status_code, json.loads(response.text)) == (404, _NOT_FOUND)
    assert "text/event-stream" not in response.headers.get("content-type", "")


async def test_streaming_another_tenants_session_returns_a_plain_404_with_no_stream(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    owner_tenant, intruder_tenant = uuid.uuid4(), uuid.uuid4()
    owner = await _user(db_session, owner_tenant)
    intruder = await _user(db_session, intruder_tenant)
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        # See the comment in the happy-path test above: this must run after
        # _client(), not before it.
        await _seed_policy(owner_tenant, embedding_model)
        session_id = (
            await client.post("/sessions", json={}, headers=_auth(owner, owner_tenant))
        ).json()["id"]
        response = await _post_stream(
            client,
            f"/sessions/{session_id}/answers/stream",
            json_body={"question": "What is the return policy?"},
            headers=_auth(intruder, intruder_tenant),
        )

    assert (response.status_code, json.loads(response.text)) == (404, _NOT_FOUND)
    assert "text/event-stream" not in response.headers.get("content-type", "")


async def test_a_rate_limited_caller_gets_429_not_a_stream(
    db_session, app_database_url, redis_url, qdrant_url, embedding_model
):
    os.environ["CHAT_RATE_LIMIT_PER_MINUTE"] = "2"
    tenant_id = uuid.uuid4()
    headers = _auth(await _user(db_session, tenant_id), tenant_id)
    question = {"question": "What is the return policy?"}
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        # See the comment in the happy-path test above: this must run after
        # _client(), not before it.
        await _seed_policy(tenant_id, embedding_model)
        session_id = (await client.post("/sessions", json={}, headers=headers)).json()["id"]
        responses = [
            await _post_stream(
                client,
                f"/sessions/{session_id}/answers/stream",
                json_body=question,
                headers=headers,
            )
            for _ in range(3)
        ]

    assert [r.status_code for r in responses] == [200, 200, 429]
    assert "text/event-stream" not in responses[2].headers.get("content-type", "")


async def test_every_stream_route_refuses_a_request_without_a_token(
    app_database_url, redis_url, qdrant_url, embedding_model
):
    async with await _client(app_database_url, redis_url, qdrant_url, embedding_model) as client:
        response = await _post_stream(
            client,
            f"/sessions/{uuid.uuid4()}/answers/stream",
            json_body={"question": "q"},
            headers={},
        )

    assert response.status_code == 401
    assert "text/event-stream" not in response.headers.get("content-type", "")
