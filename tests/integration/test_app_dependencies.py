# tests/integration/test_app_dependencies.py
import os


async def test_health_endpoint_returns_ok(app_database_url, redis_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from httpx import ASGITransport, AsyncClient

    from src.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_missing_authorization_header_returns_domain_token_expired(
    app_database_url, redis_url
):
    # No auth-gated route exists on `app` yet -- Task 13 adds the router. This
    # throwaway probe app+route exercises get_current_user_claims in isolation
    # to prove a fully-absent Authorization header surfaces as the domain's
    # TokenExpired via the registered handler, not FastAPI's default 422
    # RequestValidationError shape (which is what Header(...)'s required-field
    # marker would produce, bypassing exception_handlers.py entirely).
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from fastapi import Depends, FastAPI
    from httpx import ASGITransport, AsyncClient

    from src.api.dependencies import get_current_user_claims
    from src.api.exception_handlers import register_exception_handlers

    probe_app = FastAPI()
    register_exception_handlers(probe_app)

    @probe_app.get("/protected")
    async def protected(claims: dict = Depends(get_current_user_claims)) -> dict:
        return {"claims": claims}

    transport = ASGITransport(app=probe_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/protected")

    assert response.status_code == 401
    assert response.json() == {"detail": "Token expired"}
