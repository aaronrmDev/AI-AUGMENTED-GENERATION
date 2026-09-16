from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.api.middleware.security_headers import SecurityHeadersMiddleware


async def _ok(request):
    return PlainTextResponse("ok")


def test_every_response_carries_the_four_security_headers():
    app = Starlette(routes=[Route("/ok", _ok)])
    app.add_middleware(SecurityHeadersMiddleware)
    client = TestClient(app)

    response = client.get("/ok")

    assert response.headers["Strict-Transport-Security"] == "max-age=63072000; includeSubDomains"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
