from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.api.middleware.max_body_size import MaxBodySizeMiddleware


async def _echo(request):
    body = await request.body()
    return JSONResponse({"received": len(body)})


def _app(*, default_max_bytes: int, path_overrides: dict[str, int] | None = None) -> Starlette:
    app = Starlette(routes=[Route("/echo", _echo, methods=["POST"])])
    app.add_middleware(
        MaxBodySizeMiddleware, default_max_bytes=default_max_bytes, path_overrides=path_overrides
    )
    return app


def test_a_body_within_the_limit_reaches_the_app_unchanged():
    client = TestClient(_app(default_max_bytes=1024))
    response = client.post("/echo", content=b"x" * 100)
    assert response.status_code == 200
    assert response.json() == {"received": 100}


async def test_a_body_over_the_content_length_declared_limit_is_rejected_without_reading_it():
    # A TestClient/httpx round trip can't distinguish this from the streaming
    # drain path below -- both produce a 413. Drive the middleware directly
    # over raw ASGI instead, with a receive() spy, so the assertion actually
    # proves the Content-Length fast path fired: it rejects without ever
    # calling receive() to read (and discard) a single byte of the body.
    receive_calls = 0

    async def _receive():
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[dict] = []

    async def _send(message):
        sent.append(message)

    async def _unreachable_app(scope, receive, send):
        raise AssertionError("the wrapped app must not run for an oversized declared body")

    middleware = MaxBodySizeMiddleware(_unreachable_app, default_max_bytes=10)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/echo",
        "headers": [(b"content-length", b"1000")],
    }

    await middleware(scope, _receive, _send)

    assert receive_calls == 0
    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 413


def test_a_chunked_body_over_the_limit_with_no_content_length_is_still_rejected():
    client = TestClient(_app(default_max_bytes=10))

    def _stream():
        for _ in range(5):
            yield b"x" * 5

    response = client.post("/echo", content=_stream())
    assert response.status_code == 413


def test_a_path_override_gets_its_own_limit():
    client = TestClient(_app(default_max_bytes=10, path_overrides={"/echo": 1024}))
    response = client.post("/echo", content=b"x" * 100)
    assert response.status_code == 200
    assert response.json() == {"received": 100}


def test_a_path_override_matches_only_the_exact_path_not_a_sub_path():
    middleware = MaxBodySizeMiddleware(
        Starlette(),
        default_max_bytes=16 * 1024,
        path_overrides={"/documents": 11 * 1024 * 1024},
    )

    assert middleware._limit_for("/documents/search") == 16 * 1024
    assert middleware._limit_for("/documents") == 11 * 1024 * 1024
