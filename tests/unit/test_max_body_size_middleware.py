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


def test_a_body_over_the_content_length_declared_limit_is_rejected_without_reading_it():
    client = TestClient(_app(default_max_bytes=10))
    response = client.post("/echo", content=b"x" * 1000)
    assert response.status_code == 413


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
