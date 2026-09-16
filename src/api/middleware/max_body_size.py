"""Rejects an oversized request body before it ever reaches the app.

Every route today reads and fully parses its body before any size check runs --
Pydantic's own max_length constraints only fire once the whole body is already in
memory (CWE-770). This middleware enforces a byte ceiling itself: on the fast path
(a Content-Length header that already exceeds the limit) it rejects the request
without reading anything at all; on the slow path (no usable Content-Length) it
drains the body itself, bailing the instant the running total crosses the limit, so
the wrapped app never receives an oversized body either way.

A raw ASGI middleware, not starlette.middleware.base.BaseHTTPMiddleware: that base
class has known body-consumption quirks when a middleware needs to inspect or
replace the body, and this needs precise control over the receive() callable.
"""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class MaxBodySizeMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        default_max_bytes: int,
        path_overrides: dict[str, int] | None = None,
    ) -> None:
        self.app = app
        self.default_max_bytes = default_max_bytes
        self.path_overrides = dict(path_overrides or {})

    def _limit_for(self, path: str) -> int:
        """Returns the byte ceiling for this exact request path, or the default.

        Matched by exact string equality, never by prefix: a `path_overrides` entry
        for "/documents" applies only to that literal path (the upload route) and
        never to a distinct route that merely starts with the same string, such as
        "/documents/search", which has its own, much smaller, legitimate body size
        and would otherwise inherit an upload-sized ceiling it never asked for.
        """
        return self.path_overrides.get(path, self.default_max_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self._limit_for(scope["path"])

        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
                if declared is not None and declared > limit:
                    await _reject(scope, receive, send)
                    return
                break

        chunks: list[bytes] = []
        total = 0
        more_body = True
        while more_body:
            message: Message = await receive()
            if message["type"] != "http.request":
                # e.g. a disconnect mid-body: stop collecting and let the real
                # app see it via replay_receive's fallback below.
                break
            body = message.get("body") or b""
            total += len(body)
            if total > limit:
                await _reject(scope, receive, send)
                return
            chunks.append(body)
            more_body = message.get("more_body", False)

        buffered_body = b"".join(chunks)
        sent_once = False

        async def replay_receive() -> Message:
            nonlocal sent_once
            if not sent_once:
                sent_once = True
                return {"type": "http.request", "body": buffered_body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
    response = JSONResponse(status_code=413, content={"detail": "Request body too large"})
    await response(scope, receive, send)
