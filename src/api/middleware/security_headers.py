"""Adds the security response-header baseline this API is missing (API8, A05):
HSTS, nosniff, a frame-options denial, and a conservative referrer policy.

HSTS is sent unconditionally. A browser only acts on it over an actual HTTPS
connection -- sending it over the plain HTTP this project's own tests and local
development use is inert, not wrong, and TLS termination at a future ingress is
what makes it take effect (docs/security/SECURITY.md already treats TLS as an
infrastructure-layer property this application doesn't itself provide).

Only these four: the finding this fixes names exactly these, and adding a header
nobody asked for is exactly the kind of scope creep a review would flag.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_HEADERS = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.update(_HEADERS)
        return response
