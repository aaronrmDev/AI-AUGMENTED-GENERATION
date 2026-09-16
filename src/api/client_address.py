"""Resolves the real client address behind a configured number of trusted reverse
proxies, so the auth rate limiter's per-IP key isn't the proxy's own address for
every request.

X-Forwarded-For is a client-controlled header by default: anyone can send one. It
only becomes trustworthy for the hops an operator actually put in front of this
app -- each such hop appends the address it received the request from, so exactly
the rightmost TRUSTED_PROXY_COUNT entries were written by infrastructure the
operator controls, and everything to their left (including a first entry someone
might expect to be "the real client") could have been forged by the client itself
before it ever reached the first trusted hop.

TRUSTED_PROXY_COUNT defaults to 0 -- fail closed: no reverse proxy fronts this app
in any environment it runs in today, so the header is never read at all, and
behavior is identical to request.client.host, exactly as before this module
existed.
"""

import os

from fastapi import Request


def trusted_proxy_count() -> int:
    return int(os.environ.get("TRUSTED_PROXY_COUNT", "0"))


def real_client_ip(request: Request) -> str:
    count = trusted_proxy_count()
    if count > 0:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded is not None:
            hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
            if len(hops) >= count:
                return hops[-count]
    return request.client.host if request.client else "unknown"
