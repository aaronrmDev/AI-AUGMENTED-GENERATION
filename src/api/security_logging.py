"""Configures structlog once, so every part of the API that logs a security event
gets the same JSON-rendered, timestamped output.

Scope, matching #193's own acceptance criteria exactly: session authorization
denials and rate-limit refusals. Not the broader login/logout/token-refresh audit
trail docs/security/SECURITY.md separately describes as still ahead -- that's a
materially larger surface with its own per-event-type test requirement.

Neither event this module's callers log ever includes a token, an Authorization
header, or request body text (a question, an answer, a title): the two call sites
in exception_handlers.py only ever pass a caller's identity, a denied resource id,
a rate-limit key, and request metadata -- fields that were never capable of
carrying that content in the first place.
"""

import structlog


def configure_security_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )


security_logger = structlog.get_logger("security")
