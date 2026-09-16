import functools
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.caller import Caller, caller_from_claims
from src.api.unified_pipeline import UnifiedPipeline, build_unified_pipeline
from src.identity.domain.errors import TokenExpired
from src.identity.infrastructure.db import get_engine, get_sessionmaker, set_tenant_context
from src.identity.infrastructure.jwt_token_issuer import JWTTokenIssuer
from src.identity.infrastructure.postgres_chat_session_repository import (
    PostgresChatSessionRepository,
)
from src.identity.infrastructure.postgres_user_repository import PostgresUserRepository
from src.identity.infrastructure.redis_rate_limiter import RedisRateLimiter
from src.identity.infrastructure.redis_refresh_token_store import RedisRefreshTokenStore
from src.orchestration.application.answer_in_session import AnswerInSession
from src.rag.domain.ports import ChatModel
from src.rag.infrastructure.claude_chat_model import ClaudeChatModel
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.ollama_chat_model import OllamaChatModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.rag.infrastructure.text_extractor import TextExtractor

_engine = get_engine(os.environ["APP_DATABASE_URL"])
_sessionmaker = get_sessionmaker(_engine)


def get_token_issuer() -> JWTTokenIssuer:
    return JWTTokenIssuer(secret_key=os.environ["JWT_SECRET_KEY"])


# One Redis client per process for each of these. redis.from_url opens a connection pool,
# and building one per request opened a pool on every rate-limited request that nothing
# ever closed.
@functools.cache
def get_refresh_token_store() -> RedisRefreshTokenStore:
    return RedisRefreshTokenStore(os.environ["REDIS_URL"])


@functools.cache
def get_rate_limiter() -> RedisRateLimiter:
    return RedisRateLimiter(os.environ["REDIS_URL"])


async def close_redis_clients() -> None:
    """Close the process's shared Redis clients, so the next call builds fresh ones.

    An asyncio Redis connection belongs to the event loop that opened it, so the app calls
    this on shutdown, on the loop that served its requests. Both clients are forgotten
    before either is closed, so a close that fails can't leave a dead client cached for
    the next call. Both are attempted, and the first failure is raised afterwards.
    """
    clients: list[RedisRateLimiter | RedisRefreshTokenStore] = []
    if get_rate_limiter.cache_info().currsize:
        clients.append(get_rate_limiter())
    if get_refresh_token_store.cache_info().currsize:
        clients.append(get_refresh_token_store())
    get_rate_limiter.cache_clear()
    get_refresh_token_store.cache_clear()

    failures: list[Exception] = []
    for client in clients:
        try:
            await client.aclose()
        except Exception as exc:
            failures.append(exc)
    if failures:
        raise failures[0]


async def get_raw_db_session() -> AsyncGenerator[AsyncSession, None]:
    """A session with no tenant context set — only for pre-auth flows like register/login."""
    async with _sessionmaker() as session:
        yield session


async def get_current_user_claims(
    authorization: str | None = Header(default=None),
    token_issuer: JWTTokenIssuer = Depends(get_token_issuer),
) -> dict[str, Any]:
    # Header(default=None), not Header(...) (a required field): the required
    # form makes FastAPI raise its own RequestValidationError and return a
    # generic 422 the instant the header is absent, before this function body
    # ever runs — bypassing every handler in exception_handlers.py entirely.
    # Accepting None and checking it explicitly keeps the missing-header case
    # on the same path as the malformed-header case below, so both surface
    # as the domain's TokenExpired through the registered handler.
    if authorization is None or not authorization.startswith("Bearer "):
        raise TokenExpired()
    token = authorization.removeprefix("Bearer ")
    return token_issuer.verify_access_token(token)


async def get_db_session(
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> AsyncGenerator[AsyncSession, None]:
    """A tenant-scoped session for any endpoint behind auth.

    The tenant context is set before the caller ever sees the session, so no
    endpoint can forget to do it. Nothing consumes this yet — this sub-project
    ships no protected routes — and the first one that does is where it gets
    exercised end-to-end against real RLS.
    """
    async with _sessionmaker() as session:
        await set_tenant_context(session, uuid.UUID(claims["tenant_id"]))
        yield session


def get_user_repository_unscoped(
    session: AsyncSession = Depends(get_raw_db_session),
) -> PostgresUserRepository:
    return PostgresUserRepository(session)


def get_user_repository_scoped(
    session: AsyncSession = Depends(get_db_session),
) -> PostgresUserRepository:
    return PostgresUserRepository(session)


_embedding_model = SentenceTransformersEmbedder()
_vector_store = QdrantVectorStore(os.environ["QDRANT_URL"])


def get_embedding_model() -> SentenceTransformersEmbedder:
    return _embedding_model


def get_vector_store() -> QdrantVectorStore:
    return _vector_store


def get_chunker() -> FixedSizeChunker:
    return FixedSizeChunker()


def get_extractor() -> TextExtractor:
    return TextExtractor()


def get_file_storage() -> LocalFileStorage:
    return LocalFileStorage()


def get_chat_model() -> ChatModel:
    # CHAT_PROVIDER exists for cost-free local development (docs/architecture/
    # OVERVIEW.md's "When you might deviate from this stack" section sanctions
    # Ollama for "early experimentation or a proof of concept" -- the
    # *official* ablation measurements that satisfy each
    # GitHub issue's DoD still go through vLLM-on-ROCm per docs/evaluation/
    # COMPARISON_METHODOLOGY.md, which this switch does not touch). Defaults
    # to "anthropic" so every existing deployment (docker-compose, the
    # original Task 15 smoke test) is unaffected by this switch's addition.
    provider = os.environ.get("CHAT_PROVIDER", "anthropic")

    if provider == "ollama":
        import ollama

        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ollama_client = ollama.AsyncClient(host=ollama_host)
        return OllamaChatModel(
            client=ollama_client, model_id=os.environ.get("CHAT_MODEL", "qwen3.5")
        )

    import anthropic

    anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return ClaudeChatModel(
        client=anthropic_client, model_id=os.environ.get("CHAT_MODEL", "claude-opus-5")
    )


async def get_caller(
    request: Request, claims: dict[str, Any] = Depends(get_current_user_claims)
) -> Caller:
    caller = caller_from_claims(claims)
    # Lets an exception handler running later in the same request (which only
    # ever receives `request` and the raised exception, never a route's own
    # resolved dependencies) recover the authenticated caller's identity for a
    # security-event log -- the same request.state handoff enforce_rate_limit
    # already uses for its own headers.
    request.state.caller = caller
    return caller


def get_chat_session_repository() -> PostgresChatSessionRepository:
    return PostgresChatSessionRepository(_sessionmaker)


@functools.cache
def get_unified_pipeline() -> UnifiedPipeline:
    # Built on first use, not at import, so importing the app doesn't embed every routing
    # exemplar or open a chat-model client.
    return build_unified_pipeline(
        sessionmaker=_sessionmaker,
        sessions=get_chat_session_repository(),
        embedding_model=_embedding_model,
        vector_store=_vector_store,
        chat_model=get_chat_model(),
    )


def get_answer_in_session() -> AnswerInSession:
    return get_unified_pipeline().answer_in_session
