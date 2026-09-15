import asyncio
import logging
import uuid
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.infrastructure.db import set_tenant_context
from src.mag.domain.entities import ScoredFact
from src.mag.domain.ports import SemanticMemoryRepository
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.domain.ports import SemanticFactSearch

logger = logging.getLogger(__name__)

RepositoryFactory = Callable[[AsyncSession], SemanticMemoryRepository]


class SessionScopedSemanticFactSearch(SemanticFactSearch):
    """MAG semantic-fact search in a session this class opens, and disposes
    of, on every call.

    Measured in this batch: when the cascade's timeout cancels a query
    mid-flight, SQLAlchemy treats the CancelledError as a disconnect and
    terminates the asyncpg connection, after which rollback() and close() on
    that session both raise InterfaceError, and only invalidate() recovers
    it. A search that borrowed the request's session would leave that broken
    session behind for every later use -- a Postgres-backed hybrid RAG
    retriever falling back after a MAG timeout, for example -- and would put
    two tiers on one AsyncSession at once in PARALLEL mode, which SQLAlchemy
    does not allow. Owning the session confines a cancellation's damage to
    the search that was cancelled.

    set_tenant_context runs inside the search's own transaction, so the
    tenant_isolation RLS policy always has a tenant to enforce.
    """

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        repository_factory: RepositoryFactory = PostgresSemanticMemoryRepository,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._repository_factory = repository_factory

    async def search(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
    ) -> list[ScoredFact]:
        session = self._sessionmaker()
        try:
            await set_tenant_context(session, tenant_id)
            facts = await self._repository_factory(session).search_by_similarity(
                query_embedding, user_id, tenant_id, top_k
            )
        except BaseException:
            # Shielded so a second cancellation can't interrupt the cleanup.
            await asyncio.shield(_discard(session))
            raise
        try:
            await session.close()
        except Exception:
            # The facts are already in hand; a cleanup failure must not turn a
            # good MAG answer into an ERROR outcome.
            logger.warning(
                "closing a MAG search session failed after a successful search; discarding it",
                exc_info=True,
            )
            await asyncio.shield(_discard(session))
        except BaseException:
            await asyncio.shield(_discard(session))
            raise
        return facts


async def _discard(session: AsyncSession) -> None:
    try:
        await session.invalidate()
    except Exception:
        logger.warning(
            "could not invalidate a MAG search session after a failed or cancelled search",
            exc_info=True,
        )
