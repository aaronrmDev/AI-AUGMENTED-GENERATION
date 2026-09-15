import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.domain.ports import MemoryGraphRepository, SemanticMemoryIndex
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.domain.ports import SessionFactWriter
from src.rag.domain.ports import EmbeddingModel


class RecordSemanticFactWriter(SessionFactWriter):
    """Writes a user-scoped data source into MAG through RecordSemanticFact, unmodified.

    The fact key is the source key under a "source:" prefix. RecordSemanticFact
    upserts on (user_id, fact_key), so each new version overwrites the last, and the
    prefix keeps a router source from overwriting a fact MAG learned under the same
    name. The Postgres repository flushes into a session, and this writer owns that
    session's transaction: the same unit-of-work rule the cascade's MAG search follows.
    """

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        semantic_index: SemanticMemoryIndex,
        embedder: EmbeddingModel,
        graph: MemoryGraphRepository,
        *,
        source: str = "freshness-router",
    ) -> None:
        self._sessionmaker = sessionmaker
        self._index = semantic_index
        self._embedder = embedder
        self._graph = graph
        self._source = source

    async def record(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, fact_key: str, fact_value: str
    ) -> None:
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            command = RecordSemanticFact(
                PostgresSemanticMemoryRepository(session), self._index, self._embedder, self._graph
            )
            await command.execute(
                tenant_id, user_id, f"source:{fact_key}", fact_value, source=self._source
            )
