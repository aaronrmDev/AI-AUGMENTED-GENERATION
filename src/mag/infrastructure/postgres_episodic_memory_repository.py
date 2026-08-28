import json
import uuid

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from src.mag.domain.entities import EpisodicMemory
from src.mag.domain.ports import EpisodicMemoryRepository


class PostgresEpisodicMemoryRepository(EpisodicMemoryRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None:
        await self._session.execute(
            text(
                """
                INSERT INTO episodic_memory
                    (id, session_id, tenant_id, content, embedding, timestamp, salience_score)
                VALUES
                    (:id, :session_id, :tenant_id, CAST(:content AS jsonb), :embedding,
                     :timestamp, :salience_score)
                """
            ),
            {
                "id": episode.id,
                "session_id": episode.session_id,
                "tenant_id": tenant_id,
                "content": json.dumps(episode.content),
                "embedding": str(episode.embedding),
                "timestamp": episode.timestamp,
                "salience_score": episode.salience_score,
            },
        )
        await self._session.flush()

    async def get_by_session(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> list[EpisodicMemory]:
        result = await self._session.execute(
            text(
                "SELECT id, session_id, content, timestamp, salience_score "
                "FROM episodic_memory "
                "WHERE session_id = :session_id AND tenant_id = :tenant_id "
                "ORDER BY timestamp ASC"
            ),
            {"session_id": session_id, "tenant_id": tenant_id},
        )
        return [self._row_to_episode(row) for row in result.mappings()]

    async def search_by_similarity(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[EpisodicMemory]:
        result = await self._session.execute(
            text(
                """
                SELECT id, session_id, content, timestamp, salience_score
                FROM episodic_memory
                WHERE tenant_id = :tenant_id
                ORDER BY embedding <=> CAST(:query_embedding AS vector)
                LIMIT :top_k
                """
            ),
            {
                "tenant_id": tenant_id,
                "query_embedding": str(query_embedding),
                "top_k": top_k,
            },
        )
        return [self._row_to_episode(row) for row in result.mappings()]

    @staticmethod
    def _row_to_episode(row: RowMapping) -> EpisodicMemory:
        content = row["content"]
        if isinstance(content, str):
            content = json.loads(content)
        return EpisodicMemory(
            id=row["id"],
            session_id=row["session_id"],
            content=content,
            # Never read back from Postgres -- Qdrant
            # (qdrant_episodic_memory_index.py) is this system's actual
            # nearest-neighbor search path, and no reader in this codebase
            # parses a pgvector column's text output back into floats
            # (PostgresDocumentRepository's chunk reads follow the same
            # embedding=[] convention for the same reason).
            embedding=[],
            timestamp=row["timestamp"],
            salience_score=row["salience_score"],
        )
