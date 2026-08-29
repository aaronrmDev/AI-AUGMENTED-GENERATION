import json
import uuid
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryRepository

_SELECT_COLUMNS = "id, session_id, content, timestamp, salience_score, consolidated_at"


def _escape_like(value: str) -> str:
    # Postgres LIKE/ILIKE's default escape character is backslash -- without
    # this, an entity string containing % or _ is interpreted as a wildcard
    # instead of a literal character (entity="v1.2_beta" would ILIKE-match
    # "v1.2Xbeta" for any X, since _ matches any single character). Escape
    # the escape character itself first so a literal backslash in `value`
    # doesn't get reinterpreted as the start of a new escape sequence.
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


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
                f"SELECT {_SELECT_COLUMNS} "
                "FROM episodic_memory "
                "WHERE session_id = :session_id AND tenant_id = :tenant_id "
                "ORDER BY timestamp ASC"
            ),
            {"session_id": session_id, "tenant_id": tenant_id},
        )
        return [self._row_to_episode(row) for row in result.mappings()]

    async def get_unconsolidated_by_session(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, limit: int
    ) -> list[EpisodicMemory]:
        result = await self._session.execute(
            text(
                f"SELECT {_SELECT_COLUMNS} "
                "FROM episodic_memory "
                "WHERE session_id = :session_id AND tenant_id = :tenant_id "
                "AND consolidated_at IS NULL "
                "ORDER BY timestamp ASC "
                "LIMIT :limit"
            ),
            {"session_id": session_id, "tenant_id": tenant_id, "limit": limit},
        )
        return [self._row_to_episode(row) for row in result.mappings()]

    async def mark_consolidated(
        self, episode_ids: list[uuid.UUID], tenant_id: uuid.UUID
    ) -> None:
        if not episode_ids:
            return
        await self._session.execute(
            text(
                "UPDATE episodic_memory SET consolidated_at = now() "
                "WHERE id = ANY(:episode_ids) AND tenant_id = :tenant_id"
            ),
            {"episode_ids": episode_ids, "tenant_id": tenant_id},
        )
        await self._session.flush()

    async def search_by_similarity(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredEpisode]:
        result = await self._session.execute(
            text(
                f"""
                SELECT {_SELECT_COLUMNS},
                    1 - (embedding <=> CAST(:query_embedding AS vector)) AS score
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
        return [
            ScoredEpisode(episode=self._row_to_episode(row), score=float(row["score"]))
            for row in result.mappings()
        ]

    async def get_by_session_in_window(
        self,
        session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        start: datetime,
        end: datetime,
        top_k: int,
    ) -> list[EpisodicMemory]:
        result = await self._session.execute(
            text(
                f"SELECT {_SELECT_COLUMNS} "
                "FROM episodic_memory "
                "WHERE session_id = :session_id AND tenant_id = :tenant_id "
                "AND timestamp BETWEEN :start AND :end "
                "ORDER BY timestamp DESC "
                "LIMIT :top_k"
            ),
            {
                "session_id": session_id,
                "tenant_id": tenant_id,
                "start": start,
                "end": end,
                "top_k": top_k,
            },
        )
        return [self._row_to_episode(row) for row in result.mappings()]

    async def get_recent_by_session(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, limit: int
    ) -> list[EpisodicMemory]:
        result = await self._session.execute(
            text(
                f"SELECT {_SELECT_COLUMNS} "
                "FROM episodic_memory "
                "WHERE session_id = :session_id AND tenant_id = :tenant_id "
                "ORDER BY timestamp DESC "
                "LIMIT :limit"
            ),
            {"session_id": session_id, "tenant_id": tenant_id, "limit": limit},
        )
        return [self._row_to_episode(row) for row in result.mappings()]

    async def get_by_session_ranked_by_salience(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[EpisodicMemory]:
        result = await self._session.execute(
            text(
                f"SELECT {_SELECT_COLUMNS} "
                "FROM episodic_memory "
                "WHERE session_id = :session_id AND tenant_id = :tenant_id "
                "ORDER BY salience_score DESC "
                "LIMIT :top_k"
            ),
            {"session_id": session_id, "tenant_id": tenant_id, "top_k": top_k},
        )
        return [self._row_to_episode(row) for row in result.mappings()]

    async def get_by_session_matching_entity(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, entity: str, top_k: int
    ) -> list[EpisodicMemory]:
        result = await self._session.execute(
            text(
                f"SELECT {_SELECT_COLUMNS} "
                "FROM episodic_memory "
                "WHERE session_id = :session_id AND tenant_id = :tenant_id "
                "AND ("
                "  content->'entities' @> to_jsonb(ARRAY[:entity]::text[])"
                "  OR content::text ILIKE :pattern ESCAPE '\\'"
                ") "
                "ORDER BY timestamp DESC "
                "LIMIT :top_k"
            ),
            {
                "session_id": session_id,
                "tenant_id": tenant_id,
                "entity": entity,
                "pattern": f"%{_escape_like(entity)}%",
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
            # Never read back from Postgres: search_by_similarity above
            # already performs a real, correctly-ordered nearest-neighbor
            # search via pgvector's <=> operator (this IS a real search
            # path today, not a stub -- see
            # test_search_by_similarity_orders_by_nearest_neighbor), it just
            # doesn't parse the raw vector back out of its text
            # representation. Qdrant (qdrant_episodic_memory_index.py) is
            # this system's embedding-BEARING read path -- the one to use
            # when the caller needs the actual vector values back, not
            # merely a correctly-ranked result set. PostgresDocumentRepository
            # 's chunk reads follow the same embedding=[] convention for the
            # same reason.
            embedding=[],
            timestamp=row["timestamp"],
            salience_score=row["salience_score"],
            consolidated_at=row["consolidated_at"],
        )
