import uuid

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from src.mag.domain.entities import SemanticMemory
from src.mag.domain.ports import SemanticMemoryRepository


class PostgresSemanticMemoryRepository(SemanticMemoryRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, fact: SemanticMemory) -> None:
        await self._session.execute(
            text(
                """
                INSERT INTO semantic_memory (
                    id, user_id, fact_key, fact_value, embedding,
                    confidence, source, valid_until
                )
                VALUES (
                    :id, :user_id, :fact_key, :fact_value, :embedding,
                    :confidence, :source, :valid_until
                )
                """
            ),
            {
                "id": fact.id,
                "user_id": fact.user_id,
                "fact_key": fact.fact_key,
                "fact_value": fact.fact_value,
                "embedding": str(fact.embedding),
                "confidence": fact.confidence,
                "source": fact.source,
                "valid_until": fact.valid_until,
            },
        )
        await self._session.flush()

    async def find_by_key(self, user_id: uuid.UUID, fact_key: str) -> SemanticMemory | None:
        # "Most recent first" is best-effort: this table has no created_at
        # column (migration 0003), and id is a random uuid_generate_v4() value
        # rather than a time-ordered one, so ORDER BY id is a deterministic
        # tie-break, not a true recency ordering. A real recency ordering is
        # deferred to whichever later batch adds that column.
        result = await self._session.execute(
            text(
                """
                SELECT id, user_id, fact_key, fact_value, embedding::text AS embedding,
                       confidence, source, valid_until
                FROM semantic_memory
                WHERE user_id = :user_id AND fact_key = :fact_key
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"user_id": user_id, "fact_key": fact_key},
        )
        row = result.mappings().first()
        return self._row_to_fact(row) if row else None

    async def search_by_similarity(
        self, query_embedding: list[float], user_id: uuid.UUID, top_k: int
    ) -> list[SemanticMemory]:
        result = await self._session.execute(
            text(
                """
                SELECT id, user_id, fact_key, fact_value, embedding::text AS embedding,
                       confidence, source, valid_until
                FROM semantic_memory
                WHERE user_id = :user_id
                ORDER BY embedding <=> CAST(:query_embedding AS vector)
                LIMIT :top_k
                """
            ),
            {"user_id": user_id, "query_embedding": str(query_embedding), "top_k": top_k},
        )
        return [self._row_to_fact(row) for row in result.mappings()]

    @staticmethod
    def _row_to_fact(row: RowMapping) -> SemanticMemory:
        return SemanticMemory(
            id=row["id"],
            user_id=row["user_id"],
            fact_key=row["fact_key"],
            fact_value=row["fact_value"],
            embedding=_parse_vector(row["embedding"]),
            confidence=row["confidence"],
            source=row["source"],
            valid_until=row["valid_until"],
        )


def _parse_vector(value: str) -> list[float]:
    return [float(x) for x in value.strip("[]").split(",")]
