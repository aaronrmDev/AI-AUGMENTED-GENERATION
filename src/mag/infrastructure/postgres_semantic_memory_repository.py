import uuid

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from src.mag.domain.entities import ScoredFact, SemanticMemory
from src.mag.domain.ports import SemanticMemoryRepository


class PostgresSemanticMemoryRepository(SemanticMemoryRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None:
        # ON CONFLICT DO UPDATE, not a bare INSERT: a fact_key is a slot a
        # later call overwrites (uq_semantic_memory_user_id_fact_key from
        # migration 0003 makes (user_id, fact_key) unique), not an
        # append-only log -- an INSERT-only version here let two calls with
        # the same key produce two rows resolved nondeterministically by
        # find_by_key.
        await self._session.execute(
            text(
                """
                INSERT INTO semantic_memory (
                    id, user_id, tenant_id, fact_key, fact_value, embedding,
                    confidence, source, valid_until
                )
                VALUES (
                    :id, :user_id, :tenant_id, :fact_key, :fact_value, :embedding,
                    :confidence, :source, :valid_until
                )
                ON CONFLICT (user_id, fact_key) DO UPDATE SET
                    id = EXCLUDED.id,
                    fact_value = EXCLUDED.fact_value,
                    embedding = EXCLUDED.embedding,
                    confidence = EXCLUDED.confidence,
                    source = EXCLUDED.source,
                    valid_until = EXCLUDED.valid_until
                """
            ),
            {
                "id": fact.id,
                "user_id": fact.user_id,
                "tenant_id": tenant_id,
                "fact_key": fact.fact_key,
                "fact_value": fact.fact_value,
                "embedding": str(fact.embedding),
                "confidence": fact.confidence,
                "source": fact.source,
                "valid_until": fact.valid_until,
            },
        )
        await self._session.flush()

    async def find_by_key(
        self, user_id: uuid.UUID, fact_key: str, tenant_id: uuid.UUID
    ) -> SemanticMemory | None:
        result = await self._session.execute(
            text(
                """
                SELECT id, user_id, fact_key, fact_value, confidence, source, valid_until
                FROM semantic_memory
                WHERE user_id = :user_id AND fact_key = :fact_key AND tenant_id = :tenant_id
                """
            ),
            {"user_id": user_id, "fact_key": fact_key, "tenant_id": tenant_id},
        )
        row = result.mappings().first()
        return self._row_to_fact(row) if row else None

    async def search_by_similarity(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredFact]:
        result = await self._session.execute(
            text(
                """
                SELECT id, user_id, fact_key, fact_value, confidence, source, valid_until,
                    1 - (embedding <=> CAST(:query_embedding AS vector)) AS score
                FROM semantic_memory
                WHERE user_id = :user_id AND tenant_id = :tenant_id
                ORDER BY embedding <=> CAST(:query_embedding AS vector)
                LIMIT :top_k
                """
            ),
            {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "query_embedding": str(query_embedding),
                "top_k": top_k,
            },
        )
        return [
            ScoredFact(fact=self._row_to_fact(row), score=float(row["score"]))
            for row in result.mappings()
        ]

    @staticmethod
    def _row_to_fact(row: RowMapping) -> SemanticMemory:
        return SemanticMemory(
            id=row["id"],
            user_id=row["user_id"],
            fact_key=row["fact_key"],
            fact_value=row["fact_value"],
            # Never read back from Postgres -- Qdrant (QdrantSemanticMemoryIndex)
            # is this system's embedding-bearing read path, matching
            # PostgresEpisodicMemoryRepository's identical convention and
            # PostgresDocumentRepository's chunk reads before it.
            embedding=[],
            confidence=row["confidence"],
            source=row["source"],
            valid_until=row["valid_until"],
        )
