import uuid
from datetime import datetime

from src.mag.domain.entities import SemanticMemory
from src.mag.domain.ports import SemanticMemoryIndex, SemanticMemoryRepository
from src.rag.domain.ports import EmbeddingModel


class RecordSemanticFact:
    def __init__(
        self,
        semantic_memory_repository: SemanticMemoryRepository,
        semantic_memory_index: SemanticMemoryIndex,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._repository = semantic_memory_repository
        self._index = semantic_memory_index
        self._embedder = embedding_model

    async def execute(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        fact_key: str,
        fact_value: str,
        confidence: float = 1.0,
        source: str = "",
        valid_until: datetime | None = None,
    ) -> SemanticMemory:
        # fact_value, not fact_key, is what a future query semantically
        # searches against -- fact_key is an identifier ("favorite_color"),
        # fact_value is the actual content ("blue").
        embedding = self._embedder.embed(fact_value)
        fact = SemanticMemory(
            # Deterministic (uuid5), not uuid4: Postgres upserts by
            # (user_id, fact_key) (migration 0003's unique constraint), but
            # Qdrant has no equivalent upsert-by-fact_key -- it only
            # overwrites a point that shares the SAME id. A fresh uuid4 on
            # every call left Qdrant with an orphaned stale point every time
            # a fact was re-recorded (the old id's point was never deleted,
            # just no longer referenced from Postgres), so a similarity
            # search could surface a fact the caller had already overwritten
            # with no way to tell which copy was current. A deterministic id
            # makes re-recording the same (user_id, fact_key) overwrite the
            # same Qdrant point, matching what Postgres already does.
            id=uuid.uuid5(uuid.NAMESPACE_OID, f"semantic_memory:{user_id}:{fact_key}"),
            user_id=user_id,
            fact_key=fact_key,
            fact_value=fact_value,
            embedding=embedding,
            confidence=confidence,
            source=source,
            valid_until=valid_until,
        )
        await self._repository.save(fact, tenant_id)
        await self._index.upsert(fact, tenant_id)
        return fact
