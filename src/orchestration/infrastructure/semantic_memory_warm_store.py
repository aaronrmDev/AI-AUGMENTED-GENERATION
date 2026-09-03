import uuid
from datetime import UTC, datetime

from src.mag.domain.entities import SemanticMemory
from src.mag.domain.ports import SemanticMemoryIndex, SemanticMemoryRepository
from src.orchestration.domain.entities import WarmEntry
from src.orchestration.domain.ports import WarmStore
from src.orchestration.domain.sync_mixer import content_hash
from src.rag.domain.ports import EmbeddingModel


def _fact_key_for(document_id: uuid.UUID) -> str:
    return f"rag_promoted:{document_id}"


class SemanticMemoryWarmStore(WarmStore):
    """MAG's real warm tier: a promoted RAG document lives as a real
    semantic fact in Postgres + Qdrant, not a KV cache -- reusing this
    project's existing MAG infrastructure (PostgresSemanticMemoryRepository,
    QdrantSemanticMemoryIndex) rather than inventing a parallel store.

    fact_key is derived deterministically from document_id
    ("rag_promoted:<document_id>"), matching RecordSemanticFact's own
    deterministic-id-by-key precedent (MAG Batch F) so re-promoting the
    same document overwrites the existing fact rather than duplicating it.
    """

    def __init__(
        self,
        semantic_memory_repository: SemanticMemoryRepository,
        semantic_memory_index: SemanticMemoryIndex,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._repository = semantic_memory_repository
        self._index = semantic_memory_index
        self._embedder = embedding_model

    async def promote(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID, content: str
    ) -> None:
        fact_key = _fact_key_for(document_id)
        fact = SemanticMemory(
            id=uuid.uuid5(uuid.NAMESPACE_OID, f"semantic_memory:{user_id}:{fact_key}"),
            user_id=user_id,
            fact_key=fact_key,
            fact_value=content,
            embedding=self._embedder.embed(content),
        )
        await self._repository.save(fact, tenant_id)
        await self._index.upsert(fact, tenant_id)

    async def lookup(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> WarmEntry | None:
        fact = await self._repository.find_by_key(user_id, _fact_key_for(document_id), tenant_id)
        # find_by_key deliberately bypasses the invalidated/archived filter
        # search_by_similarity applies (SemanticMemoryRepository's own
        # documented contract, for callers that need to read a fact
        # regardless of status) -- this port's own "is it still warm"
        # semantics has to apply that filter itself, matching the REAL
        # filter's exact comparison (`valid_until IS NULL OR valid_until >
        # now()` in postgres_semantic_memory_repository.py's own SQL) --
        # not "valid_until is set at all," which would incorrectly treat a
        # future-dated expiry as already-invalid.
        if fact is None or fact.archived_at is not None:
            return None
        if fact.valid_until is not None and fact.valid_until <= datetime.now(UTC):
            return None
        return WarmEntry(content_hash=content_hash(fact.fact_value), content=fact.fact_value)

    async def demote(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> None:
        # SemanticMemoryRepository.invalidate's real implementation issues
        # an UPDATE ... RETURNING and reads exactly one row back -- it
        # raises if nothing matched, rather than silently no-op'ing. A
        # demote on a document that was never promoted (or already
        # demoted) must stay a safe no-op, matching FrozenCache.evict's
        # own established dict.pop(key, None) precedent, so this checks
        # first rather than letting that exception surface.
        if await self.lookup(tenant_id, user_id, document_id) is None:
            return
        await self._repository.invalidate(
            user_id, _fact_key_for(document_id), tenant_id, invalidated_at=None
        )

    async def contains(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> bool:
        return await self.lookup(tenant_id, user_id, document_id) is not None
