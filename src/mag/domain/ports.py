from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from src.mag.domain.entities import (
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemoryTurn,
)


class EpisodicMemoryRepository(ABC):
    @abstractmethod
    async def save(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None: ...

    @abstractmethod
    async def get_by_session(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> list[EpisodicMemory]: ...

    @abstractmethod
    async def get_unconsolidated_by_session(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, limit: int
    ) -> list[EpisodicMemory]:
        """Oldest-first, consolidated_at IS NULL only -- the recency-ordered
        "last N turns" batch Consolidation reflects on."""

    @abstractmethod
    async def mark_consolidated(
        self, episode_ids: list[uuid.UUID], tenant_id: uuid.UUID
    ) -> None:
        """Sets consolidated_at on every id given, regardless of whether
        Consolidation's reflection pass actually extracted a fact from it --
        a reflected-on episode with nothing durable in it is still done, not
        eligible for re-reflection on the next run."""

    @abstractmethod
    async def search_by_similarity(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[EpisodicMemory]:
        """Returned episodes carry embedding=[] -- Qdrant (EpisodicMemoryIndex)
        is this system's embedding-bearing read path, matching how
        PostgresDocumentRepository's chunk reads already work. A caller that
        needs the real vector back should go through the index, not this
        repository."""


class SemanticMemoryRepository(ABC):
    @abstractmethod
    async def save(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None:
        """Upserts by (user_id, fact_key): a fact_key is a slot to be
        overwritten as new information arrives, not accumulated as unbounded
        duplicates (see migration 0003's uq_semantic_memory_user_id_fact_key
        constraint)."""

    @abstractmethod
    async def find_by_key(
        self, user_id: uuid.UUID, fact_key: str, tenant_id: uuid.UUID
    ) -> SemanticMemory | None: ...

    @abstractmethod
    async def search_by_similarity(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[SemanticMemory]:
        """Returned facts carry embedding=[] -- same convention as
        EpisodicMemoryRepository.search_by_similarity above."""


class EpisodicMemoryIndex(ABC):
    # ensure_collection() is deliberately NOT part of this port: it's a
    # Qdrant collection-provisioning concern, not a domain operation, and
    # RAG's own VectorStore port (src/rag/domain/ports.py) already
    # establishes that convention -- QdrantVectorStore.ensure_collection is
    # concrete-only, called directly by whatever wires the app together, not
    # abstracted here. Matching that reference exactly (rather than
    # re-litigating it per-vertical) is what this port does.
    @abstractmethod
    async def upsert(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None: ...

    @abstractmethod
    async def search(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[EpisodicMemory]:
        """Returned episodes carry their real embedding -- this is the
        embedding-bearing read path (see EpisodicMemoryRepository above).
        That embedding is L2-normalized (unit length), not bit-identical to
        whatever was upserted: the underlying Qdrant collection is
        configured for COSINE distance, which normalizes every vector on
        storage (confirmed empirically, not assumed -- see
        test_qdrant_semantic_memory_index.py's regression test for the
        sibling index). Same direction, unit length -- correct for
        similarity comparisons, just not the original values."""


class SemanticMemoryIndex(ABC):
    # ensure_collection() deliberately omitted -- see EpisodicMemoryIndex
    # above.
    @abstractmethod
    async def upsert(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None: ...

    @abstractmethod
    async def search(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[SemanticMemory]:
        """Returned facts carry their real embedding -- this is the
        embedding-bearing read path (see SemanticMemoryRepository above),
        L2-normalized rather than bit-identical to the upserted value (same
        COSINE-distance-collection behavior as EpisodicMemoryIndex.search
        above)."""


class ProceduralMemoryRepository(ABC):
    @abstractmethod
    async def save(self, procedure: ProceduralMemory, tenant_id: uuid.UUID) -> None:
        """Upserts by (user_id, task_pattern) -- same reasoning as
        SemanticMemoryRepository.save above, see migration 0004's
        uq_procedural_memory_user_id_task_pattern constraint."""

    @abstractmethod
    async def find_by_task_pattern(
        self, user_id: uuid.UUID, task_pattern: str, tenant_id: uuid.UUID
    ) -> ProceduralMemory | None: ...


class WorkingMemoryStore(ABC):
    @abstractmethod
    async def push_turn(self, session_id: uuid.UUID, turn: WorkingMemoryTurn) -> None: ...

    @abstractmethod
    async def get_recent_turns(
        self, session_id: uuid.UUID, limit: int
    ) -> list[WorkingMemoryTurn]:
        """limit <= 0 returns an empty list -- never the whole session
        (Redis's LRANGE key -0 -1 idiom for "the whole list" is the trap this
        guards against; see redis_working_memory_store.py)."""
