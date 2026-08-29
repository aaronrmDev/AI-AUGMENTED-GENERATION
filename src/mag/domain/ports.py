from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from src.mag.domain.entities import (
    EpisodicMemory,
    ProceduralMemory,
    ScoredEpisode,
    ScoredFact,
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
        """Oldest-unconsolidated-first, consolidated_at IS NULL only -- a
        backlog drain, not a recency window. In steady state (Consolidation
        run at least as often as episodes accumulate) the unconsolidated set
        IS the recent tail, so this coincides with MAG.md's "last N turns"
        framing -- but if a backlog ever exceeds one run's limit, this
        returns the OLDEST unconsolidated episodes, not the most recent
        ones, so the backlog actually drains instead of perpetually
        reprocessing whatever's newest while older episodes wait forever."""

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
    ) -> list[ScoredEpisode]:
        """Returned episodes carry embedding=[] -- Qdrant (EpisodicMemoryIndex)
        is this system's embedding-bearing read path, matching how
        PostgresDocumentRepository's chunk reads already work. A caller that
        needs the real vector back should go through the index, not this
        repository.

        score is cosine SIMILARITY (1 - pgvector's <=> cosine distance), the
        same quantity and scale Qdrant's search() below returns natively --
        deliberately kept comparable across both backends so retrieval
        strategies that fuse scores (MAG Batch C) aren't secretly mixing two
        different conventions under one name."""

    @abstractmethod
    async def get_by_session_in_window(
        self,
        session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        start: datetime,
        end: datetime,
        top_k: int,
    ) -> list[EpisodicMemory]:
        """Episodes in [start, end], newest first, LIMIT top_k -- pushed down
        to SQL like every sibling method here, not truncated in Python after
        an unbounded fetch (a Batch C review caught the original version
        doing exactly that). Membership is binary (in the window or not), so
        callers score every result the same way rather than reading a graded
        signal out of this."""

    @abstractmethod
    async def get_recent_by_session(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, limit: int
    ) -> list[EpisodicMemory]:
        """The `limit` most recent episodes in the session, newest first.
        Used by temporal retrieval when no explicit window is given -- a
        plain recency fallback, distinct from get_by_session (which returns
        everything, oldest first, for consolidation's backlog-drain use)."""

    @abstractmethod
    async def get_by_session_ranked_by_salience(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[EpisodicMemory]:
        """The top_k episodes in the session by salience_score, highest
        first. Only meaningful once something actually writes a non-default
        salience_score -- see CaptureEpisode's salience-scoring step."""

    @abstractmethod
    async def get_by_session_matching_entity(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, entity: str, top_k: int
    ) -> list[EpisodicMemory]:
        """Episodes in the session whose content mentions `entity`, newest
        first -- matched against a structured content["entities"] list when
        present, and against the whole serialized content as a substring
        fallback otherwise. Binary relevance: this system has no per-mention
        confidence signal to grade matches by, so callers treat every
        returned episode as equally relevant rather than inventing a
        precision the underlying match doesn't have."""


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
    ) -> list[ScoredFact]:
        """Returned facts carry embedding=[] -- same convention as
        EpisodicMemoryRepository.search_by_similarity above. score is cosine
        similarity, same scale rationale as that method's docstring."""


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
        self,
        query_embedding: list[float],
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        top_k: int,
    ) -> list[ScoredEpisode]:
        """Returned episodes carry their real embedding -- this is the
        embedding-bearing read path (see EpisodicMemoryRepository above).
        That embedding is L2-normalized (unit length), not bit-identical to
        whatever was upserted: the underlying Qdrant collection is
        configured for COSINE distance, which normalizes every vector on
        storage (confirmed empirically, not assumed -- see
        test_qdrant_semantic_memory_index.py's regression test for the
        sibling index). Same direction, unit length -- correct for
        similarity comparisons, just not the original values.

        Scoped to session_id as well as tenant_id -- a Batch C review caught
        an earlier version of this method (tenant_id-only) returning another
        session's episodes within the same tenant to SemanticSimilarityRetrieval,
        contradicting this batch's own "no cross-session retrieval" design.

        consolidated_at is ALWAYS None from this path, regardless of the
        real value in Postgres -- upsert() doesn't write it to the payload
        and mark_consolidated() (EpisodicMemoryRepository) only updates
        Postgres. Nothing reads this field through the index today, but a
        caller that starts filtering/weighting on it here would get a
        silently wrong answer, not an error. Use
        EpisodicMemoryRepository.get_unconsolidated_by_session for a
        consolidation-status-aware read until this index carries the field
        too."""


class SemanticMemoryIndex(ABC):
    # ensure_collection() deliberately omitted -- see EpisodicMemoryIndex
    # above.
    @abstractmethod
    async def upsert(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None: ...

    @abstractmethod
    async def search(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredFact]:
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
