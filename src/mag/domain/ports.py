from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from src.mag.domain.entities import (
    ActivatedNode,
    EpisodicMemory,
    ProceduralMemory,
    ScoredEpisode,
    ScoredFact,
    SemanticMemory,
    SemanticMemoryHistoryEntry,
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
        different conventions under one name.

        Deliberately tenant-wide, not session-scoped, unlike
        EpisodicMemoryIndex.search below: this method's only caller
        (RetrieveEpisodes.by_similarity, from an earlier batch) searches
        across a tenant's whole episode history by design. Batch C's
        SemanticSimilarityRetrieval needed session-scoped semantic search
        instead, which is exactly why it wraps EpisodicMemoryIndex.search
        (Qdrant) rather than this method -- two deliberately different
        scopes for two different callers, not an inconsistency."""

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
        similarity, same scale rationale as that method's docstring.

        Excludes a fact whose valid_until has passed or whose archived_at
        is set (MAG Batch F, #63/#64) -- "exclude it from retrieval" is
        Invalidate's own contract, and "move to cold storage" is Archive's;
        neither means anything if this method still returns them. Use
        find_by_key for a direct, keyed lookup that bypasses this filter
        (a caller updating or refining a fact needs to read it regardless
        of its current status)."""

    @abstractmethod
    async def invalidate(
        self,
        user_id: uuid.UUID,
        fact_key: str,
        tenant_id: uuid.UUID,
        invalidated_at: datetime | None,
    ) -> datetime:
        """Sets valid_until on the existing row -- a targeted status flip,
        not a full re-upsert: no embedding change, so no reason to touch
        the vector store's point.

        invalidated_at may be None, meaning "right now" -- in which case
        the value actually written is computed by the DATABASE's own
        clock (COALESCE(:invalidated_at, now()) in the Postgres
        implementation), not the calling application's, and returned to
        the caller. This avoids a clock-skew window: search_by_similarity
        later compares valid_until against this same database's own
        now(), so a timestamp this database chose for itself is
        guaranteed consistent with that later comparison in a way an
        application-clock timestamp sent over the wire is not."""

    @abstractmethod
    async def archive(
        self,
        user_id: uuid.UUID,
        fact_key: str,
        tenant_id: uuid.UUID,
        archived_at: datetime | None,
    ) -> datetime:
        """Sets archived_at on the existing row -- same reasoning as
        invalidate above, including the None-means-database-computed-now
        behavior and its return value."""

    @abstractmethod
    async def save_history_entry(
        self, entry: SemanticMemoryHistoryEntry, tenant_id: uuid.UUID
    ) -> None:
        """A plain insert -- history entries are never updated or upserted,
        each one is a permanent snapshot of a value Update or Refine
        (#62/#66) is about to overwrite."""

    @abstractmethod
    async def find_history(
        self, user_id: uuid.UUID, fact_key: str, tenant_id: uuid.UUID
    ) -> list[SemanticMemoryHistoryEntry]:
        """Every superseded value for this (user_id, fact_key), newest
        first."""


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
        above).

        Same valid_until/archived_at exclusion as
        SemanticMemoryRepository.search_by_similarity above -- this port
        has its own implementation of the filter (Qdrant, not SQL), since
        Invalidate/Archive's "exclude from retrieval" applies to whichever
        search path a caller actually uses."""

    @abstractmethod
    async def set_valid_until(
        self, fact_id: uuid.UUID, tenant_id: uuid.UUID, valid_until: datetime | None
    ) -> None:
        """Updates ONLY the valid_until payload field (and its numeric-epoch
        mirror used for range filtering) -- never archived_at, never the
        stored vector. upsert() always replaces the whole point including
        the vector, so reusing it from InvalidateMemory with the
        embedding-less entity find_by_key returns (Postgres reads never
        carry a real embedding -- an established convention since Batch A)
        would silently blank out the stored vector; that's what this
        method exists to avoid. Touching only valid_until, never
        archived_at, also matters on its own: an earlier version took both
        fields together, with the caller supplying whatever the OTHER
        field's value happened to be at read time -- under a race between
        a concurrent InvalidateMemory and ArchiveMemory call on the same
        fact, whichever wrote second could silently clobber the other's
        field back to a stale snapshot. Touching only the one field this
        call actually owns removes that race entirely, matching how
        SemanticMemoryRepository.invalidate/archive were already
        single-column UPDATEs from the start."""

    @abstractmethod
    async def set_archived_at(
        self, fact_id: uuid.UUID, tenant_id: uuid.UUID, archived_at: datetime | None
    ) -> None:
        """Updates ONLY the archived_at payload field (and its
        numeric-epoch mirror) -- never valid_until, never the stored
        vector. Same reasoning as set_valid_until above."""


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


class MemoryGraphRepository(ABC):
    # ensure_schema() (constraint/index provisioning) is deliberately NOT
    # part of this port -- same established convention as
    # EpisodicMemoryIndex/SemanticMemoryIndex's ensure_collection() above:
    # a store-provisioning concern, concrete-only, not a domain operation.
    #
    # Every node this port writes carries a tenant_id property, and every
    # method here filters on it -- Neo4j has no RLS equivalent to Postgres,
    # so tenant isolation here is an application-level filter on every
    # write AND every read, not a database-enforced guarantee (see
    # DATABASE.md's own "three separate writes to three separate systems,
    # not one atomic transaction" framing -- this store is honest about not
    # inheriting Postgres's consistency model, not pretending to).
    @abstractmethod
    async def upsert_episode_node(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None:
        """Idempotent by episode.id -- re-upserting the same episode updates
        the existing node rather than creating a duplicate, matching this
        project's established upsert-by-key discipline elsewhere."""

    @abstractmethod
    async def upsert_fact_node(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None:
        """Idempotent by fact.id, same reasoning as upsert_episode_node.
        Also syncs valid_until and archived_at onto the node (MAG Batch F)
        -- RecordSemanticFact (and therefore Update/Refine, which compose
        it) calls this with the fact's current state, so the node's
        observable status stays in sync with Postgres. Invalidate/Archive
        do NOT call this for a status-only change -- see
        set_fact_valid_until/set_fact_archived_at below for why."""

    @abstractmethod
    async def set_fact_valid_until(
        self, fact_id: uuid.UUID, tenant_id: uuid.UUID, valid_until: datetime | None
    ) -> None:
        """Updates ONLY the valid_until property on an existing Fact node
        (MATCH, not MERGE -- never creates one) -- never archived_at.
        upsert_fact_node writes BOTH status properties together from
        whatever SemanticMemory it's given; using it for a status-only
        change means passing a snapshot read at the START of the calling
        command's execute(), which can go stale if a concurrent write to
        the OTHER field lands in between, silently clobbering it back.
        Touching only the field this call actually owns removes that
        race, matching SemanticMemoryIndex.set_valid_until's identical
        reasoning for the Qdrant side of the same fix."""

    @abstractmethod
    async def set_fact_archived_at(
        self, fact_id: uuid.UUID, tenant_id: uuid.UUID, archived_at: datetime | None
    ) -> None:
        """Updates ONLY the archived_at property -- never valid_until.
        Same reasoning as set_fact_valid_until above."""

    @abstractmethod
    async def link_participated_in(
        self, user_id: uuid.UUID, session_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        """Creates (or matches, if it already exists) User/Session nodes and
        a PARTICIPATED_IN edge between them -- the only writer of User/
        Session nodes in this batch, since neither needs any property
        beyond its id to serve as an edge anchor."""

    @abstractmethod
    async def link_temporally_follows(
        self, earlier_episode_id: uuid.UUID, later_episode_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        """Both episode nodes must already exist (via upsert_episode_node)
        -- this only creates the edge between them."""

    @abstractmethod
    async def link_mentions(
        self, episode_id: uuid.UUID, entity_name: str, tenant_id: uuid.UUID
    ) -> None:
        """Upserts an Entity node by name (creating it if this is the first
        time it's been mentioned) and links MENTIONS from the episode to
        it. entity.embedding is deliberately not written here -- nothing in
        this batch computes one; a future batch that needs Entity.embedding
        populated (the index DATABASE.md documents) writes it separately."""

    @abstractmethod
    async def link_abstracts_to(
        self, episode_id: uuid.UUID, fact_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        """Both nodes must already exist -- this only creates the edge,
        the graph's representation of Consolidation turning a raw episode
        into a distilled semantic fact (DATABASE.md's own description)."""

    @abstractmethod
    async def spread_activation(
        self,
        tenant_id: uuid.UUID,
        start_entity_names: list[str],
        max_hops: int,
        decay_factor: float,
        activation_threshold: float,
    ) -> list[ActivatedNode]:
        """Starts from every Entity node whose name is in start_entity_names,
        traverses outward up to max_hops (any edge type, any direction --
        spreading activation per MAG.md has no notion of a "wrong direction"
        edge to follow), and returns every node reached with activation
        decay_factor ** hops, keeping the MAX activation for a node reached
        by more than one path (not summed -- see the design spec for why).
        Nodes at or below activation_threshold are excluded, EXCEPT the
        start entities themselves (hops == 0, always activation 1.0),
        which are never excluded regardless of activation_threshold -- a
        caller's own query anchor isn't a decayed-relevance discovery to
        filter out. max_hops must be in [1, 10] and decay_factor in
        (0.0, 1.0) -- both raise ValueError otherwise (a Cypher limitation
        requires max_hops to be a literal in the traversal bound, so it
        can't fail safely deep in a query string; decay_factor >= 1.0 would
        make activation grow instead of decay with distance, inverting
        "most activated" from nearest to farthest)."""
