import dataclasses
import math
import uuid
from datetime import UTC, datetime

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
from src.mag.domain.ports import (
    EpisodicMemoryRepository,
    MemoryGraphRepository,
    ProceduralMemoryRepository,
    SemanticMemoryIndex,
    SemanticMemoryRepository,
    WorkingMemoryStore,
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    # Same quantity the real Postgres (1 - pgvector <=>) and Qdrant
    # (COSINE-distance collection score) backends compute -- kept real here,
    # not a stand-in constant, so a unit test asserting on result ORDER
    # (not just membership) is asserting something a real backend would
    # actually produce, matching this batch's score-carrying design.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FakeEpisodicMemoryRepository(EpisodicMemoryRepository):
    def __init__(self) -> None:
        self.saved: list[tuple[EpisodicMemory, uuid.UUID]] = []
        self._by_session: dict[uuid.UUID, list[EpisodicMemory]] = {}
        self._search_results: list[EpisodicMemory] = []

    async def save(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None:
        self.saved.append((episode, tenant_id))
        self._by_session.setdefault(episode.session_id, []).append(episode)

    async def get_by_session(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> list[EpisodicMemory]:
        return self._by_session.get(session_id, [])

    async def get_unconsolidated_by_session(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, limit: int
    ) -> list[EpisodicMemory]:
        episodes = self._by_session.get(session_id, [])
        return [e for e in episodes if e.consolidated_at is None][:limit]

    async def mark_consolidated(
        self, episode_ids: list[uuid.UUID], tenant_id: uuid.UUID
    ) -> None:
        # EpisodicMemory is frozen -- replace in place rather than mutate,
        # matching how the real UPDATE statement changes the stored row
        # without changing its identity.
        now = datetime.now(UTC)
        for session_id, episodes in self._by_session.items():
            self._by_session[session_id] = [
                dataclasses.replace(e, consolidated_at=now) if e.id in episode_ids else e
                for e in episodes
            ]

    def set_search_results(self, results: list[EpisodicMemory]) -> None:
        self._search_results = results

    async def search_by_similarity(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredEpisode]:
        scored = [
            ScoredEpisode(episode=e, score=_cosine_similarity(query_embedding, e.embedding))
            for e in self._search_results
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    async def get_by_session_in_window(
        self,
        session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        start: datetime,
        end: datetime,
        top_k: int,
    ) -> list[EpisodicMemory]:
        episodes = self._by_session.get(session_id, [])
        matches = [e for e in episodes if start <= e.timestamp <= end]
        return sorted(matches, key=lambda e: e.timestamp, reverse=True)[:top_k]

    async def get_recent_by_session(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, limit: int
    ) -> list[EpisodicMemory]:
        episodes = self._by_session.get(session_id, [])
        return sorted(episodes, key=lambda e: e.timestamp, reverse=True)[:limit]

    async def get_by_session_ranked_by_salience(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[EpisodicMemory]:
        episodes = self._by_session.get(session_id, [])
        return sorted(episodes, key=lambda e: e.salience_score, reverse=True)[:top_k]

    async def get_by_session_matching_entity(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, entity: str, top_k: int
    ) -> list[EpisodicMemory]:
        episodes = self._by_session.get(session_id, [])

        def _matches(e: EpisodicMemory) -> bool:
            # Case-sensitive here, matching real Postgres's JSONB containment
            # (content->'entities' @> to_jsonb(ARRAY[:entity]::text[])),
            # which is exact-match, not case-folded -- only the ILIKE
            # substring fallback below is genuinely case-insensitive in the
            # real backend. A Batch C review caught an earlier version of
            # this fake lowercasing BOTH paths, which could make a
            # structured-match unit test pass here while the same case
            # mismatch would fail against real Postgres.
            structured = [str(x) for x in e.content.get("entities", [])]
            if entity in structured:
                return True
            return entity.lower() in str(e.content).lower()

        matches = [e for e in episodes if _matches(e)]
        return sorted(matches, key=lambda e: e.timestamp, reverse=True)[:top_k]


class FakeSemanticMemoryRepository(SemanticMemoryRepository):
    def __init__(self) -> None:
        self.saved: list[tuple[SemanticMemory, uuid.UUID]] = []
        # Keyed by (user_id, fact_key) ONLY, matching the real repository's
        # actual unique constraint (migration 0003's
        # uq_semantic_memory_user_id_fact_key -- deliberately NOT scoped to
        # tenant_id, since user_id already implies exactly one tenant via
        # the users table; a save() under a tenant_id that doesn't match the
        # user's real tenant is a caller bug the real database's RLS policy
        # rejects outright rather than silently permitting a second row).
        # This fake previously keyed on (tenant_id, user_id, fact_key),
        # which permitted exactly the duplicate-under-a-mismatched-tenant
        # case the real constraint forbids -- a unit test built on it could
        # never catch that mismatch.
        self._by_key: dict[tuple[uuid.UUID, str], SemanticMemory] = {}
        self._search_results: list[SemanticMemory] = []
        self._history: dict[tuple[uuid.UUID, str], list[SemanticMemoryHistoryEntry]] = {}

    async def save(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None:
        self.saved.append((fact, tenant_id))
        self._by_key[(fact.user_id, fact.fact_key)] = fact

    async def find_by_key(
        self, user_id: uuid.UUID, fact_key: str, tenant_id: uuid.UUID
    ) -> SemanticMemory | None:
        # Deliberately unfiltered by valid_until/archived_at -- a direct,
        # keyed lookup, matching the real repository's identical choice
        # (MAG Batch F): a caller updating or refining a fact needs to
        # read it regardless of its current status.
        return self._by_key.get((user_id, fact_key))

    def set_search_results(self, results: list[SemanticMemory]) -> None:
        self._search_results = results

    async def search_by_similarity(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredFact]:
        now = datetime.now(UTC)
        eligible = [
            f
            for f in self._search_results
            if (f.valid_until is None or f.valid_until > now) and f.archived_at is None
        ]
        scored = [
            ScoredFact(fact=f, score=_cosine_similarity(query_embedding, f.embedding))
            for f in eligible
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    async def invalidate(
        self, user_id: uuid.UUID, fact_key: str, tenant_id: uuid.UUID, invalidated_at: datetime
    ) -> None:
        existing = self._by_key.get((user_id, fact_key))
        if existing is not None:
            self._by_key[(user_id, fact_key)] = dataclasses.replace(
                existing, valid_until=invalidated_at
            )

    async def archive(
        self, user_id: uuid.UUID, fact_key: str, tenant_id: uuid.UUID, archived_at: datetime
    ) -> None:
        existing = self._by_key.get((user_id, fact_key))
        if existing is not None:
            self._by_key[(user_id, fact_key)] = dataclasses.replace(
                existing, archived_at=archived_at
            )

    async def save_history_entry(
        self, entry: SemanticMemoryHistoryEntry, tenant_id: uuid.UUID
    ) -> None:
        self._history.setdefault((entry.user_id, entry.fact_key), []).append(entry)

    async def find_history(
        self, user_id: uuid.UUID, fact_key: str, tenant_id: uuid.UUID
    ) -> list[SemanticMemoryHistoryEntry]:
        entries = self._history.get((user_id, fact_key), [])
        return sorted(entries, key=lambda e: e.superseded_at, reverse=True)


class FakeSemanticMemoryIndex(SemanticMemoryIndex):
    def __init__(self) -> None:
        self.upserted: list[tuple[SemanticMemory, uuid.UUID]] = []
        self.status_updates: list[
            tuple[uuid.UUID, uuid.UUID, datetime | None, datetime | None]
        ] = []

    async def ensure_collection(self) -> None:
        pass

    async def upsert(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None:
        self.upserted.append((fact, tenant_id))

    async def update_status(
        self,
        fact_id: uuid.UUID,
        tenant_id: uuid.UUID,
        valid_until: datetime | None,
        archived_at: datetime | None,
    ) -> None:
        self.status_updates.append((fact_id, tenant_id, valid_until, archived_at))

    async def search(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredFact]:
        return []


class FakeProceduralMemoryRepository(ProceduralMemoryRepository):
    def __init__(self) -> None:
        self.saved: list[tuple[ProceduralMemory, uuid.UUID]] = []
        # Keyed by (user_id, task_pattern) ONLY, matching the real
        # repository's actual unique constraint (migration 0004's
        # uq_procedural_memory_user_id_task_pattern) -- same reasoning as
        # FakeSemanticMemoryRepository's identical key shape above.
        self._by_pattern: dict[tuple[uuid.UUID, str], ProceduralMemory] = {}

    async def save(self, procedure: ProceduralMemory, tenant_id: uuid.UUID) -> None:
        self.saved.append((procedure, tenant_id))
        self._by_pattern[(procedure.user_id, procedure.task_pattern)] = procedure

    async def find_by_task_pattern(
        self, user_id: uuid.UUID, task_pattern: str, tenant_id: uuid.UUID
    ) -> ProceduralMemory | None:
        return self._by_pattern.get((user_id, task_pattern))


class FakeMemoryGraphRepository(MemoryGraphRepository):
    # Records every call rather than modeling real graph state -- unlike
    # the other fakes above (which back a real query path with realistic
    # in-memory behavior), nothing in this batch's unit tests needs to
    # traverse a fake graph; they need to assert a command made the right
    # graph-write calls (CaptureEpisode/RecordSemanticFact/
    # ConsolidateEpisodes's unit tests) or to control
    # SpreadingActivationRetrieval's return value directly. Real traversal
    # behavior is covered by test_neo4j_memory_graph_repository.py against
    # real Neo4j, not re-modeled here.
    def __init__(self) -> None:
        self.upserted_episodes: list[tuple[EpisodicMemory, uuid.UUID]] = []
        self.upserted_facts: list[tuple[SemanticMemory, uuid.UUID]] = []
        self.participated_in_links: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = []
        self.temporally_follows_links: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = []
        self.mentions_links: list[tuple[uuid.UUID, str, uuid.UUID]] = []
        self.abstracts_to_links: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = []
        self._spread_activation_results: list[ActivatedNode] = []

    async def upsert_episode_node(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None:
        self.upserted_episodes.append((episode, tenant_id))

    async def upsert_fact_node(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None:
        self.upserted_facts.append((fact, tenant_id))

    async def link_participated_in(
        self, user_id: uuid.UUID, session_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        self.participated_in_links.append((user_id, session_id, tenant_id))

    async def link_temporally_follows(
        self, earlier_episode_id: uuid.UUID, later_episode_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        self.temporally_follows_links.append((earlier_episode_id, later_episode_id, tenant_id))

    async def link_mentions(
        self, episode_id: uuid.UUID, entity_name: str, tenant_id: uuid.UUID
    ) -> None:
        self.mentions_links.append((episode_id, entity_name, tenant_id))

    async def link_abstracts_to(
        self, episode_id: uuid.UUID, fact_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        self.abstracts_to_links.append((episode_id, fact_id, tenant_id))

    def set_spread_activation_results(self, results: list[ActivatedNode]) -> None:
        self._spread_activation_results = results

    async def spread_activation(
        self,
        tenant_id: uuid.UUID,
        start_entity_names: list[str],
        max_hops: int,
        decay_factor: float,
        activation_threshold: float,
    ) -> list[ActivatedNode]:
        return self._spread_activation_results


class FakeWorkingMemoryStore(WorkingMemoryStore):
    def __init__(self) -> None:
        self._turns: dict[uuid.UUID, list[WorkingMemoryTurn]] = {}

    async def push_turn(self, session_id: uuid.UUID, turn: WorkingMemoryTurn) -> None:
        self._turns.setdefault(session_id, []).append(turn)

    async def get_recent_turns(
        self, session_id: uuid.UUID, limit: int
    ) -> list[WorkingMemoryTurn]:
        # limit <= 0 -> [] , not turns[-0:] (== turns[0:], the whole list) --
        # mirrors the real store's identical guard, see
        # redis_working_memory_store.py for why.
        if limit <= 0:
            return []
        turns = self._turns.get(session_id, [])
        return turns[-limit:]
