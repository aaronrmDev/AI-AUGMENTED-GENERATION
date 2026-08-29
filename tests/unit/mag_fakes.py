import dataclasses
import math
import uuid
from datetime import UTC, datetime

from src.mag.domain.entities import (
    EpisodicMemory,
    ProceduralMemory,
    ScoredEpisode,
    ScoredFact,
    SemanticMemory,
    WorkingMemoryTurn,
)
from src.mag.domain.ports import (
    EpisodicMemoryRepository,
    ProceduralMemoryRepository,
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
        self, session_id: uuid.UUID, tenant_id: uuid.UUID, start: datetime, end: datetime
    ) -> list[EpisodicMemory]:
        episodes = self._by_session.get(session_id, [])
        matches = [e for e in episodes if start <= e.timestamp <= end]
        return sorted(matches, key=lambda e: e.timestamp, reverse=True)

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
        needle = entity.lower()

        def _matches(e: EpisodicMemory) -> bool:
            structured = [str(x).lower() for x in e.content.get("entities", [])]
            if needle in structured:
                return True
            return needle in str(e.content).lower()

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

    async def save(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None:
        self.saved.append((fact, tenant_id))
        self._by_key[(fact.user_id, fact.fact_key)] = fact

    async def find_by_key(
        self, user_id: uuid.UUID, fact_key: str, tenant_id: uuid.UUID
    ) -> SemanticMemory | None:
        return self._by_key.get((user_id, fact_key))

    def set_search_results(self, results: list[SemanticMemory]) -> None:
        self._search_results = results

    async def search_by_similarity(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[ScoredFact]:
        scored = [
            ScoredFact(fact=f, score=_cosine_similarity(query_embedding, f.embedding))
            for f in self._search_results
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


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
