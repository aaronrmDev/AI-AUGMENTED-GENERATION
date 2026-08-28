import uuid

from src.mag.domain.entities import EpisodicMemory, SemanticMemory, WorkingMemoryTurn
from src.mag.domain.ports import (
    EpisodicMemoryRepository,
    SemanticMemoryRepository,
    WorkingMemoryStore,
)


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

    def set_search_results(self, results: list[EpisodicMemory]) -> None:
        self._search_results = results

    async def search_by_similarity(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[EpisodicMemory]:
        return self._search_results[:top_k]


class FakeSemanticMemoryRepository(SemanticMemoryRepository):
    def __init__(self) -> None:
        self.saved: list[tuple[SemanticMemory, uuid.UUID]] = []
        # Keyed by (tenant_id, user_id, fact_key), matching the real
        # repository's ON CONFLICT (user_id, fact_key) upsert -- a second
        # save() with the same key replaces, never duplicates.
        self._by_key: dict[tuple[uuid.UUID, uuid.UUID, str], SemanticMemory] = {}
        self._search_results: list[SemanticMemory] = []

    async def save(self, fact: SemanticMemory, tenant_id: uuid.UUID) -> None:
        self.saved.append((fact, tenant_id))
        self._by_key[(tenant_id, fact.user_id, fact.fact_key)] = fact

    async def find_by_key(
        self, user_id: uuid.UUID, fact_key: str, tenant_id: uuid.UUID
    ) -> SemanticMemory | None:
        return self._by_key.get((tenant_id, user_id, fact_key))

    def set_search_results(self, results: list[SemanticMemory]) -> None:
        self._search_results = results

    async def search_by_similarity(
        self, query_embedding: list[float], user_id: uuid.UUID, tenant_id: uuid.UUID, top_k: int
    ) -> list[SemanticMemory]:
        return self._search_results[:top_k]


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
