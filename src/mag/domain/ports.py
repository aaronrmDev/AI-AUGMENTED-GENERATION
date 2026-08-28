from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from src.mag.domain.entities import EpisodicMemory, SemanticMemory, WorkingMemoryTurn


class EpisodicMemoryRepository(ABC):
    @abstractmethod
    async def save(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None: ...

    @abstractmethod
    async def get_by_session(
        self, session_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> list[EpisodicMemory]: ...

    @abstractmethod
    async def search_by_similarity(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[EpisodicMemory]: ...


class SemanticMemoryRepository(ABC):
    @abstractmethod
    async def save(self, fact: SemanticMemory) -> None: ...

    @abstractmethod
    async def find_by_key(self, user_id: uuid.UUID, fact_key: str) -> SemanticMemory | None: ...

    @abstractmethod
    async def search_by_similarity(
        self, query_embedding: list[float], user_id: uuid.UUID, top_k: int
    ) -> list[SemanticMemory]: ...


class WorkingMemoryStore(ABC):
    @abstractmethod
    async def push_turn(self, session_id: uuid.UUID, turn: WorkingMemoryTurn) -> None: ...

    @abstractmethod
    async def get_recent_turns(
        self, session_id: uuid.UUID, limit: int
    ) -> list[WorkingMemoryTurn]: ...
