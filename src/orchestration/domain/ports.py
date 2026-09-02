import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from src.orchestration.domain.entities import CacheHit


class AccessFrequencyTracker(ABC):
    # One shared tracking primitive behind both Cache-Warmed RAG (top-N
    # snapshot) and tiering (threshold-based promote/demote) -- OVERVIEW.md
    # describes the same underlying signal for both ("an analytics process
    # tracks which documents RAG retrieves most often" for Cache-Warmed RAG;
    # "access-pattern tracking for RAG-retrieved data" for tiering).
    # `now`/`at` are explicit parameters rather than wall-clock reads inside
    # the tracker, so tests can exercise real promotion/demotion decisions
    # over a controlled time window without sleeping through it.
    @abstractmethod
    def record_access(self, document_id: uuid.UUID, at: datetime) -> None: ...

    @abstractmethod
    def access_count(self, document_id: uuid.UUID, window: timedelta, now: datetime) -> int: ...

    @abstractmethod
    def most_accessed(self, n: int, window: timedelta, now: datetime) -> list[uuid.UUID]: ...


class FrozenCache(ABC):
    @abstractmethod
    def preload(self, document_id: uuid.UUID, content: str) -> None: ...

    @abstractmethod
    def lookup(self, document_id: uuid.UUID) -> CacheHit | None: ...

    @abstractmethod
    def evict(self, document_id: uuid.UUID) -> None: ...

    @abstractmethod
    def contains(self, document_id: uuid.UUID) -> bool: ...
