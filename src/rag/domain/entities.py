from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Document:
    id: uuid.UUID
    tenant_id: uuid.UUID
    filename: str
    mime_type: str
    storage_path: str
    chunk_count: int
    status: str  # "processing" | "completed" | "failed"
    created_at: datetime | None = None


@dataclass(frozen=True)
class Chunk:
    id: uuid.UUID
    document_id: uuid.UUID
    content: str
    embedding: list[float]
    parent_id: uuid.UUID | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    content: str
    score: float


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    sources: list[SearchResult]


@dataclass(frozen=True)
class ParentChildChunks:
    parents: list[str]
    children: list[tuple[str, int]]  # (child content, index into parents)
