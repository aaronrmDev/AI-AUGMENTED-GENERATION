from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from src.rag.domain.entities import Chunk, Document, SearchResult


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...


class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, chunk: Chunk, tenant_id: uuid.UUID) -> None: ...

    @abstractmethod
    async def search(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[SearchResult]: ...


class ChatModel(ABC):
    @abstractmethod
    async def generate(self, question: str, context: str) -> str: ...


class DocumentRepository(ABC):
    @abstractmethod
    async def save_document(self, document: Document) -> None: ...

    @abstractmethod
    async def update_document_status(
        self, document_id: uuid.UUID, status: str, chunk_count: int
    ) -> None: ...

    @abstractmethod
    async def save_chunks(self, chunks: list[Chunk], tenant_id: uuid.UUID) -> None: ...


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]: ...
