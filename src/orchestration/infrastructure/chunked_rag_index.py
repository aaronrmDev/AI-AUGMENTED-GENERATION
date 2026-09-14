import asyncio
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.infrastructure.db import set_tenant_context
from src.orchestration.domain.ports import RagIndex
from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import Chunker, EmbeddingModel
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore


class ChunkedRagIndex(RagIndex):
    """Keeps one data source's current text in RAG as a single document.

    A superseded chunk left in Qdrant would still be retrieved by vector search, and one
    left in Postgres by hybrid RAG's BM25KeywordSearch, so replace clears the previous
    version from both stores:
    - In Postgres, the delete and the new rows share one transaction, so readers see
      one version or the other.
    - In Qdrant, the new points are upserted first, then every other point of the
      document is deleted, so the document is never missing from vector search. Until
      that delete lands, both versions can be retrieved.

    If the Qdrant half fails, the router's pending marker stays set and the change is
    re-applied. Embedding runs on a worker thread, because this shares an event loop
    with the query path.
    """

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        vector_store: QdrantVectorStore,
        chunker: Chunker,
        embedder: EmbeddingModel,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._vector_store = vector_store
        self._chunker = chunker
        self._embedder = embedder

    async def replace(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, title: str, text: str
    ) -> None:
        chunks: list[Chunk] = []
        for piece in self._chunker.chunk(text):
            embedding = await asyncio.to_thread(self._embedder.embed, piece)
            chunks.append(
                Chunk(id=uuid.uuid4(), document_id=document_id, content=piece, embedding=embedding)
            )
        async with self._sessionmaker() as session, session.begin():
            await set_tenant_context(session, tenant_id)
            documents = PostgresDocumentRepository(session)
            await documents.delete_document(document_id, tenant_id)
            await documents.save_document(
                Document(
                    id=document_id,
                    tenant_id=tenant_id,
                    filename=title,
                    mime_type="text/plain",
                    storage_path=f"data-source:{title}",
                    chunk_count=len(chunks),
                    status="completed",
                )
            )
            await documents.save_chunks(chunks, tenant_id)
        for chunk in chunks:
            await self._vector_store.upsert(chunk, tenant_id)
        await self._vector_store.delete_document(
            document_id, tenant_id, keep_chunk_ids=[chunk.id for chunk in chunks]
        )
