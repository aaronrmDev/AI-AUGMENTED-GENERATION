import json
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import DocumentRepository


class PostgresDocumentRepository(DocumentRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_document(self, document: Document) -> None:
        await self._session.execute(
            text(
                """
                INSERT INTO documents (
                    id, tenant_id, filename, mime_type, storage_path, chunk_count, status
                )
                VALUES (
                    :id, :tenant_id, :filename, :mime_type, :storage_path, :chunk_count, :status
                )
                """
            ),
            {
                "id": document.id,
                "tenant_id": document.tenant_id,
                "filename": document.filename,
                "mime_type": document.mime_type,
                "storage_path": document.storage_path,
                "chunk_count": document.chunk_count,
                "status": document.status,
            },
        )
        await self._session.flush()

    async def update_document_status(
        self, document_id: uuid.UUID, status: str, chunk_count: int
    ) -> None:
        await self._session.execute(
            text(
                "UPDATE documents SET status = :status, chunk_count = :chunk_count WHERE id = :id"
            ),
            {"status": status, "chunk_count": chunk_count, "id": document_id},
        )
        await self._session.flush()

    async def save_chunks(self, chunks: list[Chunk], tenant_id: uuid.UUID) -> None:
        for chunk in chunks:
            await self._session.execute(
                text(
                    """
                    INSERT INTO chunks (
                        id, document_id, content, embedding, parent_id, metadata, tenant_id
                    )
                    VALUES (
                        :id, :document_id, :content, :embedding, :parent_id, :metadata, :tenant_id
                    )
                    """
                ),
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "embedding": str(chunk.embedding),
                    "parent_id": chunk.parent_id,
                    "metadata": json.dumps(chunk.metadata),
                    "tenant_id": tenant_id,
                },
            )
        await self._session.flush()

    async def get_chunks_for_tenant(self, tenant_id: uuid.UUID) -> list[Chunk]:
        result = await self._session.execute(
            text(
                "SELECT id, document_id, content, parent_id "
                "FROM chunks WHERE tenant_id = :tenant_id"
            ),
            {"tenant_id": tenant_id},
        )
        return [
            Chunk(
                id=row.id,
                document_id=row.document_id,
                content=row.content,
                embedding=[],
                parent_id=row.parent_id,
            )
            for row in result
        ]

    async def get_chunk_by_id(self, chunk_id: uuid.UUID) -> Chunk | None:
        result = await self._session.execute(
            text("SELECT id, document_id, content, parent_id FROM chunks WHERE id = :id"),
            {"id": chunk_id},
        )
        row = result.first()
        if row is None:
            return None
        return Chunk(
            id=row.id, document_id=row.document_id, content=row.content,
            embedding=[], parent_id=row.parent_id,
        )
