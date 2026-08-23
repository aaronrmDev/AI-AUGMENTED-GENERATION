import uuid
from datetime import datetime, timezone

from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import DocumentRepository, EmbeddingModel, VectorStore
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.text_extractor import TextExtractor


class UploadDocument:
    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunker: FixedSizeChunker,
        extractor: TextExtractor,
    ) -> None:
        self._documents = document_repository
        self._embedder = embedding_model
        self._vector_store = vector_store
        self._chunker = chunker
        self._extractor = extractor

    async def execute(
        self, tenant_id: uuid.UUID, filename: str, content: bytes, storage_path: str
    ) -> Document:
        text = self._extractor.extract(filename, content)  # raises UnsupportedFileType before anything is saved

        mime_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
        document = Document(
            id=uuid.uuid4(), tenant_id=tenant_id, filename=filename, mime_type=mime_type,
            storage_path=storage_path, chunk_count=0, status="processing",
        )
        await self._documents.save_document(document)

        chunk_texts = self._chunker.chunk(text)
        chunks: list[Chunk] = []
        for chunk_text in chunk_texts:
            embedding = self._embedder.embed(chunk_text)
            chunk = Chunk(id=uuid.uuid4(), document_id=document.id, content=chunk_text, embedding=embedding)
            chunks.append(chunk)
            await self._vector_store.upsert(chunk, tenant_id)

        await self._documents.save_chunks(chunks, tenant_id=tenant_id)
        await self._documents.update_document_status(document.id, status="completed", chunk_count=len(chunks))

        return Document(
            id=document.id, tenant_id=tenant_id, filename=filename, mime_type=mime_type,
            storage_path=storage_path, chunk_count=len(chunks), status="completed",
        )
