import uuid

from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import Chunker, DocumentRepository, EmbeddingModel, VectorStore
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.text_extractor import TextExtractor


class UploadDocument:
    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunker: Chunker,
        extractor: TextExtractor,
        file_storage: LocalFileStorage,
    ) -> None:
        self._documents = document_repository
        self._embedder = embedding_model
        self._vector_store = vector_store
        self._chunker = chunker
        self._extractor = extractor
        self._file_storage = file_storage

    async def execute(self, tenant_id: uuid.UUID, filename: str, content: bytes) -> Document:
        # Extraction runs first so an unsupported file type raises before a
        # single byte reaches disk. Persisting is this use case's job, not the
        # router's: the storage path is derived from the document id, which
        # only exists once the upload is known to be processable at all.
        text = self._extractor.extract(filename, content)

        document_id = uuid.uuid4()
        storage_path = self._file_storage.save(tenant_id, document_id, filename, content)

        mime_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
        document = Document(
            id=document_id, tenant_id=tenant_id, filename=filename, mime_type=mime_type,
            storage_path=storage_path, chunk_count=0, status="processing",
        )
        await self._documents.save_document(document)

        chunk_texts = self._chunker.chunk(text)
        chunks: list[Chunk] = []
        for chunk_text in chunk_texts:
            embedding = self._embedder.embed(chunk_text)
            chunk = Chunk(
                id=uuid.uuid4(), document_id=document.id, content=chunk_text, embedding=embedding
            )
            chunks.append(chunk)
            await self._vector_store.upsert(chunk, tenant_id)

        await self._documents.save_chunks(chunks, tenant_id=tenant_id)
        await self._documents.update_document_status(
            document.id, status="completed", chunk_count=len(chunks)
        )

        return Document(
            id=document.id, tenant_id=tenant_id, filename=filename, mime_type=mime_type,
            storage_path=storage_path, chunk_count=len(chunks), status="completed",
        )
