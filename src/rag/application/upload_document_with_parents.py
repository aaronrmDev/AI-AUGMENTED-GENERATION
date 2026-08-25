import uuid

from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import DocumentRepository, EmbeddingModel, VectorStore
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.parent_document_chunker import ParentDocumentChunker
from src.rag.infrastructure.text_extractor import TextExtractor

_EMBEDDING_DIM = 384


class UploadDocumentWithParents:
    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        parent_document_chunker: ParentDocumentChunker,
        extractor: TextExtractor,
        file_storage: LocalFileStorage,
    ) -> None:
        self._documents = document_repository
        self._embedder = embedding_model
        self._vector_store = vector_store
        self._chunker = parent_document_chunker
        self._extractor = extractor
        self._file_storage = file_storage

    async def execute(self, tenant_id: uuid.UUID, filename: str, content: bytes) -> Document:
        text = self._extractor.extract(filename, content)

        document_id = uuid.uuid4()
        storage_path = self._file_storage.save(tenant_id, document_id, filename, content)
        mime_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
        document = Document(
            id=document_id, tenant_id=tenant_id, filename=filename, mime_type=mime_type,
            storage_path=storage_path, chunk_count=0, status="processing",
        )
        await self._documents.save_document(document)

        result = self._chunker.chunk_with_parents(text)

        # Parents first: children need a real parent chunk id to link to.
        # Placeholder embedding, not []: chunks.embedding is Vector(384) NOT
        # NULL (alembic/versions/0002_documents_chunks.py) -- a parent is
        # never searched, so the value's content doesn't matter, but its
        # dimension must satisfy the column's fixed-size vector type.
        parent_chunks: list[Chunk] = [
            Chunk(
                id=uuid.uuid4(), document_id=document_id, content=parent_text,
                embedding=[0.0] * _EMBEDDING_DIM,
            )
            for parent_text in result.parents
        ]

        child_chunks: list[Chunk] = []
        for child_content, parent_index in result.children:
            embedding = self._embedder.embed(child_content)
            child_chunk = Chunk(
                id=uuid.uuid4(), document_id=document_id, content=child_content,
                embedding=embedding, parent_id=parent_chunks[parent_index].id,
            )
            child_chunks.append(child_chunk)
            await self._vector_store.upsert(child_chunk, tenant_id)

        all_chunks = parent_chunks + child_chunks
        await self._documents.save_chunks(all_chunks, tenant_id=tenant_id)
        await self._documents.update_document_status(
            document_id, status="completed", chunk_count=len(child_chunks)
        )

        return Document(
            id=document_id, tenant_id=tenant_id, filename=filename, mime_type=mime_type,
            storage_path=storage_path, chunk_count=len(child_chunks), status="completed",
        )
