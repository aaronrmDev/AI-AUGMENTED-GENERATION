import uuid

from src.rag.domain.entities import Chunk, Document, SearchResult
from src.rag.domain.ports import ChatModel, DocumentRepository, EmbeddingModel, VectorStore


class FakeEmbeddingModel(EmbeddingModel):
    def embed(self, text: str) -> list[float]:
        # Deterministic, cheap stand-in: length-derived vector, not a real embedding.
        return [float(len(text) % 7)] * 384


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.upserted: list[tuple[Chunk, uuid.UUID]] = []
        self._search_results: list[SearchResult] = []

    async def upsert(self, chunk: Chunk, tenant_id: uuid.UUID) -> None:
        self.upserted.append((chunk, tenant_id))

    def set_search_results(self, results: list[SearchResult]) -> None:
        self._search_results = results

    async def search(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[SearchResult]:
        return self._search_results[:top_k]


class FakeChatModel(ChatModel):
    def __init__(self, response: str = "a fake answer") -> None:
        self._response = response
        self.last_question: str | None = None
        self.last_context: str | None = None

    async def generate(self, question: str, context: str) -> str:
        self.last_question = question
        self.last_context = context
        return self._response


class FakeFileStorage:
    """Records what the use case handed the storage adapter, and writes nothing.

    Deliberately does NOT sanitize: sanitizing here would move the defense
    under test into the test double, so a traversal assertion against this
    fake would only be proving the fake correct. The real defense lives in
    LocalFileStorage and is exercised against a real filesystem by
    test_upload_sanitizes_a_path_traversal_filename.
    """

    def __init__(self) -> None:
        self.saved: list[tuple[uuid.UUID, uuid.UUID, str, bytes]] = []

    def save(
        self, tenant_id: uuid.UUID, document_id: uuid.UUID, filename: str, content: bytes
    ) -> str:
        self.saved.append((tenant_id, document_id, filename, content))
        return f"storage/{tenant_id}/{document_id}/{filename}"


class FakeDocumentRepository(DocumentRepository):
    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, Document] = {}
        self.chunks: list[Chunk] = []

    async def save_document(self, document: Document) -> None:
        self.documents[document.id] = document

    async def update_document_status(
        self, document_id: uuid.UUID, status: str, chunk_count: int
    ) -> None:
        doc = self.documents[document_id]
        self.documents[document_id] = Document(
            id=doc.id, tenant_id=doc.tenant_id, filename=doc.filename, mime_type=doc.mime_type,
            storage_path=doc.storage_path, chunk_count=chunk_count, status=status,
        )

    async def save_chunks(self, chunks: list[Chunk], tenant_id: uuid.UUID) -> None:
        self.chunks.extend(chunks)

    async def get_chunks_for_tenant(self, tenant_id: uuid.UUID) -> list[Chunk]:
        return self.chunks
