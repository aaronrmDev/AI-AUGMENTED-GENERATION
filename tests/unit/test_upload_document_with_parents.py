# tests/unit/test_upload_document_with_parents.py
import uuid

from src.rag.application.upload_document_with_parents import UploadDocumentWithParents
from src.rag.domain.entities import ParentChildChunks


class _FakeChunker:
    def __init__(self, result: ParentChildChunks) -> None:
        self._result = result

    def chunk_with_parents(self, text: str) -> ParentChildChunks:
        return self._result


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list = []

    async def upsert(self, chunk, tenant_id) -> None:
        self.upserted.append(chunk)


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.saved_chunks: list = []

    async def save_document(self, document) -> None:
        pass

    async def update_document_status(self, document_id, status, chunk_count) -> None:
        pass

    async def save_chunks(self, chunks, tenant_id) -> None:
        self.saved_chunks.extend(chunks)

    async def get_chunks_for_tenant(self, tenant_id):
        return self.saved_chunks

    async def get_chunk_by_id(self, chunk_id):
        return next((c for c in self.saved_chunks if c.id == chunk_id), None)


class _FakeExtractor:
    def extract(self, filename: str, content: bytes) -> str:
        return content.decode("utf-8")


class _FakeFileStorage:
    def save(self, tenant_id, document_id, filename, content) -> str:
        return f"/fake/{document_id}"


async def test_only_child_chunks_are_upserted_to_the_vector_store():
    result = ParentChildChunks(parents=["parent one"], children=[("child a", 0), ("child b", 0)])
    upload = UploadDocumentWithParents(
        document_repository=_FakeDocumentRepository(),
        embedding_model=_FakeEmbedder(),
        vector_store=(vs := _FakeVectorStore()),
        parent_document_chunker=_FakeChunker(result),
        extractor=_FakeExtractor(),
        file_storage=_FakeFileStorage(),
    )

    await upload.execute(tenant_id=uuid.uuid4(), filename="doc.txt", content=b"text")

    assert len(vs.upserted) == 2  # children only, never the parent


async def test_saved_chunks_include_both_tiers_with_correct_parent_linkage():
    result = ParentChildChunks(parents=["parent one"], children=[("child a", 0)])
    repo = _FakeDocumentRepository()
    upload = UploadDocumentWithParents(
        document_repository=repo,
        embedding_model=_FakeEmbedder(),
        vector_store=_FakeVectorStore(),
        parent_document_chunker=_FakeChunker(result),
        extractor=_FakeExtractor(),
        file_storage=_FakeFileStorage(),
    )

    await upload.execute(tenant_id=uuid.uuid4(), filename="doc.txt", content=b"text")

    assert len(repo.saved_chunks) == 2  # 1 parent + 1 child
    parent_chunk = next(c for c in repo.saved_chunks if c.parent_id is None)
    child_chunk = next(c for c in repo.saved_chunks if c.parent_id is not None)
    assert parent_chunk.content == "parent one"
    assert child_chunk.content == "child a"
    assert child_chunk.parent_id == parent_chunk.id
    assert parent_chunk.embedding == [0.0] * 384  # placeholder, never searched
    assert child_chunk.embedding == [0.1, 0.2]  # real embedding
