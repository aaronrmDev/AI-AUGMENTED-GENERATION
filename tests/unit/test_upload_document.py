# tests/unit/test_upload_document.py
import uuid

from src.rag.application.upload_document import UploadDocument
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.text_extractor import TextExtractor
from tests.unit.rag_fakes import FakeDocumentRepository, FakeEmbeddingModel, FakeVectorStore


async def test_upload_chunks_embeds_and_dual_writes():
    repo = FakeDocumentRepository()
    vector_store = FakeVectorStore()
    use_case = UploadDocument(
        document_repository=repo,
        embedding_model=FakeEmbeddingModel(),
        vector_store=vector_store,
        chunker=FixedSizeChunker(chunk_size_tokens=10, overlap_ratio=0.1),
        extractor=TextExtractor(),
    )

    tenant_id = uuid.uuid4()
    text = " ".join(f"word{i}" for i in range(50))
    document = await use_case.execute(
        tenant_id=tenant_id, filename="notes.txt", content=text.encode("utf-8"), storage_path="storage/x/notes.txt"
    )

    assert document.status == "completed"
    assert document.chunk_count > 1
    assert len(repo.chunks) == document.chunk_count
    assert len(vector_store.upserted) == document.chunk_count
    assert all(tenant == tenant_id for _, tenant in vector_store.upserted)


async def test_upload_rejects_an_unsupported_file_type():
    import pytest

    from src.rag.domain.errors import UnsupportedFileType

    use_case = UploadDocument(
        document_repository=FakeDocumentRepository(),
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
        chunker=FixedSizeChunker(),
        extractor=TextExtractor(),
    )
    with pytest.raises(UnsupportedFileType):
        await use_case.execute(
            tenant_id=uuid.uuid4(), filename="archive.docx", content=b"x", storage_path="storage/x/archive.docx"
        )
