# tests/unit/test_upload_document.py
import uuid
from pathlib import Path

from src.rag.application.upload_document import UploadDocument
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.text_extractor import TextExtractor
from tests.unit.rag_fakes import (
    FakeDocumentRepository,
    FakeEmbeddingModel,
    FakeFileStorage,
    FakeVectorStore,
)


async def test_upload_chunks_embeds_and_dual_writes():
    repo = FakeDocumentRepository()
    vector_store = FakeVectorStore()
    use_case = UploadDocument(
        document_repository=repo,
        embedding_model=FakeEmbeddingModel(),
        vector_store=vector_store,
        chunker=FixedSizeChunker(chunk_size_tokens=10, overlap_ratio=0.1),
        extractor=TextExtractor(),
        file_storage=FakeFileStorage(),
    )

    tenant_id = uuid.uuid4()
    text = " ".join(f"word{i}" for i in range(50))
    document = await use_case.execute(
        tenant_id=tenant_id, filename="notes.txt", content=text.encode("utf-8")
    )

    assert document.status == "completed"
    assert document.chunk_count > 1
    assert len(repo.chunks) == document.chunk_count
    assert len(vector_store.upserted) == document.chunk_count
    assert all(tenant == tenant_id for _, tenant in vector_store.upserted)


async def test_upload_rejects_an_unsupported_file_type():
    import pytest

    from src.rag.domain.errors import UnsupportedFileType

    file_storage = FakeFileStorage()
    use_case = UploadDocument(
        document_repository=FakeDocumentRepository(),
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
        chunker=FixedSizeChunker(),
        extractor=TextExtractor(),
        file_storage=file_storage,
    )
    with pytest.raises(UnsupportedFileType):
        await use_case.execute(tenant_id=uuid.uuid4(), filename="archive.docx", content=b"x")

    # The rejection has to happen before anything is persisted: a rejected
    # upload that still left its bytes on disk is the bug this ordering fixes.
    assert file_storage.saved == []


async def test_upload_sanitizes_a_path_traversal_filename(tmp_path, monkeypatch):
    # Wired to the real LocalFileStorage against a real (temporary) filesystem
    # on purpose -- the traversal defense is that adapter's sanitization, so a
    # fake standing in for it here would test nothing. chdir because the
    # adapter resolves its "storage" root relative to the working directory.
    monkeypatch.chdir(tmp_path)
    use_case = UploadDocument(
        document_repository=FakeDocumentRepository(),
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
        chunker=FixedSizeChunker(),
        extractor=TextExtractor(),
        file_storage=LocalFileStorage(),
    )

    tenant_id = uuid.uuid4()
    document = await use_case.execute(
        tenant_id=tenant_id, filename="../../etc/passwd.txt", content=b"hello"
    )

    written = Path(document.storage_path)
    assert written.name == "passwd.txt"
    assert written.parent == Path("storage") / str(tenant_id) / str(document.id)
    assert (tmp_path / written).read_bytes() == b"hello"
    # Nothing escaped the tenant's own directory -- that one file is the only
    # thing this upload wrote anywhere under the temporary root.
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [tmp_path / written]
