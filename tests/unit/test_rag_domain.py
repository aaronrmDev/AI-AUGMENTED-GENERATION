import uuid
from datetime import UTC, datetime

from src.rag.domain.entities import Chunk, Document
from src.rag.domain.errors import UnsupportedFileType


def test_document_equality_is_by_all_fields():
    now = datetime.now(UTC)
    shared_id = uuid.uuid4()
    tenant = uuid.uuid4()
    a = Document(
        id=shared_id, tenant_id=tenant, filename="a.txt", mime_type="text/plain",
        storage_path="storage/x/a.txt", chunk_count=0, status="processing", created_at=now,
    )
    b = Document(
        id=shared_id, tenant_id=tenant, filename="a.txt", mime_type="text/plain",
        storage_path="storage/x/a.txt", chunk_count=0, status="processing", created_at=now,
    )
    assert a == b


def test_chunk_defaults_parent_id_none_and_metadata_empty():
    chunk = Chunk(
        id=uuid.uuid4(), document_id=uuid.uuid4(), content="some text",
        embedding=[0.1, 0.2, 0.3],
    )
    assert chunk.parent_id is None
    assert chunk.metadata == {}


def test_unsupported_file_type_error_names_the_extension():
    err = UnsupportedFileType(".docx")
    assert ".docx" in str(err)
