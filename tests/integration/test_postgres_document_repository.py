import uuid

from src.rag.domain.entities import Chunk, Document
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository


def _new_document(tenant_id: uuid.UUID) -> Document:
    return Document(
        id=uuid.uuid4(), tenant_id=tenant_id, filename="notes.txt", mime_type="text/plain",
        storage_path=f"storage/{tenant_id}/notes.txt", chunk_count=0, status="processing",
    )


async def test_save_document_then_update_status(db_session):
    from src.identity.infrastructure.db import set_tenant_context

    tenant_id = uuid.uuid4()
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresDocumentRepository(db_session)
    doc = _new_document(tenant_id)
    await repo.save_document(doc)
    await repo.update_document_status(doc.id, status="completed", chunk_count=3)
    await db_session.commit()

    from sqlalchemy import text

    # set_tenant_context uses SET LOCAL semantics (see its docstring in
    # src/identity/infrastructure/db.py), so it resets at the commit above;
    # the verification query below runs in a new transaction and needs the
    # tenant context re-established, matching the pattern already used in
    # tests/integration/test_rls_tenant_isolation.py for the same reason.
    await set_tenant_context(db_session, tenant_id)
    result = await db_session.execute(
        text("SELECT status, chunk_count FROM documents WHERE id = :id"), {"id": doc.id}
    )
    row = result.mappings().first()
    assert row["status"] == "completed"
    assert row["chunk_count"] == 3


async def test_save_chunks_batch_inserts_all_of_them(db_session):
    from src.identity.infrastructure.db import set_tenant_context

    tenant_id = uuid.uuid4()
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresDocumentRepository(db_session)
    doc = _new_document(tenant_id)
    await repo.save_document(doc)

    chunks = [
        Chunk(id=uuid.uuid4(), document_id=doc.id, content=f"chunk {i}", embedding=[0.0] * 384)
        for i in range(3)
    ]
    await repo.save_chunks(chunks, tenant_id=tenant_id)
    await db_session.commit()

    from sqlalchemy import text

    # See the comment in test_save_document_then_update_status: SET LOCAL
    # resets at commit, so the tenant context must be re-established before
    # this post-commit verification query.
    await set_tenant_context(db_session, tenant_id)
    result = await db_session.execute(text("SELECT content FROM chunks WHERE document_id = :id"), {"id": doc.id})
    contents = {row.content for row in result}
    assert contents == {"chunk 0", "chunk 1", "chunk 2"}
