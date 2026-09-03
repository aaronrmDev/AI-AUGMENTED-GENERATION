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
    result = await db_session.execute(
        text("SELECT content FROM chunks WHERE document_id = :id"), {"id": doc.id}
    )
    contents = {row.content for row in result}
    assert contents == {"chunk 0", "chunk 1", "chunk 2"}


async def test_get_chunks_for_tenant_returns_the_saved_chunks_with_no_embedding(db_session):
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

    # See the comment in test_save_document_then_update_status: SET LOCAL
    # resets at commit, so the tenant context must be re-established before
    # this post-commit read.
    await set_tenant_context(db_session, tenant_id)
    result = await repo.get_chunks_for_tenant(tenant_id)

    assert {chunk.content for chunk in result} == {"chunk 0", "chunk 1", "chunk 2"}
    # embedding=[] is deliberate: this reader never fetches the stored vector
    # back from Postgres (BM25KeywordSearch, its only consumer, never reads
    # .embedding), matching PostgresDocumentRepository.get_chunks_for_tenant.
    assert all(chunk.embedding == [] for chunk in result)


async def test_get_chunk_by_id_returns_the_saved_chunk(db_session):
    from src.identity.infrastructure.db import set_tenant_context

    tenant_id = uuid.uuid4()
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresDocumentRepository(db_session)
    doc = _new_document(tenant_id)
    await repo.save_document(doc)

    parent_id = uuid.uuid4()
    chunks = [
        Chunk(id=parent_id, document_id=doc.id, content="parent chunk", embedding=[0.0] * 384),
        Chunk(
            id=uuid.uuid4(), document_id=doc.id, content="child chunk",
            embedding=[0.0] * 384, parent_id=parent_id,
        ),
    ]
    await repo.save_chunks(chunks, tenant_id=tenant_id)
    await db_session.commit()

    # See the comment in test_save_document_then_update_status: SET LOCAL
    # resets at commit, so the tenant context must be re-established before
    # this post-commit read.
    await set_tenant_context(db_session, tenant_id)
    result = await repo.get_chunk_by_id(chunks[1].id)

    assert result is not None
    assert result.content == "child chunk"
    assert result.parent_id == parent_id


async def test_get_chunk_by_id_returns_none_for_an_unknown_id(db_session):
    from src.identity.infrastructure.db import set_tenant_context

    tenant_id = uuid.uuid4()
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresDocumentRepository(db_session)

    result = await repo.get_chunk_by_id(uuid.uuid4())

    assert result is None
