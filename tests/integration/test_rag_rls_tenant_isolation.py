import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.rag.domain.entities import Chunk, Document
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository


async def test_rls_returns_zero_cross_tenant_chunks_even_without_an_app_level_filter(db_session):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    repo = PostgresDocumentRepository(db_session)

    await set_tenant_context(db_session, tenant_a)
    doc_a = Document(
        id=uuid.uuid4(), tenant_id=tenant_a, filename="a.txt", mime_type="text/plain",
        storage_path="storage/a/a.txt", chunk_count=1, status="completed",
    )
    await repo.save_document(doc_a)
    await repo.save_chunks(
        [
            Chunk(
                id=uuid.uuid4(), document_id=doc_a.id, content="tenant a's chunk",
                embedding=[0.0] * 384,
            )
        ],
        tenant_id=tenant_a,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_b)
    doc_b = Document(
        id=uuid.uuid4(), tenant_id=tenant_b, filename="b.txt", mime_type="text/plain",
        storage_path="storage/b/b.txt", chunk_count=1, status="completed",
    )
    await repo.save_document(doc_b)
    await repo.save_chunks(
        [
            Chunk(
                id=uuid.uuid4(), document_id=doc_b.id, content="tenant b's chunk",
                embedding=[0.0] * 384,
            )
        ],
        tenant_id=tenant_b,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    # Deliberately no WHERE tenant_id = ... — RLS alone must do the filtering.
    result = await db_session.execute(text("SELECT content FROM chunks"))
    contents = {row.content for row in result}

    assert contents == {"tenant a's chunk"}
    assert "tenant b's chunk" not in contents
