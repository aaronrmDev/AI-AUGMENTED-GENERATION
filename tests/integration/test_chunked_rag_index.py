import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.orchestration.infrastructure.chunked_rag_index import ChunkedRagIndex
from src.rag.domain.entities import Chunk, Document
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore

V1 = " ".join(["The return window for unopened items is forty-five days."] * 6)
V2 = " ".join(["The return window for unopened items is sixty days."] * 6)


async def _stores(db_session, qdrant_url, embedding_model):
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    chunker = FixedSizeChunker(chunk_size_tokens=16, overlap_ratio=0.0)
    index = ChunkedRagIndex(
        get_sessionmaker(db_session.bind), vector_store, chunker, embedding_model
    )
    return index, vector_store


async def _postgres_chunks(db_session, tenant_id, document_id) -> list[str]:
    await set_tenant_context(db_session, tenant_id)
    result = await db_session.execute(
        text("SELECT content FROM chunks WHERE document_id = :id"), {"id": document_id}
    )
    return [row.content for row in result]


async def test_replacing_a_source_leaves_only_its_new_text_in_both_stores(
    db_session, qdrant_url, embedding_model
):
    index, vector_store = await _stores(db_session, qdrant_url, embedding_model)
    tenant_id, document_id = uuid.uuid4(), uuid.uuid4()

    await index.replace(tenant_id, document_id, "return-policy", V1)
    await index.replace(tenant_id, document_id, "return-policy", V2)

    stored = await _postgres_chunks(db_session, tenant_id, document_id)
    assert len(stored) > 1  # really chunked, so every chunk had to be replaced
    assert not any("forty" in chunk for chunk in stored)
    rows = (
        await db_session.execute(
            text("SELECT chunk_count FROM documents WHERE id = :id"), {"id": document_id}
        )
    ).all()
    assert [row.chunk_count for row in rows] == [len(stored)]

    hits = await vector_store.search(embedding_model.embed(V1), tenant_id, top_k=50)
    assert hits
    assert all(hit.document_id == document_id for hit in hits)
    assert not any("forty" in hit.content for hit in hits)


async def test_qdrant_delete_is_scoped_by_tenant(db_session, qdrant_url, embedding_model):
    index, vector_store = await _stores(db_session, qdrant_url, embedding_model)
    tenant_id, other_tenant, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    foreign = Chunk(
        uuid.uuid4(), document_id, "another tenant's text", embedding_model.embed("x")
    )
    await vector_store.upsert(foreign, other_tenant)

    await index.replace(tenant_id, document_id, "return-policy", V2)

    hits = await vector_store.search(embedding_model.embed("x"), other_tenant, top_k=5)
    assert [hit.chunk_id for hit in hits] == [foreign.id]


async def test_qdrant_delete_keeps_the_named_chunks_of_the_document(qdrant_url, embedding_model):
    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    tenant_id, document_id = uuid.uuid4(), uuid.uuid4()
    vector = embedding_model.embed("return window")
    old, new = (Chunk(uuid.uuid4(), document_id, text, vector) for text in ("old", "new"))
    for chunk in (old, new):
        await vector_store.upsert(chunk, tenant_id)

    await vector_store.delete_document(document_id, tenant_id, keep_chunk_ids=[new.id])

    hits = await vector_store.search(vector, tenant_id, top_k=5)
    assert [hit.chunk_id for hit in hits] == [new.id]


async def test_postgres_delete_document_removes_chunks_then_the_row(db_session, embedding_model):
    tenant_id, document_id = uuid.uuid4(), uuid.uuid4()
    await set_tenant_context(db_session, tenant_id)
    repository = PostgresDocumentRepository(db_session)
    await repository.save_document(
        Document(document_id, tenant_id, "f", "text/plain", "p", 1, "completed")
    )
    await repository.save_chunks(
        [Chunk(uuid.uuid4(), document_id, "text", embedding_model.embed("text"))], tenant_id
    )

    await repository.delete_document(document_id, tenant_id)
    await db_session.commit()

    await set_tenant_context(db_session, tenant_id)
    counts = (
        await db_session.execute(
            text(
                "SELECT (SELECT count(*) FROM chunks WHERE document_id = :id) AS chunks, "
                "(SELECT count(*) FROM documents WHERE id = :id) AS documents"
            ),
            {"id": document_id},
        )
    ).one()
    assert (counts.chunks, counts.documents) == (0, 0)
