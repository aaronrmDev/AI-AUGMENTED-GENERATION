import uuid

from src.rag.domain.entities import Chunk
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore


async def test_upsert_then_search_finds_the_chunk(qdrant_url):
    store = QdrantVectorStore(qdrant_url)
    await store.ensure_collection()

    tenant_id = uuid.uuid4()
    chunk = Chunk(
        id=uuid.uuid4(), document_id=uuid.uuid4(), content="the quick brown fox",
        embedding=[0.1] * 384,
    )
    await store.upsert(chunk, tenant_id)

    results = await store.search(query_embedding=[0.1] * 384, tenant_id=tenant_id, top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == chunk.id
    assert results[0].content == "the quick brown fox"


async def test_search_never_returns_another_tenants_chunks(qdrant_url):
    store = QdrantVectorStore(qdrant_url)
    await store.ensure_collection()

    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    chunk_a = Chunk(
        id=uuid.uuid4(), document_id=uuid.uuid4(), content="tenant a's content",
        embedding=[0.2] * 384,
    )
    chunk_b = Chunk(
        id=uuid.uuid4(), document_id=uuid.uuid4(), content="tenant b's content",
        embedding=[0.2] * 384,
    )
    await store.upsert(chunk_a, tenant_a)
    await store.upsert(chunk_b, tenant_b)

    results = await store.search(query_embedding=[0.2] * 384, tenant_id=tenant_a, top_k=10)
    chunk_ids = {r.chunk_id for r in results}
    assert chunk_a.id in chunk_ids
    assert chunk_b.id not in chunk_ids
