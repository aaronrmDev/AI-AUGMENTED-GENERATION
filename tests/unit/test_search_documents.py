import uuid

from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import SearchResult
from tests.unit.rag_fakes import FakeEmbeddingModel, FakeVectorStore


async def test_search_embeds_the_query_and_returns_vector_store_results():
    vector_store = FakeVectorStore()
    expected = [
        SearchResult(
            document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="a match", score=0.9
        )
    ]
    vector_store.set_search_results(expected)

    use_case = SearchDocuments(embedding_model=FakeEmbeddingModel(), vector_store=vector_store)
    results = await use_case.execute(tenant_id=uuid.uuid4(), query="find this", top_k=5)

    assert results == expected


async def test_search_respects_top_k():
    vector_store = FakeVectorStore()
    vector_store.set_search_results(
        [
            SearchResult(
                document_id=uuid.uuid4(),
                chunk_id=uuid.uuid4(),
                content=f"match {i}",
                score=1.0 - i * 0.1,
            )
            for i in range(10)
        ]
    )

    use_case = SearchDocuments(embedding_model=FakeEmbeddingModel(), vector_store=vector_store)
    results = await use_case.execute(tenant_id=uuid.uuid4(), query="find this", top_k=3)

    assert len(results) == 3
