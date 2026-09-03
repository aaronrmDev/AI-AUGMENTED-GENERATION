import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.cross_encoder_reranker import CrossEncoderReranker


async def test_promotes_the_actually_relevant_chunk_above_a_topically_similar_but_wrong_one():
    reranker = CrossEncoderReranker()
    results = [
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="Logging in FastAPI uses the standard library logging module.", score=0.5),
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="Deploy FastAPI in production using Gunicorn with Uvicorn workers behind Nginx.", score=0.5),
    ]

    reranked = await reranker.rerank(query="How to deploy FastAPI in production?", results=results, top_k=2)

    assert "Gunicorn" in reranked[0].content
