import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.bi_encoder_rerank_reranker import BiEncoderRerankReranker


async def test_promotes_the_chunk_with_more_semantic_and_lexical_overlap(embedding_model):
    reranker = BiEncoderRerankReranker(embedding_model=embedding_model)
    results = [
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="Cats are small domesticated mammals.", score=0.5),
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="FastAPI background tasks run after the response is returned to the client.", score=0.5),
    ]

    reranked = await reranker.rerank(query="FastAPI background tasks", results=results, top_k=2)

    assert "background tasks" in reranked[0].content
