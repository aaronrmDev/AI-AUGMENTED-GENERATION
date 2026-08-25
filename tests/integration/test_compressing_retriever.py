import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.compressing_retriever import CompressingRetriever


class _FakeInner:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def execute(self, tenant_id, query, top_k):
        return self._results[:top_k]


async def test_compresses_toward_the_query_relevant_sentence_with_a_real_embedder(embedding_model):
    results = [
        SearchResult(
            document_id=uuid.uuid4(), chunk_id=uuid.uuid4(),
            content=(
                "FastAPI background tasks run after the response is sent to the client. "
                "The weather today is sunny and unrelated to this topic."
            ),
            score=0.9,
        )
    ]
    retriever = CompressingRetriever(inner=_FakeInner(results), embedding_model=embedding_model, target_tokens=15)

    compressed = await retriever.execute(
        tenant_id=uuid.uuid4(), query="How do FastAPI background tasks work?", top_k=1
    )

    assert len(compressed) == 1
    assert "background tasks" in compressed[0].content
    assert "weather" not in compressed[0].content
