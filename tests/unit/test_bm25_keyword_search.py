import threading
import uuid

from src.rag.domain.entities import Chunk
from src.rag.infrastructure import bm25_keyword_search
from src.rag.infrastructure.bm25_keyword_search import BM25KeywordSearch


class _FakeDocumentRepository:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    async def get_chunks_for_tenant(self, tenant_id):
        return self._chunks


def _chunk(content: str) -> Chunk:
    return Chunk(id=uuid.uuid4(), document_id=uuid.uuid4(), content=content, embedding=[])


async def test_search_ranks_the_chunk_with_more_query_term_overlap_first():
    chunks = [
        _chunk("The weather today is sunny and warm."),
        _chunk("FastAPI background tasks run after the response is sent."),
    ]
    search = BM25KeywordSearch(document_repository=_FakeDocumentRepository(chunks))

    results = await search.execute(
        tenant_id=uuid.uuid4(), query="FastAPI background tasks", top_k=2
    )

    assert "FastAPI background tasks" in results[0].content


async def test_search_returns_empty_list_for_a_tenant_with_no_chunks():
    search = BM25KeywordSearch(document_repository=_FakeDocumentRepository([]))
    results = await search.execute(tenant_id=uuid.uuid4(), query="anything", top_k=5)
    assert results == []


async def test_search_respects_top_k():
    chunks = [_chunk(f"document number {i} about FastAPI") for i in range(5)]
    search = BM25KeywordSearch(document_repository=_FakeDocumentRepository(chunks))
    results = await search.execute(tenant_id=uuid.uuid4(), query="FastAPI", top_k=2)
    assert len(results) == 2


async def test_building_the_index_and_scoring_run_off_the_event_loop(monkeypatch):
    # Both grow with the tenant's corpus, so on the loop they would stall the other
    # cascade tiers sharing it in a PARALLEL route.
    ran_on: list[int] = []

    class _RecordingBM25(bm25_keyword_search.BM25Plus):
        def __init__(self, corpus):
            ran_on.append(threading.get_ident())
            super().__init__(corpus)

        def get_scores(self, query):
            ran_on.append(threading.get_ident())
            return super().get_scores(query)

    monkeypatch.setattr(bm25_keyword_search, "BM25Plus", _RecordingBM25)
    chunks = [_chunk("FastAPI background tasks"), _chunk("The weather is sunny.")]
    search = BM25KeywordSearch(document_repository=_FakeDocumentRepository(chunks))

    results = await search.execute(tenant_id=uuid.uuid4(), query="FastAPI", top_k=1)

    assert "FastAPI" in results[0].content
    assert len(ran_on) == 2
    assert threading.get_ident() not in ran_on
