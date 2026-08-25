import uuid

from src.rag.domain.entities import Chunk, SearchResult
from src.rag.infrastructure.parent_document_retriever import ParentDocumentRetriever

_TENANT = uuid.uuid4()


class _FakeInner:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def execute(self, tenant_id, query, top_k):
        return self._results[:top_k]


class _FakeDocumentRepository:
    def __init__(self, chunks_by_id: dict[uuid.UUID, Chunk]) -> None:
        self._chunks = chunks_by_id

    async def get_chunk_by_id(self, chunk_id):
        return self._chunks.get(chunk_id)


def _child_result(chunk_id: uuid.UUID) -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=chunk_id, content="child", score=0.9)


async def test_a_matched_child_is_expanded_to_its_parents_content():
    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()
    chunks = {
        child_id: Chunk(id=child_id, document_id=uuid.uuid4(), content="child", embedding=[], parent_id=parent_id),
        parent_id: Chunk(id=parent_id, document_id=uuid.uuid4(), content="full parent section", embedding=[]),
    }
    retriever = ParentDocumentRetriever(
        inner=_FakeInner([_child_result(child_id)]),
        document_repository=_FakeDocumentRepository(chunks),
    )

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert len(results) == 1
    assert results[0].content == "full parent section"
    assert results[0].chunk_id == parent_id


async def test_two_children_of_the_same_parent_collapse_to_one_result():
    parent_id = uuid.uuid4()
    child_a, child_b = uuid.uuid4(), uuid.uuid4()
    chunks = {
        child_a: Chunk(id=child_a, document_id=uuid.uuid4(), content="a", embedding=[], parent_id=parent_id),
        child_b: Chunk(id=child_b, document_id=uuid.uuid4(), content="b", embedding=[], parent_id=parent_id),
        parent_id: Chunk(id=parent_id, document_id=uuid.uuid4(), content="shared parent", embedding=[]),
    }
    retriever = ParentDocumentRetriever(
        inner=_FakeInner([_child_result(child_a), _child_result(child_b)]),
        document_repository=_FakeDocumentRepository(chunks),
    )

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert len(results) == 1


async def test_a_child_with_no_parent_id_is_skipped():
    orphan_id = uuid.uuid4()
    chunks = {
        orphan_id: Chunk(id=orphan_id, document_id=uuid.uuid4(), content="orphan", embedding=[], parent_id=None),
    }
    retriever = ParentDocumentRetriever(
        inner=_FakeInner([_child_result(orphan_id)]),
        document_repository=_FakeDocumentRepository(chunks),
    )

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert results == []


async def test_requests_top_k_from_inner_not_a_wider_pool():
    class _SpyInner:
        def __init__(self) -> None:
            self.last_top_k = None

        async def execute(self, tenant_id, query, top_k):
            self.last_top_k = top_k
            return []

    inner = _SpyInner()
    retriever = ParentDocumentRetriever(inner=inner, document_repository=_FakeDocumentRepository({}))

    await retriever.execute(tenant_id=_TENANT, query="q", top_k=7)

    assert inner.last_top_k == 7
