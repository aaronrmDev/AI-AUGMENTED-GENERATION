# RAG Parent Document Chunking + Retrieval + Context Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two-tier Parent Document Chunking + Retrieval and Context Compression, then measure both plus their combination (RAG.md's "perfect balance" pairing) against a real corpus.

**Architecture:** `ParentDocumentRetriever` and `CompressingRetriever` both implement `Retriever` and both wrap an inner `Retriever` — same decorator shape `RerankingRetriever`/`HybridSearchDocuments` already established. Composition order (outermost→innermost): Compression → Parent expansion → (Reranking, if present) → search. See the design spec's "Composition order" section for why this order, not the reverse.

**Tech Stack:** Reuses `FixedSizeChunker`, `_sentence_splitter.split_sentences`, `SentenceTransformersEmbedder`, `PostgresDocumentRepository`. No new external dependency.

**Spec:** `docs/superpowers/specs/2026-08-25-rag-parent-doc-compression-design.md`

## Global Constraints

- `Retriever.execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]` — every retriever/decorator implements exactly this signature (already defined in `src/rag/domain/ports.py`).
- **Schema constraint, verified against the real migration (`alembic/versions/0002_documents_chunks.py:61`): `chunks.embedding` is `Vector(384) NOT NULL` (pgvector, fixed dimension).** A parent `Chunk`'s `embedding` field CANNOT be `[]` — `str([])` is not a valid 384-dim vector literal and `save_chunks` would fail the INSERT. Use `[0.0] * 384` (a real, dimension-valid, semantically-inert placeholder) for every parent chunk's embedding. This is a plan-level correction to the design spec's looser "`embedding=[]`" phrasing — the spec was written before this constraint was checked against the real schema; this plan is the corrected, checked version.
- `mypy --strict` and `ruff check src/` must stay clean.
- Tests: unit tests for pure logic (chunking math, expansion/dedup logic, sentence-selection math) may use fakes; anything touching a real model (`SentenceTransformersEmbedder`) or a real DB is an integration test with the real dependency, matching this project's established convention.

---

### Task 1: `ParentChildChunks` entity, `ParentDocumentChunker`, `get_chunk_by_id`, `UploadDocumentWithParents`

**Files:**
- Modify: `src/rag/domain/entities.py` — add `ParentChildChunks`
- Modify: `src/rag/domain/ports.py` — add `DocumentRepository.get_chunk_by_id`
- Modify: `src/rag/infrastructure/postgres_document_repository.py` — implement `get_chunk_by_id`
- Create: `src/rag/infrastructure/parent_document_chunker.py` — `ParentDocumentChunker`
- Create: `src/rag/application/upload_document_with_parents.py` — `UploadDocumentWithParents`
- Test: `tests/unit/test_parent_document_chunker.py`, `tests/unit/test_upload_document_with_parents.py`
- Test: add to `tests/integration/test_postgres_document_repository.py` (already exists, do not create a new file)

**Interfaces:**
- Produces: `ParentChildChunks(parents: list[str], children: list[tuple[str, int]])` (frozen dataclass)
- Produces: `ParentDocumentChunker.__init__(self, parent_chunk_size_tokens: int = 1000, child_chunk_size_tokens: int = 200) -> None`, `.chunk_with_parents(self, text: str) -> ParentChildChunks`
- Produces: `DocumentRepository.get_chunk_by_id(self, chunk_id: uuid.UUID) -> Chunk | None` (abstract + Postgres impl)
- Produces: `UploadDocumentWithParents.__init__(self, document_repository, embedding_model, vector_store, parent_document_chunker: ParentDocumentChunker, extractor, file_storage) -> None`, `.execute(self, tenant_id: uuid.UUID, filename: str, content: bytes) -> Document`
- Consumes: `FixedSizeChunker` (existing), `Chunk`/`Document` entities (existing), `TextExtractor`/`LocalFileStorage` (existing, same as `UploadDocument`)

- [ ] **Step 1: Add `ParentChildChunks`**

In `src/rag/domain/entities.py`, add:
```python
@dataclass(frozen=True)
class ParentChildChunks:
    parents: list[str]
    children: list[tuple[str, int]]  # (child content, index into parents)
```

- [ ] **Step 2: Add `get_chunk_by_id` to the port**

In `src/rag/domain/ports.py`, add to `DocumentRepository(ABC)`:
```python
    @abstractmethod
    async def get_chunk_by_id(self, chunk_id: uuid.UUID) -> Chunk | None: ...
```

- [ ] **Step 3: Write the failing tests for `ParentDocumentChunker`**

```python
# tests/unit/test_parent_document_chunker.py
from src.rag.infrastructure.parent_document_chunker import ParentDocumentChunker


def test_children_are_smaller_than_their_parent_and_reference_it_by_index():
    text = "First sentence. Second sentence. " * 200  # long enough to force multiple parents
    chunker = ParentDocumentChunker(parent_chunk_size_tokens=100, child_chunk_size_tokens=20)

    result = chunker.chunk_with_parents(text)

    assert len(result.parents) > 1
    assert len(result.children) > len(result.parents)
    for child_content, parent_index in result.children:
        assert 0 <= parent_index < len(result.parents)
        assert child_content in result.parents[parent_index]


def test_empty_text_produces_no_parents_and_no_children():
    chunker = ParentDocumentChunker()
    result = chunker.chunk_with_parents("")
    assert result.parents == []
    assert result.children == []


def test_short_text_produces_one_parent_and_at_least_one_child():
    chunker = ParentDocumentChunker(parent_chunk_size_tokens=1000, child_chunk_size_tokens=200)
    result = chunker.chunk_with_parents("A short document with just one sentence.")
    assert len(result.parents) == 1
    assert len(result.children) >= 1
    assert result.children[0][1] == 0
```

- [ ] **Step 4: Run to verify failure, then implement `ParentDocumentChunker`**

Run: `uv run pytest tests/unit/test_parent_document_chunker.py -v` — expect `ModuleNotFoundError`.

```python
# src/rag/infrastructure/parent_document_chunker.py
from src.rag.domain.entities import ParentChildChunks
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker


class ParentDocumentChunker:
    def __init__(
        self, parent_chunk_size_tokens: int = 1000, child_chunk_size_tokens: int = 200
    ) -> None:
        self._parent_chunker = FixedSizeChunker(chunk_size_tokens=parent_chunk_size_tokens)
        self._child_chunker = FixedSizeChunker(chunk_size_tokens=child_chunk_size_tokens)

    def chunk_with_parents(self, text: str) -> ParentChildChunks:
        parents = self._parent_chunker.chunk(text)
        children: list[tuple[str, int]] = []
        for i, parent in enumerate(parents):
            children.extend((child, i) for child in self._child_chunker.chunk(parent))
        return ParentChildChunks(parents=parents, children=children)
```

Run: `uv run pytest tests/unit/test_parent_document_chunker.py -v` — expect PASS, 3/3.

- [ ] **Step 5: Implement `PostgresDocumentRepository.get_chunk_by_id`, add an integration test**

Add to `tests/integration/test_postgres_document_repository.py`, matching that file's existing fixture style (read the file first — it already has a `db_session`/tenant-context pattern from the `get_chunks_for_tenant` test added in the prior batch):
```python
async def test_get_chunk_by_id_returns_the_saved_chunk():
    # follow the exact save-then-read pattern the existing
    # get_chunks_for_tenant test in this file uses
    ...

async def test_get_chunk_by_id_returns_none_for_an_unknown_id():
    ...
```

Add to `src/rag/infrastructure/postgres_document_repository.py`:
```python
    async def get_chunk_by_id(self, chunk_id: uuid.UUID) -> Chunk | None:
        result = await self._session.execute(
            text("SELECT id, document_id, content, parent_id FROM chunks WHERE id = :id"),
            {"id": chunk_id},
        )
        row = result.first()
        if row is None:
            return None
        return Chunk(
            id=row.id, document_id=row.document_id, content=row.content,
            embedding=[], parent_id=row.parent_id,
        )
```

Run the two new integration tests, expect PASS.

- [ ] **Step 6: Write the failing tests for `UploadDocumentWithParents`**

```python
# tests/unit/test_upload_document_with_parents.py
import uuid

from src.rag.application.upload_document_with_parents import UploadDocumentWithParents
from src.rag.domain.entities import ParentChildChunks


class _FakeChunker:
    def __init__(self, result: ParentChildChunks) -> None:
        self._result = result

    def chunk_with_parents(self, text: str) -> ParentChildChunks:
        return self._result


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list = []

    async def upsert(self, chunk, tenant_id) -> None:
        self.upserted.append(chunk)


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.saved_chunks: list = []

    async def save_document(self, document) -> None:
        pass

    async def update_document_status(self, document_id, status, chunk_count) -> None:
        pass

    async def save_chunks(self, chunks, tenant_id) -> None:
        self.saved_chunks.extend(chunks)

    async def get_chunks_for_tenant(self, tenant_id):
        return self.saved_chunks

    async def get_chunk_by_id(self, chunk_id):
        return next((c for c in self.saved_chunks if c.id == chunk_id), None)


class _FakeExtractor:
    def extract(self, filename: str, content: bytes) -> str:
        return content.decode("utf-8")


class _FakeFileStorage:
    def save(self, tenant_id, document_id, filename, content) -> str:
        return f"/fake/{document_id}"


async def test_only_child_chunks_are_upserted_to_the_vector_store():
    result = ParentChildChunks(parents=["parent one"], children=[("child a", 0), ("child b", 0)])
    upload = UploadDocumentWithParents(
        document_repository=_FakeDocumentRepository(),
        embedding_model=_FakeEmbedder(),
        vector_store=(vs := _FakeVectorStore()),
        parent_document_chunker=_FakeChunker(result),
        extractor=_FakeExtractor(),
        file_storage=_FakeFileStorage(),
    )

    await upload.execute(tenant_id=uuid.uuid4(), filename="doc.txt", content=b"text")

    assert len(vs.upserted) == 2  # children only, never the parent


async def test_saved_chunks_include_both_tiers_with_correct_parent_linkage():
    result = ParentChildChunks(parents=["parent one"], children=[("child a", 0)])
    repo = _FakeDocumentRepository()
    upload = UploadDocumentWithParents(
        document_repository=repo,
        embedding_model=_FakeEmbedder(),
        vector_store=_FakeVectorStore(),
        parent_document_chunker=_FakeChunker(result),
        extractor=_FakeExtractor(),
        file_storage=_FakeFileStorage(),
    )

    await upload.execute(tenant_id=uuid.uuid4(), filename="doc.txt", content=b"text")

    assert len(repo.saved_chunks) == 2  # 1 parent + 1 child
    parent_chunk = next(c for c in repo.saved_chunks if c.parent_id is None)
    child_chunk = next(c for c in repo.saved_chunks if c.parent_id is not None)
    assert parent_chunk.content == "parent one"
    assert child_chunk.content == "child a"
    assert child_chunk.parent_id == parent_chunk.id
    assert parent_chunk.embedding == [0.0] * 384  # placeholder, never searched
    assert child_chunk.embedding == [0.1, 0.2]  # real embedding
```

- [ ] **Step 7: Run to verify failure, then implement `UploadDocumentWithParents`**

```python
# src/rag/application/upload_document_with_parents.py
import uuid

from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import DocumentRepository, EmbeddingModel, VectorStore
from src.rag.infrastructure.local_file_storage import LocalFileStorage
from src.rag.infrastructure.parent_document_chunker import ParentDocumentChunker
from src.rag.infrastructure.text_extractor import TextExtractor

_EMBEDDING_DIM = 384


class UploadDocumentWithParents:
    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        parent_document_chunker: ParentDocumentChunker,
        extractor: TextExtractor,
        file_storage: LocalFileStorage,
    ) -> None:
        self._documents = document_repository
        self._embedder = embedding_model
        self._vector_store = vector_store
        self._chunker = parent_document_chunker
        self._extractor = extractor
        self._file_storage = file_storage

    async def execute(self, tenant_id: uuid.UUID, filename: str, content: bytes) -> Document:
        text = self._extractor.extract(filename, content)

        document_id = uuid.uuid4()
        storage_path = self._file_storage.save(tenant_id, document_id, filename, content)
        mime_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
        document = Document(
            id=document_id, tenant_id=tenant_id, filename=filename, mime_type=mime_type,
            storage_path=storage_path, chunk_count=0, status="processing",
        )
        await self._documents.save_document(document)

        result = self._chunker.chunk_with_parents(text)

        # Parents first: children need a real parent chunk id to link to.
        # Placeholder embedding, not []: chunks.embedding is Vector(384) NOT
        # NULL (alembic/versions/0002_documents_chunks.py) -- a parent is
        # never searched, so the value's content doesn't matter, but its
        # dimension must satisfy the column's fixed-size vector type.
        parent_chunks: list[Chunk] = [
            Chunk(
                id=uuid.uuid4(), document_id=document_id, content=parent_text,
                embedding=[0.0] * _EMBEDDING_DIM,
            )
            for parent_text in result.parents
        ]

        child_chunks: list[Chunk] = []
        for child_content, parent_index in result.children:
            embedding = self._embedder.embed(child_content)
            child_chunk = Chunk(
                id=uuid.uuid4(), document_id=document_id, content=child_content,
                embedding=embedding, parent_id=parent_chunks[parent_index].id,
            )
            child_chunks.append(child_chunk)
            await self._vector_store.upsert(child_chunk, tenant_id)

        all_chunks = parent_chunks + child_chunks
        await self._documents.save_chunks(all_chunks, tenant_id=tenant_id)
        await self._documents.update_document_status(
            document_id, status="completed", chunk_count=len(child_chunks)
        )

        return Document(
            id=document_id, tenant_id=tenant_id, filename=filename, mime_type=mime_type,
            storage_path=storage_path, chunk_count=len(child_chunks), status="completed",
        )
```

`chunk_count` counts only children (the searchable unit) — matches `Document.chunk_count`'s existing meaning throughout this codebase (how many things are in the index), not total DB rows.

Run: `uv run pytest tests/unit/test_upload_document_with_parents.py -v` — expect PASS, 2/2.

- [ ] **Step 8: Full unit + integration suite, ruff, mypy, commit**

```bash
uv run pytest tests/unit/ tests/integration/ -v
uv run ruff check src/rag/
uv run mypy src/rag/
git add src/rag/domain/entities.py src/rag/domain/ports.py src/rag/infrastructure/postgres_document_repository.py src/rag/infrastructure/parent_document_chunker.py src/rag/application/upload_document_with_parents.py tests/unit/test_parent_document_chunker.py tests/unit/test_upload_document_with_parents.py tests/integration/test_postgres_document_repository.py
git commit -m "feat: add Parent Document Chunking and the two-tier upload use case"
```

---

### Task 2: `ParentDocumentRetriever`

**Files:**
- Create: `src/rag/infrastructure/parent_document_retriever.py`
- Test: `tests/unit/test_parent_document_retriever.py`

**Interfaces:**
- Consumes: `Retriever`, `DocumentRepository.get_chunk_by_id` (Task 1), `SearchResult`/`Chunk` entities
- Produces: `ParentDocumentRetriever(Retriever).__init__(self, inner: Retriever, document_repository: DocumentRepository) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_parent_document_retriever.py
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
```

- [ ] **Step 2: Run to verify failure, implement `ParentDocumentRetriever`**

```python
# src/rag/infrastructure/parent_document_retriever.py
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import DocumentRepository, Retriever


class ParentDocumentRetriever(Retriever):
    def __init__(self, inner: Retriever, document_repository: DocumentRepository) -> None:
        self._inner = inner
        self._documents = document_repository

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        # Requests top_k, not a wider candidate pool: per the design spec's
        # resolved composition order, expansion operates on the
        # already-final ranked set, not on a pool that gets narrowed later.
        child_results = await self._inner.execute(tenant_id=tenant_id, query=query, top_k=top_k)

        expanded: list[SearchResult] = []
        seen_parent_ids: set[uuid.UUID] = set()
        for child in child_results:
            child_chunk = await self._documents.get_chunk_by_id(child.chunk_id)
            if child_chunk is None or child_chunk.parent_id is None:
                continue
            if child_chunk.parent_id in seen_parent_ids:
                continue
            parent_chunk = await self._documents.get_chunk_by_id(child_chunk.parent_id)
            if parent_chunk is None:
                continue
            seen_parent_ids.add(child_chunk.parent_id)
            expanded.append(
                SearchResult(
                    document_id=child.document_id, chunk_id=child_chunk.parent_id,
                    content=parent_chunk.content, score=child.score,
                )
            )
        return expanded
```

Run: `uv run pytest tests/unit/test_parent_document_retriever.py -v` — expect PASS, 4/4.

- [ ] **Step 3: Full suite, ruff, mypy, commit**

```bash
uv run pytest tests/unit/ tests/integration/ -v
uv run ruff check src/rag/
uv run mypy src/rag/
git add src/rag/infrastructure/parent_document_retriever.py tests/unit/test_parent_document_retriever.py
git commit -m "feat: add ParentDocumentRetriever"
```

---

### Task 3: `CompressingRetriever` (extractive Context Compression)

**Files:**
- Create: `src/rag/infrastructure/compressing_retriever.py`
- Test: `tests/unit/test_compressing_retriever.py` (fake embedder — tests the pooling/selection arithmetic in isolation)
- Test: `tests/integration/test_compressing_retriever.py` (real `embedding_model` fixture, matching the established convention)

**Interfaces:**
- Consumes: `Retriever`, `EmbeddingModel.embed`, `_sentence_splitter.split_sentences` (existing, `src/rag/infrastructure/_sentence_splitter.py`), `tiktoken` (already a dependency, same pattern as every chunker)
- Produces: `CompressingRetriever(Retriever).__init__(self, inner: Retriever, embedding_model: EmbeddingModel, target_tokens: int = 2000) -> None`

- [ ] **Step 1: Write the failing unit test (fake embedder, tests selection logic only)**

```python
# tests/unit/test_compressing_retriever.py
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.compressing_retriever import CompressingRetriever


class _FakeInner:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def execute(self, tenant_id, query, top_k):
        return self._results[:top_k]


class _KeywordOverlapEmbedder:
    # A fake that fakes "semantic similarity" as literal word overlap with
    # the query -- enough to test the pooling/greedy-selection arithmetic
    # without needing a real model in a unit test.
    def embed(self, text: str) -> list[float]:
        words = set(text.lower().split())
        vocab = ["query", "relevant", "irrelevant", "padding"]
        return [1.0 if w in words else 0.0 for w in vocab]


async def test_keeps_the_query_relevant_sentence_and_drops_the_irrelevant_one():
    results = [
        SearchResult(
            document_id=uuid.uuid4(), chunk_id=uuid.uuid4(),
            content="This sentence answers the query directly. This sentence is irrelevant padding.",
            score=0.9,
        )
    ]
    retriever = CompressingRetriever(
        inner=_FakeInner(results), embedding_model=_KeywordOverlapEmbedder(), target_tokens=10
    )

    compressed = await retriever.execute(tenant_id=uuid.uuid4(), query="query relevant", top_k=1)

    assert "answers the query directly" in compressed[0].content
    assert "irrelevant padding" not in compressed[0].content


async def test_a_result_contributing_zero_kept_sentences_is_dropped():
    results = [
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="Totally irrelevant padding here.", score=0.9),
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="This directly answers the query relevant to it.", score=0.8),
    ]
    retriever = CompressingRetriever(
        inner=_FakeInner(results), embedding_model=_KeywordOverlapEmbedder(), target_tokens=8
    )

    compressed = await retriever.execute(tenant_id=uuid.uuid4(), query="query relevant", top_k=2)

    assert len(compressed) == 1
    assert "answers the query" in compressed[0].content
```

- [ ] **Step 2: Run to verify failure, implement `CompressingRetriever`**

```python
# src/rag/infrastructure/compressing_retriever.py
import math
import uuid

import tiktoken

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import EmbeddingModel, Retriever
from src.rag.infrastructure._sentence_splitter import split_sentences


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


class CompressingRetriever(Retriever):
    def __init__(
        self, inner: Retriever, embedding_model: EmbeddingModel, target_tokens: int = 2000
    ) -> None:
        self._inner = inner
        self._embedder = embedding_model
        self._target_tokens = target_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        results = await self._inner.execute(tenant_id=tenant_id, query=query, top_k=top_k)
        if not results:
            return []

        query_embedding = self._embedder.embed(query)

        # Pool every result's sentences together (not per-result) so
        # selection can remove redundancy ACROSS chunks, not just within
        # one -- matching RAG.md's own "removes duplicate chunks... 
        # redundancy removal" framing for what compression is supposed to do.
        scored_sentences: list[tuple[float, int, int, str]] = []  # (score, result_idx, order, text)
        for result_idx, result in enumerate(results):
            for order, sentence in enumerate(split_sentences(result.content)):
                score = _cosine(query_embedding, self._embedder.embed(sentence))
                scored_sentences.append((score, result_idx, order, sentence))

        scored_sentences.sort(key=lambda s: s[0], reverse=True)

        kept_tokens = 0
        kept_by_result: dict[int, list[tuple[int, str]]] = {}
        for score, result_idx, order, sentence in scored_sentences:
            sentence_tokens = len(self._encoding.encode(sentence))
            if kept_tokens + sentence_tokens > self._target_tokens:
                continue
            kept_tokens += sentence_tokens
            kept_by_result.setdefault(result_idx, []).append((order, sentence))

        compressed: list[SearchResult] = []
        for result_idx, result in enumerate(results):
            kept = kept_by_result.get(result_idx)
            if not kept:
                continue
            kept.sort(key=lambda pair: pair[0])  # restore original sentence order
            compressed.append(
                SearchResult(
                    document_id=result.document_id, chunk_id=result.chunk_id,
                    content=" ".join(sentence for _, sentence in kept), score=result.score,
                )
            )
        return compressed
```

Run: `uv run pytest tests/unit/test_compressing_retriever.py -v` — expect PASS, 2/2.

- [ ] **Step 3: Integration test with the real embedder**

```python
# tests/integration/test_compressing_retriever.py
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
```

Run: `uv run pytest tests/integration/test_compressing_retriever.py -v` — expect PASS (real model download/load, same pattern as every other integration test in this repo).

- [ ] **Step 4: Full suite, ruff, mypy, commit**

```bash
uv run pytest tests/unit/ tests/integration/ -v
uv run ruff check src/rag/
uv run mypy src/rag/
git add src/rag/infrastructure/compressing_retriever.py tests/unit/test_compressing_retriever.py tests/integration/test_compressing_retriever.py
git commit -m "feat: add CompressingRetriever (extractive Context Compression)"
```

---

### Task 4: Scenario, live measurement, GitHub results

**Files:**
- Create: `evaluation/scenarios/rag-parent-doc-compression/corpus/rag.md` (copy of `docs/architecture/RAG.md`)
- Create: `evaluation/scenarios/rag-parent-doc-compression/queries.yaml`
- Create: `evaluation/scenarios/rag-parent-doc-compression/run_parent_doc_compression_comparison.py` — **name it exactly this, not `run_comparison.py`**: the final review on the prior batch (`feature/rag-hybrid-reranking`, already merged) found and fixed a real `mypy evaluation/` regression caused by two scenario scripts both resolving to the bare module name `run_comparison` (neither scenario directory can be a package, hyphenated names). Every scenario's entrypoint must have a name unique across the whole `evaluation/scenarios/` tree from now on.

**Interfaces:**
- Consumes everything from Tasks 1-3: `ParentDocumentChunker`, `UploadDocumentWithParents`, `ParentDocumentRetriever`, `CompressingRetriever`, plus existing `OllamaJudge`, `RunComparison`, `FixedSizeChunker`, `UploadDocument`, `AnswerQuestion`, `SearchDocuments`, `OllamaChatModel`, `QdrantVectorStore`, `SentenceTransformersEmbedder`. Pattern the whole script directly on `evaluation/scenarios/rag-hybrid-reranking/run_hybrid_reranking_comparison.py` (the most recently built and reviewed one — it already solves `PYTHONPATH`, the `.md`-extension upload workaround, `report_path.parent.mkdir`, the `_InMemoryDocumentRepository` pattern including its `get_chunk_by_id` requirement now too, and the `OllamaJudge` self-grading-bias notes text). Copy its structure, don't re-derive it.
- The in-memory document repository fake needs `get_chunk_by_id` now too (Task 1/2's new port method) — implement it as a dict lookup over the same list `get_chunks_for_tenant` already returns.

- [ ] **Step 1: Corpus**

```bash
cp docs/architecture/RAG.md evaluation/scenarios/rag-parent-doc-compression/corpus/rag.md
```

- [ ] **Step 2: `queries.yaml` — 5 real questions grounded in the live `docs/architecture/RAG.md`**

Read the current file yourself (don't reuse another batch's quotes) and write 5 questions covering: Parent Document Retrieval's mechanic and its own expected impact figure (+15-20% completeness), Context Compression's two distinct payoff numbers (the -75% worked-example figure AND the more conservative -50%/+10% general figure — RAG.md explicitly says not to conflate them, so this is a real disambiguation test, not a redundant one), the "Parent Document + Context Compression" combination's named quality ("the perfect balance"), and 2 more of your choice grounded in the same file. Quote the exact source sentence for each in your task report.

- [ ] **Step 3: `run_parent_doc_compression_comparison.py` with 3 named strategies**

```python
def _make_retriever(name, vector_retriever, document_repository, embedder):
    if name == "parent-document":
        return ParentDocumentRetriever(inner=vector_retriever, document_repository=document_repository)
    if name == "context-compression":
        return CompressingRetriever(inner=vector_retriever, embedding_model=embedder)
    if name == "parent-document-compression":
        parent = ParentDocumentRetriever(inner=vector_retriever, document_repository=document_repository)
        return CompressingRetriever(inner=parent, embedding_model=embedder)
    raise ValueError(f"unknown strategy {name!r}")
```

Only the `parent-document` and `parent-document-compression` strategies use `UploadDocumentWithParents`/`ParentDocumentChunker` for the upload; `context-compression` alone uses the plain `UploadDocument`/`FixedSizeChunker`, since compression doesn't need the two-tier index. Structure the script so the upload path branches on strategy (a simple `if strategy in (...)` around which upload use case + chunker to construct is enough — don't over-engineer a shared abstraction for 2 branches).

Baseline = no-RAG (same as every prior batch). Judge: `OllamaJudge`, same caveat comment/notes-field language as the hybrid-reranking script. `PYTHONPATH=.`, `report_path.parent.mkdir(parents=True, exist_ok=True)`, `filename="rag.txt"` upload workaround — all carried over verbatim.

**Disclose proactively, don't wait for a review to find it** (matching the pattern the last two batches' final reviews established — the controller would rather you flag a real measurement limitation now): if `candidate_k`-style pool-then-narrow doesn't apply here (it doesn't — `ParentDocumentRetriever`/`CompressingRetriever` both request exactly `top_k` from their inner, per Task 2's design), say so explicitly in a code comment near where the retriever is constructed, so nobody has to rediscover this from scratch.

- [ ] **Step 4: Report to the controller, do NOT run live comparisons**

Same boundary as every prior batch's Task 4 first half: get the code into a state the controller can run directly, verify it fails only at the point of reaching a local Qdrant instance (no Ollama/Anthropic call made), run `uv run pytest tests/unit/ tests/integration/ -v` and report the real result, run `uv run mypy src/rag/ evaluation/` (the full command, not the narrower one — this is what the prior batch's regression taught) and `uv run ruff check src/rag/ evaluation/` and report their real output too. Status DONE or DONE_WITH_CONCERNS with the quoted questions/sources and any deviations.

(Steps 5+ — bringing up Qdrant/Ollama, running the 3 live comparisons, reading results, committing, posting to GitHub issues #85, #56, #105, #61, #107, #129, #140 — are the controller's own, not part of this task's dispatch.)
