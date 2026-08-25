# RAG Hybrid Search + Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Hybrid Search (vector + BM25 via RRF) and three Reranking variants (Cross-Encoder, Bi-Encoder+Rerank, LLM-based), then measure all of them plus the Hybrid+Reranking combination against a real corpus.

**Architecture:** A new `Retriever` port generalizes `SearchDocuments`; `HybridSearchDocuments` and `RerankingRetriever` both implement it, so `AnswerQuestion` never changes. `RerankingRetriever` is a decorator over any `Retriever`, which is what makes the Hybrid+Reranking combination fall out for free in Task 4.

**Tech Stack:** `rank_bm25` (new dependency), `sentence_transformers.CrossEncoder` (already-available library, new model), existing `SentenceTransformersEmbedder`/`ChatModel`/`OllamaChatModel`.

**Spec:** `docs/superpowers/specs/2026-08-24-rag-hybrid-search-reranking-design.md`

## Global Constraints

- `Retriever.execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]` — every retriever/decorator in this plan implements exactly this signature.
- RRF constant `k_rrf = 60`, standard value, not configurable.
- `candidate_k` (the wider pool size before reranking/merging narrows to `top_k`) defaults to 20 everywhere it appears.
- `mypy --strict` and `ruff check src/` must stay clean (project-wide non-negotiable, already established).
- All new tests follow the project's real-dependency convention: unit tests for pure logic (RRF math, BM25 scoring shape) may use fakes; anything touching a real model (CrossEncoder, SentenceTransformersEmbedder) is an integration test using the real model, matching `test_semantic_chunker.py`'s precedent.

---

### Task 1: `Retriever` port, `SearchDocuments` migration, BM25 keyword search, Hybrid Search

**Files:**
- Modify: `src/rag/domain/ports.py` — add `Retriever` ABC, add `get_chunks_for_tenant` to `DocumentRepository`
- Modify: `src/rag/application/search_documents.py` — `class SearchDocuments(Retriever):`
- Modify: `src/rag/application/answer_question.py` — `search_documents: Retriever` (was `SearchDocuments`)
- Modify: `src/rag/infrastructure/postgres_document_repository.py` — implement `get_chunks_for_tenant`
- Create: `src/rag/infrastructure/bm25_keyword_search.py` — `BM25KeywordSearch`
- Create: `src/rag/infrastructure/hybrid_search_documents.py` — `HybridSearchDocuments`
- Modify: `pyproject.toml` — add `rank-bm25>=0.2`
- Test: `tests/unit/test_bm25_keyword_search.py`, `tests/unit/test_hybrid_search_documents.py`
- Test: add a test to the existing `tests/integration/test_postgres_document_repository.py` (already covers `PostgresDocumentRepository` — do not create a new file)

**Interfaces:**
- Produces: `Retriever(ABC)` with `async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]: ...`
- Produces: `DocumentRepository.get_chunks_for_tenant(self, tenant_id: uuid.UUID) -> list[Chunk]` (abstract + Postgres impl)
- Produces: `BM25KeywordSearch(Retriever).__init__(self, document_repository: DocumentRepository) -> None`
- Produces: `HybridSearchDocuments(Retriever).__init__(self, vector_retriever: Retriever, keyword_retriever: Retriever, candidate_k: int = 20) -> None`
- Consumes: `SearchResult(document_id, chunk_id, content, score)` (existing entity, `src/rag/domain/entities.py`), `Chunk(id, document_id, content, embedding, parent_id, metadata)` (existing entity)

- [ ] **Step 1: Add the `Retriever` port and `DocumentRepository.get_chunks_for_tenant`**

In `src/rag/domain/ports.py`, add:

```python
class Retriever(ABC):
    @abstractmethod
    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]: ...
```

Add to the existing `DocumentRepository(ABC)`:

```python
    @abstractmethod
    async def get_chunks_for_tenant(self, tenant_id: uuid.UUID) -> list[Chunk]: ...
```

- [ ] **Step 2: Migrate `SearchDocuments` to the port, narrow `AnswerQuestion`'s type**

`src/rag/application/search_documents.py`: change `class SearchDocuments:` to `class SearchDocuments(Retriever):` and import `Retriever` from `src.rag.domain.ports`. No other change — `execute`'s body and signature already match.

`src/rag/application/answer_question.py`: change the `search_documents: SearchDocuments` constructor param to `search_documents: Retriever`, import `Retriever` instead of (or alongside) `SearchDocuments`.

- [ ] **Step 3: Run existing tests to confirm behavior-preservation**

Run: `uv run pytest tests/unit/test_search_documents.py tests/unit/test_answer_question.py -v` (check the exact existing test file names first with `ls tests/unit/ | grep -i search` / `| grep -i answer` — use whatever they're actually called)
Expected: PASS, unchanged — this step is pure type narrowing, verify with `git diff` that no logic changed if anything fails.

- [ ] **Step 4: Implement `PostgresDocumentRepository.get_chunks_for_tenant`**

In `src/rag/infrastructure/postgres_document_repository.py`, add:

```python
    async def get_chunks_for_tenant(self, tenant_id: uuid.UUID) -> list[Chunk]:
        result = await self._session.execute(
            text("SELECT id, document_id, content, parent_id FROM chunks WHERE tenant_id = :tenant_id"),
            {"tenant_id": tenant_id},
        )
        return [
            Chunk(id=row.id, document_id=row.document_id, content=row.content, embedding=[], parent_id=row.parent_id)
            for row in result
        ]
```

`embedding=[]` is deliberate: BM25 (Task 1's consumer of this method) never reads `.embedding`, and fetching every chunk's full vector back from Postgres for a method that doesn't need it wastes a real amount of I/O — the field exists on `Chunk` because the entity is shared with the embedding-bearing write path, not because every reader needs it populated.

- [ ] **Step 5: Write the failing tests for `BM25KeywordSearch`**

```python
# tests/unit/test_bm25_keyword_search.py
import uuid

from src.rag.domain.entities import Chunk
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

    results = await search.execute(tenant_id=uuid.uuid4(), query="FastAPI background tasks", top_k=2)

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
```

- [ ] **Step 5b: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_bm25_keyword_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.infrastructure.bm25_keyword_search'`

- [ ] **Step 6: Add the `rank-bm25` dependency**

In `pyproject.toml`'s `dependencies` list, add `"rank-bm25>=0.2",` (alphabetical-ish placement near the other retrieval-adjacent deps is fine, exact position doesn't matter). Run `uv sync` to install it.

- [ ] **Step 7: Implement `BM25KeywordSearch`**

```python
# src/rag/infrastructure/bm25_keyword_search.py
import re
import uuid

from rank_bm25 import BM25Okapi

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import DocumentRepository, Retriever

_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25KeywordSearch(Retriever):
    def __init__(self, document_repository: DocumentRepository) -> None:
        self._documents = document_repository

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        chunks = await self._documents.get_chunks_for_tenant(tenant_id)
        if not chunks:
            return []

        corpus = [_tokenize(chunk.content) for chunk in chunks]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))

        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(document_id=chunk.document_id, chunk_id=chunk.id, content=chunk.content, score=float(score))
            for chunk, score in ranked[:top_k]
        ]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_bm25_keyword_search.py -v`
Expected: PASS, 3/3

- [ ] **Step 9: Write the failing tests for `HybridSearchDocuments`**

```python
# tests/unit/test_hybrid_search_documents.py
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.hybrid_search_documents import HybridSearchDocuments

_TENANT = uuid.uuid4()


def _result(chunk_id: uuid.UUID, score: float) -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=chunk_id, content=f"chunk {chunk_id}", score=score)


class _FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def execute(self, tenant_id, query, top_k):
        return self._results[:top_k]


async def test_a_chunk_ranked_first_by_both_retrievers_wins_the_merge():
    shared_id = uuid.uuid4()
    vector = _FakeRetriever([_result(shared_id, 0.9), _result(uuid.uuid4(), 0.5)])
    keyword = _FakeRetriever([_result(shared_id, 12.0), _result(uuid.uuid4(), 3.0)])
    hybrid = HybridSearchDocuments(vector_retriever=vector, keyword_retriever=keyword, candidate_k=10)

    results = await hybrid.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert results[0].chunk_id == shared_id


async def test_a_chunk_found_by_only_one_retriever_still_appears():
    only_vector_id = uuid.uuid4()
    vector = _FakeRetriever([_result(only_vector_id, 0.9)])
    keyword = _FakeRetriever([_result(uuid.uuid4(), 5.0)])
    hybrid = HybridSearchDocuments(vector_retriever=vector, keyword_retriever=keyword, candidate_k=10)

    results = await hybrid.execute(tenant_id=_TENANT, query="q", top_k=5)

    assert any(r.chunk_id == only_vector_id for r in results)


async def test_respects_top_k_after_merging():
    vector = _FakeRetriever([_result(uuid.uuid4(), 1.0) for _ in range(5)])
    keyword = _FakeRetriever([_result(uuid.uuid4(), 1.0) for _ in range(5)])
    hybrid = HybridSearchDocuments(vector_retriever=vector, keyword_retriever=keyword, candidate_k=10)

    results = await hybrid.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert len(results) == 3
```

- [ ] **Step 10: Run tests to verify they fail, then implement `HybridSearchDocuments`**

Run: `uv run pytest tests/unit/test_hybrid_search_documents.py -v` — expect `ModuleNotFoundError`.

```python
# src/rag/infrastructure/hybrid_search_documents.py
import asyncio
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import Retriever

_RRF_K = 60


class HybridSearchDocuments(Retriever):
    def __init__(self, vector_retriever: Retriever, keyword_retriever: Retriever, candidate_k: int = 20) -> None:
        self._vector = vector_retriever
        self._keyword = keyword_retriever
        self._candidate_k = candidate_k

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        vector_results, keyword_results = await asyncio.gather(
            self._vector.execute(tenant_id=tenant_id, query=query, top_k=self._candidate_k),
            self._keyword.execute(tenant_id=tenant_id, query=query, top_k=self._candidate_k),
        )

        rrf_scores: dict[uuid.UUID, float] = {}
        by_id: dict[uuid.UUID, SearchResult] = {}
        for result_list in (vector_results, keyword_results):
            for rank, result in enumerate(result_list):
                rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
                by_id[result.chunk_id] = result

        merged_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        return [
            SearchResult(
                document_id=by_id[cid].document_id,
                chunk_id=cid,
                content=by_id[cid].content,
                score=rrf_scores[cid],
            )
            for cid in merged_ids[:top_k]
        ]
```

Run: `uv run pytest tests/unit/test_hybrid_search_documents.py -v` — expect PASS, 3/3.

- [ ] **Step 11: Full unit + integration suite, then commit**

Run: `uv run pytest tests/unit/ tests/integration/ -v` — expect all passing, no regressions.
Run: `uv run ruff check src/rag/` and `uv run mypy src/rag/` — expect clean.

```bash
git add src/rag/domain/ports.py src/rag/application/search_documents.py src/rag/application/answer_question.py src/rag/infrastructure/postgres_document_repository.py src/rag/infrastructure/bm25_keyword_search.py src/rag/infrastructure/hybrid_search_documents.py pyproject.toml uv.lock tests/unit/test_bm25_keyword_search.py tests/unit/test_hybrid_search_documents.py
git commit -m "feat: add Retriever port, BM25 keyword search, and Hybrid Search"
```

---

### Task 2: `Reranker` port, `RerankingRetriever` decorator, Cross-Encoder + Bi-Encoder rerankers

**Files:**
- Modify: `src/rag/domain/ports.py` — add `Reranker` ABC
- Create: `src/rag/infrastructure/reranking_retriever.py` — `RerankingRetriever`
- Create: `src/rag/infrastructure/cross_encoder_reranker.py` — `CrossEncoderReranker`
- Create: `src/rag/infrastructure/bi_encoder_rerank_reranker.py` — `BiEncoderRerankReranker`
- Test: `tests/unit/test_reranking_retriever.py`
- Test: `tests/integration/test_cross_encoder_reranker.py`, `tests/integration/test_bi_encoder_rerank_reranker.py` (real models, matching `test_semantic_chunker.py`'s convention)

**Interfaces:**
- Consumes: `Retriever` (Task 1), `SearchResult`, `EmbeddingModel.embed(text: str) -> list[float]` (existing port)
- Produces: `Reranker(ABC)` with `async def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...`
- Produces: `RerankingRetriever(Retriever).__init__(self, inner: Retriever, reranker: Reranker, candidate_k: int = 20) -> None`

- [ ] **Step 1: Add the `Reranker` port**

```python
class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...
```

- [ ] **Step 2: Write the failing test for `RerankingRetriever`**

```python
# tests/unit/test_reranking_retriever.py
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.reranking_retriever import RerankingRetriever

_TENANT = uuid.uuid4()


def _result(cid: uuid.UUID | None = None) -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=cid or uuid.uuid4(), content="c", score=1.0)


class _FakeInner:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_top_k: int | None = None

    async def execute(self, tenant_id, query, top_k):
        self.last_top_k = top_k
        return self._results[:top_k]


class _ReverseReranker:
    async def rerank(self, query, results, top_k):
        return list(reversed(results))[:top_k]


async def test_asks_the_inner_retriever_for_the_wider_candidate_pool():
    inner = _FakeInner([_result() for _ in range(20)])
    retriever = RerankingRetriever(inner=inner, reranker=_ReverseReranker(), candidate_k=15)

    await retriever.execute(tenant_id=_TENANT, query="q", top_k=3)

    assert inner.last_top_k == 15


async def test_returns_the_rerankers_output_truncated_to_top_k():
    first, second, third = _result(), _result(), _result()
    inner = _FakeInner([first, second, third])
    retriever = RerankingRetriever(inner=inner, reranker=_ReverseReranker(), candidate_k=10)

    results = await retriever.execute(tenant_id=_TENANT, query="q", top_k=2)

    assert results == [third, second]
```

- [ ] **Step 3: Run to verify failure, implement `RerankingRetriever`**

```python
# src/rag/infrastructure/reranking_retriever.py
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import Reranker, Retriever


class RerankingRetriever(Retriever):
    def __init__(self, inner: Retriever, reranker: Reranker, candidate_k: int = 20) -> None:
        self._inner = inner
        self._reranker = reranker
        self._candidate_k = candidate_k

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        candidates = await self._inner.execute(tenant_id=tenant_id, query=query, top_k=self._candidate_k)
        return await self._reranker.rerank(query=query, results=candidates, top_k=top_k)
```

Run: `uv run pytest tests/unit/test_reranking_retriever.py -v` — expect PASS, 2/2.

- [ ] **Step 4: Implement `CrossEncoderReranker` (integration test, real model)**

```python
# tests/integration/test_cross_encoder_reranker.py
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
```

```python
# src/rag/infrastructure/cross_encoder_reranker.py
from sentence_transformers import CrossEncoder

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(self) -> None:
        self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    async def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, r.content) for r in results]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [r for r, _ in ranked[:top_k]]
```

Run: `uv run pytest tests/integration/test_cross_encoder_reranker.py -v` — expect PASS (downloads the model on first run, same pattern as `SentenceTransformersEmbedder`'s tests).

- [ ] **Step 5: Implement `BiEncoderRerankReranker` (integration test, real embedder)**

```python
# tests/integration/test_bi_encoder_rerank_reranker.py
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
```

(`embedding_model` is the existing session-scoped fixture `test_semantic_chunker.py`/`tests/integration/conftest.py` already provides — reuse it, don't redefine it.)

```python
# src/rag/infrastructure/bi_encoder_rerank_reranker.py
import math
import re

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import EmbeddingModel, Reranker

_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


def _jaccard(a: str, b: str) -> float:
    tokens_a = set(_TOKEN.findall(a.lower()))
    tokens_b = set(_TOKEN.findall(b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class BiEncoderRerankReranker(Reranker):
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._embedder = embedding_model

    async def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        query_embedding = self._embedder.embed(query)
        scored = []
        for r in results:
            semantic = _cosine(query_embedding, self._embedder.embed(r.content))
            lexical = _jaccard(query, r.content)
            scored.append((r, 0.7 * semantic + 0.3 * lexical))
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return [r for r, _ in ranked[:top_k]]
```

Run: `uv run pytest tests/integration/test_bi_encoder_rerank_reranker.py -v` — expect PASS.

- [ ] **Step 6: Full unit + integration suite, ruff, mypy, then commit**

```bash
uv run pytest tests/unit/ tests/integration/ -v
uv run ruff check src/rag/
uv run mypy src/rag/
git add src/rag/domain/ports.py src/rag/infrastructure/reranking_retriever.py src/rag/infrastructure/cross_encoder_reranker.py src/rag/infrastructure/bi_encoder_rerank_reranker.py tests/unit/test_reranking_retriever.py tests/integration/test_cross_encoder_reranker.py tests/integration/test_bi_encoder_rerank_reranker.py
git commit -m "feat: add Reranker port, RerankingRetriever decorator, Cross-Encoder and Bi-Encoder rerankers"
```

---

### Task 3: `LLMReranker`

**Files:**
- Create: `src/rag/infrastructure/llm_reranker.py` — `LLMReranker`
- Test: `tests/unit/test_llm_reranker.py` (fake `ChatModel`, matching `test_ollama_chat_model.py`'s fake shape)

**Interfaces:**
- Consumes: `ChatModel.generate(question: str, context: str) -> str` (existing port)
- Produces: `LLMReranker(Reranker).__init__(self, chat_model: ChatModel) -> None`

**Design note carried from the spec:** `ChatModel`'s port shape is `generate(question, context)`, built for "answer a question given context." `LLMReranker` reuses it rather than inventing a new port — the scoring instruction and the candidate chunk both go into `question`, `context` stays `""`. This is a deliberate, pragmatic reuse of an existing generic-enough contract, not a port violation; don't add a new `Scorer` port for this.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_llm_reranker.py
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.llm_reranker import LLMReranker


class _FakeChatModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.questions: list[str] = []

    async def generate(self, question: str, context: str) -> str:
        self.questions.append(question)
        return next(self._responses)


def _result(content: str) -> SearchResult:
    return SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content=content, score=0.5)


async def test_sorts_candidates_by_parsed_score_descending():
    chat = _FakeChatModel(["3", "9"])
    reranker = LLMReranker(chat_model=chat)
    results = [_result("low relevance chunk"), _result("high relevance chunk")]

    reranked = await reranker.rerank(query="q", results=results, top_k=2)

    assert reranked[0].content == "high relevance chunk"


async def test_a_malformed_score_is_treated_as_zero_not_a_crash():
    chat = _FakeChatModel(["not a number", "7"])
    reranker = LLMReranker(chat_model=chat)
    results = [_result("garbled response chunk"), _result("clean response chunk")]

    reranked = await reranker.rerank(query="q", results=results, top_k=2)

    assert reranked[0].content == "clean response chunk"
    assert len(reranked) == 2


async def test_respects_top_k():
    chat = _FakeChatModel(["1", "2", "3"])
    reranker = LLMReranker(chat_model=chat)
    results = [_result("a"), _result("b"), _result("c")]

    reranked = await reranker.rerank(query="q", results=results, top_k=1)

    assert len(reranked) == 1
```

- [ ] **Step 2: Run to verify failure, implement `LLMReranker`**

```python
# src/rag/infrastructure/llm_reranker.py
import re

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import ChatModel, Reranker

_SCORE_PATTERN = re.compile(r"-?\d+")


class LLMReranker(Reranker):
    def __init__(self, chat_model: ChatModel) -> None:
        self._chat_model = chat_model

    async def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        scored = []
        for r in results:
            prompt = (
                f"Query: {query}\n\nCandidate passage:\n{r.content}\n\n"
                f"Score how relevant this passage is to answering the query, "
                f"from 0 (irrelevant) to 10 (directly answers it). "
                f"Respond with ONLY the integer score, nothing else."
            )
            response = await self._chat_model.generate(question=prompt, context="")
            match = _SCORE_PATTERN.search(response)
            score = int(match.group()) if match else 0
            scored.append((r, score))
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return [r for r, _ in ranked[:top_k]]
```

Run: `uv run pytest tests/unit/test_llm_reranker.py -v` — expect PASS, 3/3.

- [ ] **Step 3: Full suite, ruff, mypy, commit**

```bash
uv run pytest tests/unit/ tests/integration/ -v
uv run ruff check src/rag/
uv run mypy src/rag/
git add src/rag/infrastructure/llm_reranker.py tests/unit/test_llm_reranker.py
git commit -m "feat: add LLM-based reranker"
```

---

### Task 4: Scenario, live measurement, GitHub results

**Files:**
- Create: `evaluation/scenarios/rag-hybrid-reranking/corpus/rag.md` (copy of `docs/architecture/RAG.md`, same as the chunking-strategies scenario)
- Create: `evaluation/scenarios/rag-hybrid-reranking/queries.yaml`
- Create: `evaluation/scenarios/rag-hybrid-reranking/run_comparison.py`

**Interfaces:**
- Consumes everything from Tasks 1-3: `HybridSearchDocuments`, `RerankingRetriever`, `CrossEncoderReranker`, `BiEncoderRerankReranker`, `LLMReranker`, plus the already-existing `OllamaJudge`, `RunComparison`, `FixedSizeChunker`, `UploadDocument`, `AnswerQuestion`, `OllamaChatModel`, `QdrantVectorStore`, `PostgresDocumentRepository` — pattern the whole script directly on `evaluation/scenarios/rag-chunking-strategies/run_comparison.py`, which already solves the `PYTHONPATH`, `TextExtractor` `.md`-extension, and reports-directory-creation issues this one will hit identically. Copy that script's structure rather than re-deriving it.

- [ ] **Step 1: Corpus**

```bash
cp docs/architecture/RAG.md evaluation/scenarios/rag-hybrid-reranking/corpus/rag.md
```

- [ ] **Step 2: `queries.yaml` — 5 real questions grounded in the live `docs/architecture/RAG.md`**

Read the current file yourself and write 5 questions the way `rag-chunking-strategies/queries.yaml` did — one on RRF/Hybrid Search's merge method, one on the three reranker types' accuracy/speed/cost tradeoff, one on Reranking's expected impact figure, one on the "Hybrid Search + Reranking" combination's own name ("the production standard"), one more of your choice grounded in the same file. Quote the exact source sentence for each in your task report, the same way the chunking-strategies report did. Do not invent or approximate a quote — read the live file first.

- [ ] **Step 3: `run_comparison.py` with 6 named strategies**

Pattern directly on `evaluation/scenarios/rag-chunking-strategies/run_comparison.py`. The strategy dictionary this script needs (replacing that script's single `_make_chunker` dispatch with a retriever dispatch, chunker held fixed at `FixedSizeChunker`):

```python
def _make_retriever(name: str, vector_retriever, document_repository, embedder, chat_model):
    if name == "hybrid":
        return HybridSearchDocuments(vector_retriever, BM25KeywordSearch(document_repository))
    if name == "rerank-cross-encoder":
        return RerankingRetriever(vector_retriever, CrossEncoderReranker())
    if name == "rerank-bi-encoder":
        return RerankingRetriever(vector_retriever, BiEncoderRerankReranker(embedder))
    if name == "rerank-llm":
        return RerankingRetriever(vector_retriever, LLMReranker(chat_model))
    if name == "hybrid-rerank-cross-encoder":
        return RerankingRetriever(
            HybridSearchDocuments(vector_retriever, BM25KeywordSearch(document_repository)),
            CrossEncoderReranker(),
        )
    raise ValueError(f"unknown strategy {name!r}")
```

Every run: baseline = no-RAG (same as the chunking-strategies script), treatment = `AnswerQuestion` built with the named retriever, chunker fixed at `FixedSizeChunker()`. Judge: `OllamaJudge`, same caveat comment and `notes` field text as the chunking-strategies script (self-grading-bias disclosure) — copy that exact language. `PYTHONPATH=.` invocation, `report_path.parent.mkdir(parents=True, exist_ok=True)`, `filename="rag.txt"` for the upload — all three carried over verbatim, they are not new discoveries this task needs to re-make.

- [ ] **Step 4: Report to the controller, do NOT run live comparisons**

Same boundary as the chunking-strategies Task 6 first half: get the code into a state the controller can run directly, verify it fails only at the point of reaching a local Qdrant instance (no Ollama or Anthropic call made), run `uv run pytest tests/unit/ tests/integration/ -v` and report the real result. Status DONE or DONE_WITH_CONCERNS with the quoted questions/sources and any deviations, same report shape as the chunking-strategies precedent.

(Steps 5+ — bringing up Qdrant/Ollama, running the 6 live comparisons, reading results, committing, posting to GitHub issues #44, #102, #40, #90, #94, #98, #123, #137 — are the controller's own, not part of this task's dispatch.)
