# RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Epic #144's second Story — document upload, fixed-size chunking with local MiniLM embeddings, a vector search endpoint, and a synchronous chat endpoint answering from retrieved context — as protected, tenant-scoped routes built on Auth Foundation's existing JWT/RLS infrastructure.

**Architecture:** A new `src/rag/` paradigm module, hexagonal like `src/identity/` — `domain/` (framework-free entities and port interfaces), `application/` (use cases orchestrating ports), `infrastructure/` (concrete Postgres/Qdrant/MiniLM/Claude adapters), driven by new routers in the existing `src/api/`. Unit tests exercise `domain/`/`application/` against fakes; integration tests exercise `infrastructure/` and the full HTTP stack against TestContainers-provisioned PostgreSQL, Redis, and Qdrant, plus a real, session-scoped MiniLM model load.

**Tech Stack:** Everything Auth Foundation already established (Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2.0 async, Alembic, uv, pytest/TestContainers, mypy, ruff), plus this sub-project's own additions: `sentence-transformers` (MiniLM embeddings), `qdrant-client` (Qdrant Python client), `pgvector` (the `Vector` SQLAlchemy column type), `pypdf` (PDF text extraction), `tiktoken` (token counting for chunk sizing — used purely as a fast, standard, widely-available BPE tokenizer for measuring chunk length; not tied to any specific model's actual tokenizer, since chunk-size targets are inherently approximate), and the `anthropic` SDK (Claude API, per this project's `claude-api` skill conventions).

**Spec:** docs/superpowers/specs/2026-08-22-rag-pipeline-design.md

**Tracking:** GitHub Epic #144 (Story to be filed alongside Task 1's dispatch — see the Setup note below).

## Global Constraints

- `src/rag/domain/` has zero framework imports (no FastAPI/SQLAlchemy/Qdrant/sentence-transformers/anthropic) — identical rule to `src/identity/domain/`.
- `src/rag/application/` depends only on `src/rag/domain/ports.py` interfaces, never a concrete `src/rag/infrastructure/` class directly.
- Chunking is fixed-size: 512 tokens per chunk, 10% overlap, token length measured via `tiktoken`'s `cl100k_base` encoding (an approximation — chunk boundaries aren't tied to MiniLM's or Claude's own tokenizer).
- Supported upload types are `.txt` (decoded directly as UTF-8) and `.pdf` (text extracted via `pypdf`) — any other extension raises the domain's `UnsupportedFileType`, mapped to a `422`.
- Embeddings are 384-dimensional, produced by `sentence-transformers`' `all-MiniLM-L6-v2` model, loaded once (a module-level singleton in the running app, a session-scoped fixture in tests — the same "load once, reuse" shape as Auth Foundation's TestContainers fixtures).
- Every chunk is written to BOTH `Chunks.embedding` (Postgres, via the `pgvector` SQLAlchemy `Vector(384)` type, for transactional consistency with the row) AND Qdrant's `documents` collection (the actual search path) — `Chunks.embedding` is written but never queried by application code.
- The Qdrant `documents` collection uses HNSW with `m=16`, `ef_construct=128` (docs/database/DATABASE.md's own specified values), and every point's payload carries `tenant_id` and `document_id` so a search can be scoped to one tenant before the nearest-neighbor search runs.
- `Chunks.parent_id` is always `NULL` and `Chunks.metadata` is always `{}` this sub-project — the columns exist per the documented schema, but Parent Document Retrieval (which would populate `parent_id`) is out of scope.
- `Documents` and `Chunks` both get `ENABLE`/`FORCE ROW LEVEL SECURITY` plus a `tenant_isolation` policy identical in shape to `Sessions`' (`current_setting('app.current_tenant_id', true)`), and `app_user` (the non-superuser role Auth Foundation's migration created) gets `GRANT SELECT, INSERT, UPDATE, DELETE` on both. This is the straightforward RLS case Auth Foundation's own ruling described: the app always knows the tenant (from the verified JWT) before it ever queries a user's own documents — the opposite of `Users`' login-time chicken-and-egg problem, so no `Users`-style carve-out is needed here.
- Uploaded files are written to local disk at `storage/{tenant_id}/{document_id}/{filename}` — no object storage (MinIO) this sub-project.
- Every one of the three new endpoints (`POST /documents`, `POST /documents/search`, `POST /chat`) requires a valid access token and is tenant-scoped via the existing `get_db_session`/`get_current_user_claims` dependencies — a request with no/invalid `Authorization` header gets the domain's `401`, never a raw framework error.
- Search and chat both use `top_k=5`.
- The chat model is `claude-opus-5` by default, overridable via a `CHAT_MODEL` environment variable.
- No raw or string-interpolated SQL — every `PostgresDocumentRepository` query uses SQLAlchemy's `text()` with bound `:name` parameters, matching `PostgresUserRepository`'s existing pattern exactly.
- Unit tests (`tests/unit/`) use fakes — no Docker, no network, no real model load. Integration tests (`tests/integration/`) use TestContainers-provisioned PostgreSQL, Redis, and (new) Qdrant, plus the real MiniLM model. `ClaudeChatModel` is the one adapter NOT integration-tested against the real API on every run — its prompt/response logic is unit-tested against a faked Anthropic client, and a single manual, non-pytest-collected smoke script (mirroring `tests/integration/test_docker_compose_smoke.py`) proves the real integration by hand.
- Coverage target is ≥80%, checked via `uv run pytest tests/ --cov=src --cov-report=term-missing` (the whole suite — the same corrected command Auth Foundation's final review established).
- Every commit follows the `.gitmessage` template and Conventional Commits format.

---

### Task 1: Dependencies, scaffold, and a tracking Story

**Files:**
- Modify: `pyproject.toml` (add `sentence-transformers`, `qdrant-client`, `pgvector`, `pypdf`, `tiktoken`, `anthropic`)
- Create: `src/rag/__init__.py`, `src/rag/domain/__init__.py`, `src/rag/application/__init__.py`, `src/rag/infrastructure/__init__.py`
- Modify: `.gitignore` (add `storage/`)
- Modify: `.env.example` (add `ANTHROPIC_API_KEY`, `CHAT_MODEL`)

**Interfaces:**
- Produces: an installable project with the new dependencies resolved, `ruff check src/ tests/` and `mypy src/` both clean on the (near-empty) new tree.

- [ ] **Step 1: Add the new dependencies to `pyproject.toml`**

Add to the `dependencies` list (alongside what Auth Foundation already added):

```toml
    "sentence-transformers>=3.0",
    "qdrant-client>=1.9",
    "pgvector>=0.3",
    "pypdf>=4.2",
    "tiktoken>=0.7",
    "anthropic>=0.40",
```

- [ ] **Step 2: Create the package skeleton**

Create `src/rag/__init__.py`, `src/rag/domain/__init__.py`, `src/rag/application/__init__.py`, `src/rag/infrastructure/__init__.py` as empty files.

- [ ] **Step 3: Add `storage/` to `.gitignore`**

Append to `.gitignore`:

```
# Local per-tenant document storage (RAG Pipeline sub-project)
storage/
```

- [ ] **Step 4: Add the new secrets to `.env.example`**

Append two lines:

```
ANTHROPIC_API_KEY=
CHAT_MODEL=claude-opus-5
```

- [ ] **Step 5: Install and verify**

Run: `uv sync --extra dev`
Expected: dependency resolution succeeds. Note: `sentence-transformers` pulls in `torch` as a transitive dependency, so this install is noticeably larger and slower than Auth Foundation's — this is expected, not a sign of a problem.

Run: `uv run ruff check src/ tests/`
Expected: no errors.

Run: `uv run mypy src/`
Expected: no errors (nothing new to type-check yet beyond empty files).

- [ ] **Step 6: File the tracking Story**

Before committing, run (this records the actual issue number the commit message and later tasks reference — do not guess a number):

```bash
gh issue create \
  --title "[STORY] RAG Pipeline" \
  --label "type:story,needs-triage,priority:p1" \
  --body "$(cat <<'EOF'
## User Story
As a developer building on this system,
I want document upload, chunking, embedding, vector search, and a chat endpoint answering from retrieved context,
so that Epic #144's Phase 1 slice is complete end to end: a user can register, log in, upload a document, and get an answer grounded in it.

## Parent Epic
Epic: #144

## Acceptance Criteria
- [ ] A user can upload a .txt or .pdf document; it's chunked (512 tokens, 10% overlap), embedded (MiniLM, 384-dim), and written to both Postgres and Qdrant
- [ ] PostgreSQL row-level security enforces tenant isolation on Documents and Chunks, verified the same way Auth Foundation verified it on Sessions (real seeded cross-tenant data, no application-level filter)
- [ ] Qdrant search is tenant-filtered at the payload level before the nearest-neighbor search runs
- [ ] POST /documents/search returns ranked chunks for a query
- [ ] POST /chat returns an answer plus its grounding sources from a real Claude API call
- [ ] Every new endpoint requires a valid access token
- [ ] Test coverage on src/ is >=80% (uv run pytest tests/ --cov=src)

## Linked Tasks
Tracked in the implementation plan (docs/superpowers/plans/) and its SDD ledger rather than as individual GitHub issues.

## Priority
P1

## Story Points
13

## Definition of Done
- [ ] All acceptance criteria met
- [ ] Linked PR(s) merged with required review approval
- [ ] CI checks passed (lint, test, build)
- [ ] Documentation updated where applicable

## Additional Context
Spec: docs/superpowers/specs/2026-08-22-rag-pipeline-design.md
EOF
)"
```

Note the returned issue number — substitute it for `<STORY_NUMBER>` in every later task's commit message and in Task 1's own commit below.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/rag/ .gitignore .env.example
git commit -m "feat: scaffold RAG Pipeline project structure and dependencies

Refs Story #<STORY_NUMBER>"
```

---

### Task 2: RAG domain layer

**Files:**
- Create: `src/rag/domain/entities.py`
- Create: `src/rag/domain/ports.py`
- Create: `src/rag/domain/errors.py`
- Test: `tests/unit/test_rag_domain.py`

**Interfaces:**
- Produces: `Document`, `Chunk`, `SearchResult`, `ChatAnswer` entities; `EmbeddingModel`, `VectorStore`, `ChatModel`, `DocumentRepository` port ABCs; `UnsupportedFileType` error. Every later task in this plan imports from here.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_rag_domain.py
import uuid
from datetime import datetime, timezone

from src.rag.domain.entities import Chunk, Document
from src.rag.domain.errors import UnsupportedFileType


def test_document_equality_is_by_all_fields():
    now = datetime.now(timezone.utc)
    shared_id = uuid.uuid4()
    tenant = uuid.uuid4()
    a = Document(
        id=shared_id, tenant_id=tenant, filename="a.txt", mime_type="text/plain",
        storage_path="storage/x/a.txt", chunk_count=0, status="processing",
    )
    b = Document(
        id=shared_id, tenant_id=tenant, filename="a.txt", mime_type="text/plain",
        storage_path="storage/x/a.txt", chunk_count=0, status="processing",
    )
    assert a == b


def test_chunk_defaults_parent_id_none_and_metadata_empty():
    chunk = Chunk(
        id=uuid.uuid4(), document_id=uuid.uuid4(), content="some text",
        embedding=[0.1, 0.2, 0.3],
    )
    assert chunk.parent_id is None
    assert chunk.metadata == {}


def test_unsupported_file_type_error_names_the_extension():
    err = UnsupportedFileType(".docx")
    assert ".docx" in str(err)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_rag_domain.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/domain/errors.py`**

```python
class UnsupportedFileType(Exception):
    def __init__(self, extension: str) -> None:
        super().__init__(f"Unsupported file type: {extension}")
        self.extension = extension
```

- [ ] **Step 4: Write `src/rag/domain/entities.py`**

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Document:
    id: uuid.UUID
    tenant_id: uuid.UUID
    filename: str
    mime_type: str
    storage_path: str
    chunk_count: int
    status: str  # "processing" | "completed" | "failed"
    created_at: datetime | None = None


@dataclass(frozen=True)
class Chunk:
    id: uuid.UUID
    document_id: uuid.UUID
    content: str
    embedding: list[float]
    parent_id: uuid.UUID | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    content: str
    score: float


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    sources: list[SearchResult]
```

- [ ] **Step 5: Write `src/rag/domain/ports.py`**

```python
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from src.rag.domain.entities import Chunk, Document, SearchResult


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...


class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, chunk: Chunk, tenant_id: uuid.UUID) -> None: ...

    @abstractmethod
    async def search(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[SearchResult]: ...


class ChatModel(ABC):
    @abstractmethod
    async def generate(self, question: str, context: str) -> str: ...


class DocumentRepository(ABC):
    @abstractmethod
    async def save_document(self, document: Document) -> None: ...

    @abstractmethod
    async def update_document_status(
        self, document_id: uuid.UUID, status: str, chunk_count: int
    ) -> None: ...

    @abstractmethod
    async def save_chunks(self, chunks: list[Chunk], tenant_id: uuid.UUID) -> None: ...
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_rag_domain.py -v`
Expected: `3 passed`.

- [ ] **Step 7: Confirm zero framework imports**

Run: `grep -rE "^(import|from) (fastapi|sqlalchemy|qdrant_client|sentence_transformers|anthropic)" src/rag/domain/`
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add src/rag/domain/ tests/unit/test_rag_domain.py
git commit -m "feat: add RAG domain entities, ports, and errors

Refs Story #<STORY_NUMBER>"
```

---

### Task 3: Fixed-size chunker

**Files:**
- Create: `src/rag/infrastructure/fixed_size_chunker.py`
- Test: `tests/unit/test_fixed_size_chunker.py`

**Interfaces:**
- Produces: `FixedSizeChunker.chunk(text: str) -> list[str]`, consumed by Task 11's `UploadDocument` use case.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_fixed_size_chunker.py
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker


def test_short_text_produces_a_single_chunk():
    chunker = FixedSizeChunker(chunk_size_tokens=512, overlap_ratio=0.1)
    chunks = chunker.chunk("This is a short piece of text.")
    assert len(chunks) == 1
    assert chunks[0] == "This is a short piece of text."


def test_long_text_produces_multiple_chunks_with_overlap():
    chunker = FixedSizeChunker(chunk_size_tokens=50, overlap_ratio=0.1)
    # ~500 words is comfortably more than 50 tokens worth of content.
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    # Overlap means the tail of one chunk should reappear near the head of the next.
    first_tail_words = chunks[0].split()[-3:]
    second_text = chunks[1]
    assert any(word in second_text for word in first_tail_words)


def test_empty_text_produces_no_chunks():
    chunker = FixedSizeChunker(chunk_size_tokens=512, overlap_ratio=0.1)
    assert chunker.chunk("") == []


def test_every_chunk_is_at_most_the_configured_token_size():
    chunker = FixedSizeChunker(chunk_size_tokens=20, overlap_ratio=0.1)
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunker.chunk(text)
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    for c in chunks:
        assert len(encoding.encode(c)) <= 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_fixed_size_chunker.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/infrastructure/fixed_size_chunker.py`**

```python
import tiktoken


class FixedSizeChunker:
    def __init__(self, chunk_size_tokens: int = 512, overlap_ratio: float = 0.1) -> None:
        self._chunk_size = chunk_size_tokens
        self._overlap = int(chunk_size_tokens * overlap_ratio)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        tokens = self._encoding.encode(text)
        if len(tokens) <= self._chunk_size:
            return [text]

        chunks: list[str] = []
        step = self._chunk_size - self._overlap
        start = 0
        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            chunks.append(self._encoding.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += step
        return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_fixed_size_chunker.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rag/infrastructure/fixed_size_chunker.py tests/unit/test_fixed_size_chunker.py
git commit -m "feat: add fixed-size chunker with token-based overlap

Refs Story #<STORY_NUMBER>"
```

---

### Task 4: Text extractor

**Files:**
- Create: `src/rag/infrastructure/text_extractor.py`
- Test: `tests/unit/test_text_extractor.py`

**Interfaces:**
- Consumes: `UnsupportedFileType` (Task 2).
- Produces: `TextExtractor.extract(filename: str, content: bytes) -> str`, consumed by Task 11's `UploadDocument` use case.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_text_extractor.py
import io

import pytest
from pypdf import PdfWriter

from src.rag.domain.errors import UnsupportedFileType
from src.rag.infrastructure.text_extractor import TextExtractor


def _make_minimal_pdf_bytes(text: str) -> bytes:
    # Build a tiny real PDF in memory rather than committing a binary fixture file.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # pypdf's PdfWriter has no direct "write text" helper; a blank-page PDF is
    # sufficient to prove extraction runs without error against a real PDF
    # structure. The extraction test below checks pypdf.PdfReader is invoked
    # correctly, not that specific text round-trips — that would require a
    # heavier PDF-generation dependency this project doesn't otherwise need.
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_txt_content_is_decoded_directly():
    extractor = TextExtractor()
    result = extractor.extract("notes.txt", "hello world".encode("utf-8"))
    assert result == "hello world"


def test_pdf_content_is_extracted_via_pypdf():
    extractor = TextExtractor()
    pdf_bytes = _make_minimal_pdf_bytes("irrelevant")
    # A blank page extracts to an empty string — this proves the pypdf path
    # runs without raising, which is what this test is actually checking.
    result = extractor.extract("report.pdf", pdf_bytes)
    assert result == ""


def test_unsupported_extension_raises():
    extractor = TextExtractor()
    with pytest.raises(UnsupportedFileType):
        extractor.extract("archive.docx", b"whatever")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_text_extractor.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/infrastructure/text_extractor.py`**

```python
import io

from pypdf import PdfReader

from src.rag.domain.errors import UnsupportedFileType


class TextExtractor:
    def extract(self, filename: str, content: bytes) -> str:
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if extension == ".txt":
            return content.decode("utf-8")

        if extension == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        raise UnsupportedFileType(extension or filename)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_text_extractor.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rag/infrastructure/text_extractor.py tests/unit/test_text_extractor.py
git commit -m "feat: add text extractor for .txt and .pdf uploads

Refs Story #<STORY_NUMBER>"
```

---

### Task 5: MiniLM embedder

**Files:**
- Create: `src/rag/infrastructure/sentence_transformers_embedder.py`
- Modify: `tests/integration/conftest.py` (add a session-scoped `embedding_model` fixture)
- Test: `tests/integration/test_sentence_transformers_embedder.py`

**Interfaces:**
- Consumes: `EmbeddingModel` port (Task 2).
- Produces: `SentenceTransformersEmbedder`, consumed by Task 11/12's use cases via `src/api/dependencies.py` (Task 14). Also produces the `embedding_model` fixture other integration tests in this plan reuse.

- [ ] **Step 1: Add the session-scoped model fixture to `tests/integration/conftest.py`**

The model load (downloading `all-MiniLM-L6-v2` on first use, then loading it into memory) is a real, one-time cost — load it once per test session, the same shape as the container fixtures. Append to the existing file:

```python
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder


@pytest.fixture(scope="session")
def embedding_model() -> SentenceTransformersEmbedder:
    return SentenceTransformersEmbedder()
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/integration/test_sentence_transformers_embedder.py
def test_embed_returns_a_384_dimensional_vector(embedding_model):
    result = embedding_model.embed("a sentence to embed")
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)


def test_similar_sentences_embed_closer_than_dissimilar_ones(embedding_model):
    import math

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b)

    base = embedding_model.embed("the cat sat on the mat")
    similar = embedding_model.embed("a cat was sitting on a mat")
    different = embedding_model.embed("quarterly financial earnings report")

    assert cosine_similarity(base, similar) > cosine_similarity(base, different)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_sentence_transformers_embedder.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Write `src/rag/infrastructure/sentence_transformers_embedder.py`**

```python
from sentence_transformers import SentenceTransformer

from src.rag.domain.ports import EmbeddingModel

_MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformersEmbedder(EmbeddingModel):
    def __init__(self) -> None:
        self._model = SentenceTransformer(_MODEL_NAME)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_sentence_transformers_embedder.py -v`
Expected: `2 passed`. First run downloads the model (needs network access) and will be noticeably slower than any prior test in this plan — subsequent runs reuse the local cache and are fast.

- [ ] **Step 6: Commit**

```bash
git add src/rag/infrastructure/sentence_transformers_embedder.py tests/integration/conftest.py tests/integration/test_sentence_transformers_embedder.py
git commit -m "feat: add MiniLM sentence embedder

Refs Story #<STORY_NUMBER>"
```

---

### Task 6: Qdrant vector store

**Files:**
- Modify: `docker/docker-compose.yml` (add a `qdrant` service)
- Modify: `tests/integration/conftest.py` (add `qdrant_container`/`qdrant_url` fixtures)
- Create: `src/rag/infrastructure/qdrant_vector_store.py`
- Test: `tests/integration/test_qdrant_vector_store.py`

**Interfaces:**
- Consumes: `VectorStore` port, `Chunk`/`SearchResult` entities (Task 2).
- Produces: `QdrantVectorStore`, consumed by Task 11/12's use cases via `src/api/dependencies.py` (Task 14).

- [ ] **Step 1: Add the `qdrant` service to `docker/docker-compose.yml`**

Add alongside the existing `postgres`/`redis`/`api` services:

```yaml
  qdrant:
    image: qdrant/qdrant:v1.9.0
    ports:
      - "6333:6333"
    healthcheck:
      test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 10
```

Add `qdrant` to the `api` service's `depends_on` block (`condition: service_healthy`, matching `postgres`/`redis`), and add a `QDRANT_URL: http://qdrant:6333` line to the `api` service's `environment` block.

- [ ] **Step 2: Add `qdrant_container`/`qdrant_url` fixtures to `tests/integration/conftest.py`**

```python
from testcontainers.qdrant import QdrantContainer


@pytest.fixture(scope="session")
def qdrant_container():
    with QdrantContainer() as container:
        yield container


@pytest.fixture(scope="session")
def qdrant_url(qdrant_container: QdrantContainer) -> str:
    return qdrant_container.get_client().rest_uri
```

(Confirmed via `testcontainers.qdrant.QdrantContainer` — the module exists, and `get_client()` returns a real `qdrant_client.QdrantClient`; `rest_uri` gives the URL the app's own `qdrant_client.QdrantClient(url=...)` construction needs.)

- [ ] **Step 3: Write the failing tests**

```python
# tests/integration/test_qdrant_vector_store.py
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
    chunk_a = Chunk(id=uuid.uuid4(), document_id=uuid.uuid4(), content="tenant a's content", embedding=[0.2] * 384)
    chunk_b = Chunk(id=uuid.uuid4(), document_id=uuid.uuid4(), content="tenant b's content", embedding=[0.2] * 384)
    await store.upsert(chunk_a, tenant_a)
    await store.upsert(chunk_b, tenant_b)

    results = await store.search(query_embedding=[0.2] * 384, tenant_id=tenant_a, top_k=10)
    chunk_ids = {r.chunk_id for r in results}
    assert chunk_a.id in chunk_ids
    assert chunk_b.id not in chunk_ids
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_qdrant_vector_store.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 5: Write `src/rag/infrastructure/qdrant_vector_store.py`**

```python
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from src.rag.domain.entities import Chunk, SearchResult
from src.rag.domain.ports import VectorStore

_COLLECTION_NAME = "documents"
_VECTOR_SIZE = 384


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str) -> None:
        self._client = AsyncQdrantClient(url=url)

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(_COLLECTION_NAME)
        if not exists:
            await self._client.create_collection(
                collection_name=_COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=_VECTOR_SIZE, distance=qmodels.Distance.COSINE
                ),
                hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=128),
            )

    async def upsert(self, chunk: Chunk, tenant_id: uuid.UUID) -> None:
        await self._client.upsert(
            collection_name=_COLLECTION_NAME,
            points=[
                qmodels.PointStruct(
                    id=str(chunk.id),
                    vector=chunk.embedding,
                    payload={
                        "tenant_id": str(tenant_id),
                        "document_id": str(chunk.document_id),
                        "content": chunk.content,
                    },
                )
            ],
        )

    async def search(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[SearchResult]:
        response = await self._client.query_points(
            collection_name=_COLLECTION_NAME,
            query=query_embedding,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value=str(tenant_id)))]
            ),
            limit=top_k,
        )
        return [
            SearchResult(
                document_id=uuid.UUID(point.payload["document_id"]),
                chunk_id=uuid.UUID(str(point.id)),
                content=point.payload["content"],
                score=point.score,
            )
            for point in response.points
        ]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_qdrant_vector_store.py -v`
Expected: `2 passed`.

- [ ] **Step 7: Commit**

```bash
git add docker/docker-compose.yml tests/integration/conftest.py src/rag/infrastructure/qdrant_vector_store.py tests/integration/test_qdrant_vector_store.py
git commit -m "feat: add Qdrant vector store with tenant-filtered search

Refs Story #<STORY_NUMBER>"
```

---

### Task 7: Documents/Chunks migration with row-level security

**Files:**
- Create: `alembic/versions/0002_documents_chunks.py`
- Test: `tests/integration/test_migration.py` (extend, not replace)

**Interfaces:**
- Produces: the `Documents`/`Chunks` tables with RLS enabled and enforced, `app_user` granted DML on both. Consumed by Task 8's repository and every later integration test.

- [ ] **Step 1: Write the failing tests**

Append to the existing `tests/integration/test_migration.py` (don't replace Task 5's own tests from Auth Foundation; that file already imports `text` from `sqlalchemy` for its own Sessions-table assertions, so no new import is needed for these two):

```python
async def test_documents_and_chunks_tables_exist_with_rls_enabled(db_session):
    result = await db_session.execute(
        text("SELECT relname, relrowsecurity FROM pg_class WHERE relname IN ('documents', 'chunks')")
    )
    rows = {row.relname: row.relrowsecurity for row in result}
    assert rows == {"documents": True, "chunks": True}


async def test_tenant_isolation_policy_exists_on_documents_and_chunks(db_session):
    result = await db_session.execute(
        text("SELECT tablename FROM pg_policies WHERE policyname = 'tenant_isolation'")
    )
    tables = {row.tablename for row in result}
    assert tables == {"sessions", "documents", "chunks"}
```

(The second test's expected set includes `sessions` because Auth Foundation's own migration already put the same policy there — this test is verifying the full, cumulative state of the database, not just what this task added.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_migration.py -v`
Expected: the two new tests fail (tables/policies don't exist yet); the pre-existing tests from Auth Foundation still pass.

- [ ] **Step 3: Write `alembic/versions/0002_documents_chunks.py`**

```python
"""documents and chunks with row-level security

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-22

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String, nullable=False),
        sa.Column("mime_type", sa.String, nullable=False),
        sa.Column("storage_path", sa.String, nullable=False),
        sa.Column("chunk_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String, nullable=False, server_default="processing"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chunks.id"), nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_tenant_id", "chunks", ["tenant_id"])

    for table in ("documents", "chunks"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
            USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
            """
        )

    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON documents, chunks TO app_user")


def downgrade() -> None:
    op.execute("REVOKE ALL ON documents, chunks FROM app_user")
    op.drop_table("chunks")
    op.drop_table("documents")
```

Note: `Chunks` carries its own `tenant_id` column here even though `DATABASE.md`'s original schema table only shows it as one of the three tables with an *explicit* `tenant_id` column (`Users`, `Sessions`, `Documents`) — `Chunks` was documented as scoping indirectly through its `document_id` foreign key. This migration adds it directly instead, because the RLS policy needs to evaluate `tenant_id` on the row being inserted/read without a join back to `Documents` on every single query — the same reasoning `DATABASE.md` itself flags as unresolved ("whether the intermediate tables also need their own tenant_id column for performance is a detail this document doesn't resolve on its own"). This plan resolves it: yes, for the RLS policy to work directly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_migration.py -v`
Expected: `5 passed` (2 new + 3 from Auth Foundation).

- [ ] **Step 5: Commit**

```bash
git add alembic/versions/0002_documents_chunks.py tests/integration/test_migration.py
git commit -m "feat: add Documents/Chunks migration with row-level security

Refs Story #<STORY_NUMBER>"
```

---

### Task 8: Postgres document repository

**Files:**
- Create: `src/rag/infrastructure/postgres_document_repository.py`
- Test: `tests/integration/test_postgres_document_repository.py`

**Interfaces:**
- Consumes: `DocumentRepository` port, `Document`/`Chunk` entities (Task 2), `db_session` fixture (Task 7's migration makes this fixture's database now include `documents`/`chunks`).
- Produces: `PostgresDocumentRepository(session)`, consumed by Task 11's `UploadDocument` use case via `src/api/dependencies.py` (Task 14).

- [ ] **Step 1: Write the failing tests**

```python
# tests/integration/test_postgres_document_repository.py
import uuid

from src.rag.domain.entities import Chunk, Document
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository


def _new_document(tenant_id: uuid.UUID) -> Document:
    return Document(
        id=uuid.uuid4(), tenant_id=tenant_id, filename="notes.txt", mime_type="text/plain",
        storage_path=f"storage/{tenant_id}/notes.txt", chunk_count=0, status="processing",
    )


async def test_save_document_then_update_status(db_session):
    from src.identity.infrastructure.db import set_tenant_context

    tenant_id = uuid.uuid4()
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresDocumentRepository(db_session)
    doc = _new_document(tenant_id)
    await repo.save_document(doc)
    await repo.update_document_status(doc.id, status="completed", chunk_count=3)
    await db_session.commit()

    from sqlalchemy import text

    result = await db_session.execute(
        text("SELECT status, chunk_count FROM documents WHERE id = :id"), {"id": doc.id}
    )
    row = result.mappings().first()
    assert row["status"] == "completed"
    assert row["chunk_count"] == 3


async def test_save_chunks_batch_inserts_all_of_them(db_session):
    from src.identity.infrastructure.db import set_tenant_context

    tenant_id = uuid.uuid4()
    await set_tenant_context(db_session, tenant_id)

    repo = PostgresDocumentRepository(db_session)
    doc = _new_document(tenant_id)
    await repo.save_document(doc)

    chunks = [
        Chunk(id=uuid.uuid4(), document_id=doc.id, content=f"chunk {i}", embedding=[0.0] * 384)
        for i in range(3)
    ]
    await repo.save_chunks(chunks, tenant_id=tenant_id)
    await db_session.commit()

    from sqlalchemy import text

    result = await db_session.execute(text("SELECT content FROM chunks WHERE document_id = :id"), {"id": doc.id})
    contents = {row.content for row in result}
    assert contents == {"chunk 0", "chunk 1", "chunk 2"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_postgres_document_repository.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/infrastructure/postgres_document_repository.py`**

```python
import json
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import DocumentRepository


class PostgresDocumentRepository(DocumentRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_document(self, document: Document) -> None:
        await self._session.execute(
            text(
                """
                INSERT INTO documents (id, tenant_id, filename, mime_type, storage_path, chunk_count, status)
                VALUES (:id, :tenant_id, :filename, :mime_type, :storage_path, :chunk_count, :status)
                """
            ),
            {
                "id": document.id,
                "tenant_id": document.tenant_id,
                "filename": document.filename,
                "mime_type": document.mime_type,
                "storage_path": document.storage_path,
                "chunk_count": document.chunk_count,
                "status": document.status,
            },
        )
        await self._session.flush()

    async def update_document_status(self, document_id: uuid.UUID, status: str, chunk_count: int) -> None:
        await self._session.execute(
            text("UPDATE documents SET status = :status, chunk_count = :chunk_count WHERE id = :id"),
            {"status": status, "chunk_count": chunk_count, "id": document_id},
        )
        await self._session.flush()

    async def save_chunks(self, chunks: list[Chunk], tenant_id: uuid.UUID) -> None:
        for chunk in chunks:
            await self._session.execute(
                text(
                    """
                    INSERT INTO chunks (id, document_id, content, embedding, parent_id, metadata, tenant_id)
                    VALUES (:id, :document_id, :content, :embedding, :parent_id, :metadata, :tenant_id)
                    """
                ),
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "embedding": str(chunk.embedding),
                    "parent_id": chunk.parent_id,
                    "metadata": json.dumps(chunk.metadata),
                    "tenant_id": tenant_id,
                },
            )
        await self._session.flush()
```

Note `save_chunks` takes an explicit `tenant_id` parameter, matching the `DocumentRepository.save_chunks` signature already defined in `src/rag/domain/ports.py` (Task 2). `Chunk` itself doesn't carry a `tenant_id` field (it's scoped through its parent `Document`), but the `chunks` table's own `tenant_id` column (added in Task 7 for the RLS policy) needs a value at insert time, and the repository is where that value gets attached — the same place `set_tenant_context` already establishes which tenant a session is scoped to.

`embedding` is passed as `str(chunk.embedding)` — asyncpg doesn't have a native Python-list-to-pgvector-literal codec registered by default, and pgvector's on-the-wire text format is exactly a bracketed comma-separated list, which is what `str()` on a `list[float]` already produces (`"[0.1, 0.2, ...]"`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_postgres_document_repository.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rag/infrastructure/postgres_document_repository.py tests/integration/test_postgres_document_repository.py
git commit -m "feat: add Postgres document repository

Refs Story #<STORY_NUMBER>"
```

---

### Task 9: Row-level tenant isolation on Documents/Chunks

**Files:**
- Test: `tests/integration/test_rag_rls_tenant_isolation.py`

**Interfaces:**
- Consumes: `db_session` fixture, `set_tenant_context` (Auth Foundation's `src/identity/infrastructure/db.py`), `PostgresDocumentRepository` (Task 8).
- Produces: the flagship proof this sub-project's RLS actually works — the same shape as Auth Foundation's `test_rls_tenant_isolation.py`, now on a second real table pair.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_rag_rls_tenant_isolation.py
import uuid

from sqlalchemy import text

from src.identity.infrastructure.db import set_tenant_context
from src.rag.domain.entities import Chunk, Document
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository


async def test_rls_returns_zero_cross_tenant_chunks_even_without_an_app_level_filter(db_session):
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    repo = PostgresDocumentRepository(db_session)

    await set_tenant_context(db_session, tenant_a)
    doc_a = Document(
        id=uuid.uuid4(), tenant_id=tenant_a, filename="a.txt", mime_type="text/plain",
        storage_path="storage/a/a.txt", chunk_count=1, status="completed",
    )
    await repo.save_document(doc_a)
    await repo.save_chunks(
        [Chunk(id=uuid.uuid4(), document_id=doc_a.id, content="tenant a's chunk", embedding=[0.0] * 384)],
        tenant_id=tenant_a,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_b)
    doc_b = Document(
        id=uuid.uuid4(), tenant_id=tenant_b, filename="b.txt", mime_type="text/plain",
        storage_path="storage/b/b.txt", chunk_count=1, status="completed",
    )
    await repo.save_document(doc_b)
    await repo.save_chunks(
        [Chunk(id=uuid.uuid4(), document_id=doc_b.id, content="tenant b's chunk", embedding=[0.0] * 384)],
        tenant_id=tenant_b,
    )
    await db_session.commit()

    await set_tenant_context(db_session, tenant_a)
    # Deliberately no WHERE tenant_id = ... — RLS alone must do the filtering.
    result = await db_session.execute(text("SELECT content FROM chunks"))
    contents = {row.content for row in result}

    assert contents == {"tenant a's chunk"}
    assert "tenant b's chunk" not in contents
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_rag_rls_tenant_isolation.py -v`
Expected: `1 passed`. No new production code is needed for this task — it proves Task 7's migration and Task 8's repository already compose correctly for tenant isolation on real data.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_rag_rls_tenant_isolation.py
git commit -m "test: verify row-level tenant isolation on Documents/Chunks

Refs Story #<STORY_NUMBER>"
```

---

### Task 10: Claude chat model

**Files:**
- Create: `src/rag/infrastructure/claude_chat_model.py`
- Test: `tests/unit/test_claude_chat_model.py`

**Interfaces:**
- Consumes: `ChatModel` port (Task 2).
- Produces: `ClaudeChatModel`, consumed by Task 13's `AnswerQuestion` use case via `src/api/dependencies.py` (Task 14).

- [ ] **Step 1: Write the failing tests**

Unit-tested against a faked Anthropic client — this task never calls the real API (that's Task 15's manual smoke script).

```python
# tests/unit/test_claude_chat_model.py
from src.rag.infrastructure.claude_chat_model import ClaudeChatModel


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [type("Block", (), {"text": text})()]


class _FakeMessages:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_call_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeMessage(self._response_text)


class _FakeAnthropicClient:
    def __init__(self, response_text: str) -> None:
        self.messages = _FakeMessages(response_text)


async def test_generate_returns_the_response_text():
    fake_client = _FakeAnthropicClient("The answer is 42.")
    model = ClaudeChatModel(client=fake_client, model_id="claude-opus-5")

    answer = await model.generate(question="What is the answer?", context="Some context.")

    assert answer == "The answer is 42."


async def test_generate_includes_both_question_and_context_in_the_request():
    fake_client = _FakeAnthropicClient("irrelevant")
    model = ClaudeChatModel(client=fake_client, model_id="claude-opus-5")

    await model.generate(question="What is FastAPI?", context="FastAPI is a Python web framework.")

    sent = fake_client.messages.last_call_kwargs
    assert sent["model"] == "claude-opus-5"
    full_prompt = str(sent["messages"])
    assert "What is FastAPI?" in full_prompt
    assert "FastAPI is a Python web framework." in full_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_claude_chat_model.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/infrastructure/claude_chat_model.py`**

```python
from src.rag.domain.ports import ChatModel

_SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. "
    "If the context doesn't contain the answer, say so plainly rather than guessing."
)


class ClaudeChatModel(ChatModel):
    def __init__(self, client, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    async def generate(self, question: str, context: str) -> str:
        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        return response.content[0].text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_claude_chat_model.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rag/infrastructure/claude_chat_model.py tests/unit/test_claude_chat_model.py
git commit -m "feat: add Claude chat model adapter

Refs Story #<STORY_NUMBER>"
```

---

### Task 11: Upload document use case

**Files:**
- Create: `src/rag/application/upload_document.py`
- Create: `tests/unit/rag_fakes.py` (fake in-memory ports, shared by Tasks 11-13)
- Test: `tests/unit/test_upload_document.py`

**Interfaces:**
- Consumes: every port from Task 2, `FixedSizeChunker` (Task 3), `TextExtractor` (Task 4).
- Produces: `UploadDocument`, consumed by Task 14's `POST /documents` route. `tests/unit/rag_fakes.py` also consumed by Tasks 12-13.

- [ ] **Step 1: Write `tests/unit/rag_fakes.py`**

```python
import uuid

from src.rag.domain.entities import Chunk, Document, SearchResult
from src.rag.domain.ports import ChatModel, DocumentRepository, EmbeddingModel, VectorStore


class FakeEmbeddingModel(EmbeddingModel):
    def embed(self, text: str) -> list[float]:
        # Deterministic, cheap stand-in: length-derived vector, not a real embedding.
        return [float(len(text) % 7)] * 384


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.upserted: list[tuple[Chunk, uuid.UUID]] = []
        self._search_results: list[SearchResult] = []

    async def upsert(self, chunk: Chunk, tenant_id: uuid.UUID) -> None:
        self.upserted.append((chunk, tenant_id))

    def set_search_results(self, results: list[SearchResult]) -> None:
        self._search_results = results

    async def search(self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int) -> list[SearchResult]:
        return self._search_results[:top_k]


class FakeChatModel(ChatModel):
    def __init__(self, response: str = "a fake answer") -> None:
        self._response = response
        self.last_question: str | None = None
        self.last_context: str | None = None

    async def generate(self, question: str, context: str) -> str:
        self.last_question = question
        self.last_context = context
        return self._response


class FakeDocumentRepository(DocumentRepository):
    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, Document] = {}
        self.chunks: list[Chunk] = []

    async def save_document(self, document: Document) -> None:
        self.documents[document.id] = document

    async def update_document_status(self, document_id: uuid.UUID, status: str, chunk_count: int) -> None:
        doc = self.documents[document_id]
        self.documents[document_id] = Document(
            id=doc.id, tenant_id=doc.tenant_id, filename=doc.filename, mime_type=doc.mime_type,
            storage_path=doc.storage_path, chunk_count=chunk_count, status=status,
        )

    async def save_chunks(self, chunks: list[Chunk], tenant_id: uuid.UUID) -> None:
        self.chunks.extend(chunks)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_upload_document.py
import uuid

from src.rag.application.upload_document import UploadDocument
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.text_extractor import TextExtractor
from tests.unit.rag_fakes import FakeDocumentRepository, FakeEmbeddingModel, FakeVectorStore


async def test_upload_chunks_embeds_and_dual_writes():
    repo = FakeDocumentRepository()
    vector_store = FakeVectorStore()
    use_case = UploadDocument(
        document_repository=repo,
        embedding_model=FakeEmbeddingModel(),
        vector_store=vector_store,
        chunker=FixedSizeChunker(chunk_size_tokens=10, overlap_ratio=0.1),
        extractor=TextExtractor(),
    )

    tenant_id = uuid.uuid4()
    text = " ".join(f"word{i}" for i in range(50))
    document = await use_case.execute(
        tenant_id=tenant_id, filename="notes.txt", content=text.encode("utf-8"), storage_path="storage/x/notes.txt"
    )

    assert document.status == "completed"
    assert document.chunk_count > 1
    assert len(repo.chunks) == document.chunk_count
    assert len(vector_store.upserted) == document.chunk_count
    assert all(tenant == tenant_id for _, tenant in vector_store.upserted)


async def test_upload_rejects_an_unsupported_file_type():
    import pytest

    from src.rag.domain.errors import UnsupportedFileType

    use_case = UploadDocument(
        document_repository=FakeDocumentRepository(),
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
        chunker=FixedSizeChunker(),
        extractor=TextExtractor(),
    )
    with pytest.raises(UnsupportedFileType):
        await use_case.execute(
            tenant_id=uuid.uuid4(), filename="archive.docx", content=b"x", storage_path="storage/x/archive.docx"
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_upload_document.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Write `src/rag/application/upload_document.py`**

```python
import uuid
from datetime import datetime, timezone

from src.rag.domain.entities import Chunk, Document
from src.rag.domain.ports import DocumentRepository, EmbeddingModel, VectorStore
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.text_extractor import TextExtractor


class UploadDocument:
    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunker: FixedSizeChunker,
        extractor: TextExtractor,
    ) -> None:
        self._documents = document_repository
        self._embedder = embedding_model
        self._vector_store = vector_store
        self._chunker = chunker
        self._extractor = extractor

    async def execute(
        self, tenant_id: uuid.UUID, filename: str, content: bytes, storage_path: str
    ) -> Document:
        text = self._extractor.extract(filename, content)  # raises UnsupportedFileType before anything is saved

        mime_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
        document = Document(
            id=uuid.uuid4(), tenant_id=tenant_id, filename=filename, mime_type=mime_type,
            storage_path=storage_path, chunk_count=0, status="processing",
        )
        await self._documents.save_document(document)

        chunk_texts = self._chunker.chunk(text)
        chunks: list[Chunk] = []
        for chunk_text in chunk_texts:
            embedding = self._embedder.embed(chunk_text)
            chunk = Chunk(id=uuid.uuid4(), document_id=document.id, content=chunk_text, embedding=embedding)
            chunks.append(chunk)
            await self._vector_store.upsert(chunk, tenant_id)

        await self._documents.save_chunks(chunks, tenant_id=tenant_id)
        await self._documents.update_document_status(document.id, status="completed", chunk_count=len(chunks))

        return Document(
            id=document.id, tenant_id=tenant_id, filename=filename, mime_type=mime_type,
            storage_path=storage_path, chunk_count=len(chunks), status="completed",
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_upload_document.py -v`
Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/rag/application/upload_document.py tests/unit/rag_fakes.py tests/unit/test_upload_document.py
git commit -m "feat: add upload document use case

Refs Story #<STORY_NUMBER>"
```

---

### Task 12: Search documents use case

**Files:**
- Create: `src/rag/application/search_documents.py`
- Test: `tests/unit/test_search_documents.py`

**Interfaces:**
- Consumes: `EmbeddingModel`, `VectorStore` ports (Task 2), `FakeEmbeddingModel`/`FakeVectorStore` (Task 11's `tests/unit/rag_fakes.py`).
- Produces: `SearchDocuments`, consumed by Task 13's `AnswerQuestion` and Task 14's `POST /documents/search` route.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_search_documents.py
import uuid

from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import SearchResult
from tests.unit.rag_fakes import FakeEmbeddingModel, FakeVectorStore


async def test_search_embeds_the_query_and_returns_vector_store_results():
    vector_store = FakeVectorStore()
    expected = [SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="a match", score=0.9)]
    vector_store.set_search_results(expected)

    use_case = SearchDocuments(embedding_model=FakeEmbeddingModel(), vector_store=vector_store)
    results = await use_case.execute(tenant_id=uuid.uuid4(), query="find this", top_k=5)

    assert results == expected


async def test_search_respects_top_k():
    vector_store = FakeVectorStore()
    vector_store.set_search_results(
        [SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content=f"match {i}", score=1.0 - i * 0.1) for i in range(10)]
    )

    use_case = SearchDocuments(embedding_model=FakeEmbeddingModel(), vector_store=vector_store)
    results = await use_case.execute(tenant_id=uuid.uuid4(), query="find this", top_k=3)

    assert len(results) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_search_documents.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/application/search_documents.py`**

```python
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import EmbeddingModel, VectorStore


class SearchDocuments:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore) -> None:
        self._embedder = embedding_model
        self._vector_store = vector_store

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        query_embedding = self._embedder.embed(query)
        return await self._vector_store.search(query_embedding, tenant_id=tenant_id, top_k=top_k)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_search_documents.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rag/application/search_documents.py tests/unit/test_search_documents.py
git commit -m "feat: add search documents use case

Refs Story #<STORY_NUMBER>"
```

---

### Task 13: Answer question use case

**Files:**
- Create: `src/rag/application/answer_question.py`
- Test: `tests/unit/test_answer_question.py`

**Interfaces:**
- Consumes: `SearchDocuments` (Task 12), `ChatModel` port (Task 2), `FakeChatModel` (Task 11's `tests/unit/rag_fakes.py`).
- Produces: `AnswerQuestion`, consumed by Task 14's `POST /chat` route.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_answer_question.py
import uuid

from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import SearchResult
from tests.unit.rag_fakes import FakeChatModel, FakeEmbeddingModel, FakeVectorStore


async def test_answer_question_grounds_the_answer_in_retrieved_sources():
    vector_store = FakeVectorStore()
    sources = [
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="FastAPI is a Python web framework.", score=0.95),
        SearchResult(document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content="It has automatic docs.", score=0.87),
    ]
    vector_store.set_search_results(sources)
    search = SearchDocuments(embedding_model=FakeEmbeddingModel(), vector_store=vector_store)
    chat_model = FakeChatModel(response="FastAPI is a web framework with automatic docs.")

    use_case = AnswerQuestion(search_documents=search, chat_model=chat_model, top_k=5)
    result = await use_case.execute(tenant_id=uuid.uuid4(), question="What is FastAPI?")

    assert result.answer == "FastAPI is a web framework with automatic docs."
    assert result.sources == sources
    assert "FastAPI is a Python web framework." in chat_model.last_context
    assert "It has automatic docs." in chat_model.last_context
    assert chat_model.last_question == "What is FastAPI?"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_answer_question.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/rag/application/answer_question.py`**

```python
import uuid

from src.rag.domain.entities import ChatAnswer
from src.rag.domain.ports import ChatModel
from src.rag.application.search_documents import SearchDocuments


class AnswerQuestion:
    def __init__(self, search_documents: SearchDocuments, chat_model: ChatModel, top_k: int) -> None:
        self._search = search_documents
        self._chat_model = chat_model
        self._top_k = top_k

    async def execute(self, tenant_id: uuid.UUID, question: str) -> ChatAnswer:
        sources = await self._search.execute(tenant_id=tenant_id, query=question, top_k=self._top_k)
        context = "\n\n".join(source.content for source in sources)
        answer = await self._chat_model.generate(question=question, context=context)
        return ChatAnswer(answer=answer, sources=sources)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_answer_question.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rag/application/answer_question.py tests/unit/test_answer_question.py
git commit -m "feat: add answer question use case

Refs Story #<STORY_NUMBER>"
```

---

### Task 14: API layer — dependencies, schemas, routers

**Files:**
- Modify: `src/api/dependencies.py` (add RAG service factories)
- Create: `src/api/schemas/documents.py`
- Create: `src/api/schemas/chat.py`
- Create: `src/api/routers/documents.py`
- Create: `src/api/routers/chat.py`
- Modify: `src/api/main.py` (mount both routers, call `ensure_collection()` at startup)
- Test: `tests/integration/test_documents_endpoints.py`
- Test: `tests/integration/test_chat_endpoint.py`

**Interfaces:**
- Consumes: every use case from Tasks 11-13, every infrastructure adapter from Tasks 3-6/10, `get_db_session`/`get_current_user_claims` (Auth Foundation's `src/api/dependencies.py`).
- Produces: `POST /documents`, `POST /documents/search`, `POST /chat` — the full slice this Story delivers.

- [ ] **Step 1: Add RAG service factories to `src/api/dependencies.py`**

Append to the existing file:

```python
import os
from pathlib import Path

from src.rag.infrastructure.claude_chat_model import ClaudeChatModel
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore
from src.rag.infrastructure.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.rag.infrastructure.text_extractor import TextExtractor

_embedding_model = SentenceTransformersEmbedder()
_vector_store = QdrantVectorStore(os.environ["QDRANT_URL"])


def get_embedding_model() -> SentenceTransformersEmbedder:
    return _embedding_model


def get_vector_store() -> QdrantVectorStore:
    return _vector_store


def get_chunker() -> FixedSizeChunker:
    return FixedSizeChunker()


def get_extractor() -> TextExtractor:
    return TextExtractor()


def get_chat_model() -> ClaudeChatModel:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return ClaudeChatModel(client=client, model_id=os.environ.get("CHAT_MODEL", "claude-opus-5"))
```

Note `_embedding_model` and `_vector_store` are module-level singletons, matching the existing `_engine`/`_sessionmaker` pattern in this same file — loading the MiniLM model and opening the Qdrant client connection happen once, at import time, not per request. `get_chat_model()` deliberately does NOT follow that pattern: constructing a fresh `AsyncAnthropic` client per call is correct here, since (unlike a DB connection pool or a loaded ML model) there's no expensive resource to reuse — the Anthropic SDK client is lightweight to construct.

- [ ] **Step 2: Write `src/api/schemas/documents.py`**

```python
import uuid

from pydantic import BaseModel


class UploadResponse(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    chunk_count: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResultSchema(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    content: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultSchema]
```

- [ ] **Step 3: Write `src/api/schemas/chat.py`**

```python
import uuid

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatSourceSchema(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSourceSchema]
```

- [ ] **Step 4: Write `src/api/routers/documents.py`**

```python
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_chunker,
    get_current_user_claims,
    get_db_session,
    get_embedding_model,
    get_extractor,
    get_vector_store,
)
from src.api.schemas.documents import SearchRequest, SearchResponse, SearchResultSchema, UploadResponse
from src.rag.application.search_documents import SearchDocuments
from src.rag.application.upload_document import UploadDocument
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=UploadResponse, status_code=201)
async def upload(
    file: UploadFile,
    claims: dict = Depends(get_current_user_claims),
    session: AsyncSession = Depends(get_db_session),
) -> UploadResponse:
    tenant_id = uuid.UUID(claims["tenant_id"])
    content = await file.read()

    storage_dir = Path("storage") / str(tenant_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / file.filename
    storage_path.write_bytes(content)

    use_case = UploadDocument(
        document_repository=PostgresDocumentRepository(session),
        embedding_model=get_embedding_model(),
        vector_store=get_vector_store(),
        chunker=get_chunker(),
        extractor=get_extractor(),
    )
    document = await use_case.execute(
        tenant_id=tenant_id, filename=file.filename, content=content, storage_path=str(storage_path)
    )
    await session.commit()

    return UploadResponse(
        id=document.id, filename=document.filename, status=document.status, chunk_count=document.chunk_count
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    claims: dict = Depends(get_current_user_claims),
) -> SearchResponse:
    tenant_id = uuid.UUID(claims["tenant_id"])
    use_case = SearchDocuments(embedding_model=get_embedding_model(), vector_store=get_vector_store())
    results = await use_case.execute(tenant_id=tenant_id, query=payload.query, top_k=payload.top_k)

    return SearchResponse(
        results=[
            SearchResultSchema(document_id=r.document_id, chunk_id=r.chunk_id, content=r.content, score=r.score)
            for r in results
        ]
    )
```

Note `/documents/search` depends only on `get_current_user_claims`, not `get_db_session` — search reads from Qdrant, not Postgres, so there's no database session to scope. `get_current_user_claims` alone is still what verifies the JWT and supplies `tenant_id` for Qdrant's own payload filter.

- [ ] **Step 5: Write `src/api/routers/chat.py`**

```python
import uuid

from fastapi import APIRouter, Depends

from src.api.dependencies import get_chat_model, get_current_user_claims, get_embedding_model, get_vector_store
from src.api.schemas.chat import ChatRequest, ChatResponse, ChatSourceSchema
from src.rag.application.answer_question import AnswerQuestion
from src.rag.application.search_documents import SearchDocuments

router = APIRouter(prefix="/chat", tags=["chat"])

_TOP_K = 5


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    claims: dict = Depends(get_current_user_claims),
) -> ChatResponse:
    tenant_id = uuid.UUID(claims["tenant_id"])
    search = SearchDocuments(embedding_model=get_embedding_model(), vector_store=get_vector_store())
    use_case = AnswerQuestion(search_documents=search, chat_model=get_chat_model(), top_k=_TOP_K)
    result = await use_case.execute(tenant_id=tenant_id, question=payload.question)

    return ChatResponse(
        answer=result.answer,
        sources=[
            ChatSourceSchema(document_id=s.document_id, chunk_id=s.chunk_id, content=s.content)
            for s in result.sources
        ],
    )
```

- [ ] **Step 6: Mount both routers and initialize the Qdrant collection in `src/api/main.py`**

Add the imports and mounting:

```python
from src.api.routers.chat import router as chat_router
from src.api.routers.documents import router as documents_router

app.include_router(documents_router)
app.include_router(chat_router)


@app.on_event("startup")
async def ensure_qdrant_collection() -> None:
    from src.api.dependencies import get_vector_store

    await get_vector_store().ensure_collection()
```

- [ ] **Step 7: Write the failing integration tests**

```python
# tests/integration/test_documents_endpoints.py
import os

from httpx import ASGITransport, AsyncClient


async def _client(app_database_url, redis_url, qdrant_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from src.api.main import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def _register_and_login(client) -> str:
    await client.post("/auth/register", json={"email": "rag@example.com", "password": "hunter2hunter2"})
    response = await client.post("/auth/login", json={"email": "rag@example.com", "password": "hunter2hunter2"})
    return response.json()["access_token"]


async def test_upload_then_search_finds_the_uploaded_content(app_database_url, redis_url, qdrant_url):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        token = await _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}

        upload_response = await client.post(
            "/documents",
            headers=headers,
            files={"file": ("notes.txt", b"FastAPI is a modern Python web framework.", "text/plain")},
        )
        assert upload_response.status_code == 201
        assert upload_response.json()["chunk_count"] >= 1

        search_response = await client.post(
            "/documents/search", headers=headers, json={"query": "Python web framework", "top_k": 5}
        )
        assert search_response.status_code == 200
        results = search_response.json()["results"]
        assert any("FastAPI" in r["content"] for r in results)


async def test_upload_without_a_token_returns_401(app_database_url, redis_url, qdrant_url):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        response = await client.post("/documents", files={"file": ("notes.txt", b"content", "text/plain")})
        assert response.status_code == 401


async def test_search_without_a_token_returns_401(app_database_url, redis_url, qdrant_url):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        response = await client.post("/documents/search", json={"query": "anything", "top_k": 5})
        assert response.status_code == 401


async def test_upload_rejects_an_unsupported_file_type(app_database_url, redis_url, qdrant_url):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        token = await _register_and_login(client)
        response = await client.post(
            "/documents",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("archive.docx", b"content", "application/octet-stream")},
        )
        assert response.status_code == 422
```

```python
# tests/integration/test_chat_endpoint.py
import os

from httpx import ASGITransport, AsyncClient


async def _client(app_database_url, redis_url, qdrant_url):
    os.environ["APP_DATABASE_URL"] = app_database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    from src.api.main import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_chat_without_a_token_returns_401(app_database_url, redis_url, qdrant_url):
    async with await _client(app_database_url, redis_url, qdrant_url) as client:
        response = await client.post("/chat", json={"question": "What is FastAPI?"})
        assert response.status_code == 401
```

Note `test_chat_endpoint.py` only covers the auth-gate here — a real end-to-end chat test (upload, then ask a question grounded in it) needs a real Claude API call, which this task's automated suite deliberately doesn't make. That full flow is Task 15's manual smoke script.

- [ ] **Step 8: Run tests, fix, verify green**

Run: `uv run pytest tests/integration/test_documents_endpoints.py tests/integration/test_chat_endpoint.py -v`
Expected final state: `5 passed`.

- [ ] **Step 9: Commit**

```bash
git add src/api/dependencies.py src/api/schemas/documents.py src/api/schemas/chat.py src/api/routers/documents.py src/api/routers/chat.py src/api/main.py tests/integration/test_documents_endpoints.py tests/integration/test_chat_endpoint.py
git commit -m "feat: add document upload, search, and chat endpoints

Refs Story #<STORY_NUMBER>"
```

---

### Task 15: Docker Compose full-stack verification and Claude smoke test

**Files:**
- Create: `tests/integration/test_rag_smoke.py`

No changes to `docker/Dockerfile.api` are expected — Step 1 verifies it still builds with the new dependencies. Note `torch`/`sentence-transformers` meaningfully increase image size and build time.

**Interfaces:**
- Consumes: the full `src/rag/` and updated `src/api/` from Tasks 1-14.
- Produces: end-to-end proof against the real four-service stack, including one real Claude API call — this task is where "does the whole thing actually work" gets answered for real, the same role Auth Foundation's Task 14 played.

- [ ] **Step 1: Bring the full stack up**

Run: `JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") APP_DB_PASSWORD=$(python3 -c "import secrets; print(secrets.token_hex(16))") ANTHROPIC_API_KEY=<a real key from .env> docker compose -f docker/docker-compose.yml up --build -d`
Expected: all four services (postgres, redis, qdrant, api) report healthy. The build is noticeably slower than Auth Foundation's own (downloading `torch` inside the image); this is expected.

- [ ] **Step 2: Write `tests/integration/test_rag_smoke.py`**

A documented manual script, following the exact same non-pytest-collected shape as `test_docker_compose_smoke.py`:

```python
# tests/integration/test_rag_smoke.py
"""Manual end-to-end smoke test for the real Docker Compose stack, including a
real Claude API call. Not part of `pytest tests/` (no `test_*` function is
defined at module level — see test_docker_compose_smoke.py's own docstring
for why). Run directly, after the stack is up (see docker/docker-compose.yml
and Task 15 of the rag-pipeline plan) and ANTHROPIC_API_KEY is a real key.

    uv run python tests/integration/test_rag_smoke.py

Expected: register/login succeed, a small .txt document uploads and is
searchable, and /chat returns a real Claude-generated answer that's actually
grounded in the uploaded content.
"""
import asyncio

import httpx


async def _run() -> None:
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        await client.post("/auth/register", json={"email": "smoke-rag@example.com", "password": "hunter2hunter2"})
        login = await client.post("/auth/login", json={"email": "smoke-rag@example.com", "password": "hunter2hunter2"})
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        upload = await client.post(
            "/documents",
            headers=headers,
            files={"file": ("facts.txt", b"The unified RAG x CAG x MAG project targets a 7900 XTX GPU for local serving.", "text/plain")},
        )
        print("upload:", upload.status_code, upload.json())
        assert upload.status_code == 201

        search = await client.post(
            "/documents/search", headers=headers, json={"query": "what GPU does this project target", "top_k": 5}
        )
        print("search:", search.status_code, search.json())
        assert search.status_code == 200

        chat = await client.post("/chat", headers=headers, json={"question": "What GPU does this project target?"})
        print("chat:", chat.status_code, chat.json())
        assert chat.status_code == 200
        assert "answer" in chat.json()


if __name__ == "__main__":
    asyncio.run(_run())
```

- [ ] **Step 3: Run the smoke script and record the real output**

Run: `uv run python tests/integration/test_rag_smoke.py`
Expected: register/login succeed, upload returns `201` with `chunk_count: 1`, search returns the uploaded chunk, and chat returns a real answer mentioning the 7900 XTX — paste the actual output into the implementer's report; this is the one genuinely end-to-end proof this whole plan produces.

- [ ] **Step 4: Tear down**

Run: `docker compose -f docker/docker-compose.yml down -v`

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_rag_smoke.py
git commit -m "test: add end-to-end Docker Compose smoke test including a real Claude call

Refs Story #<STORY_NUMBER>"
```

---

### Task 16: Documentation sync

**Files:**
- Modify: `docs/architecture/OVERVIEW.md` (Phase 1 module blueprint section)

**Interfaces:**
- None — brings documentation in line with what now exists, per CLAUDE.md's documentation-sync rule, the same role Auth Foundation's own Task 15 played.

- [ ] **Step 1: Update the Phase 1 blueprint's framing**

`docs/architecture/OVERVIEW.md`'s Phase 1 section currently says (after Auth Foundation's own update) that `src/rag/`, `src/cag/`, `src/mag/`, and `src/orchestration/` "remain pending." That's no longer true for `src/rag/`. Update the sentence to state plainly that `src/rag/` now exists (domain/application/infrastructure layers, per this plan), alongside `src/api/`'s new `documents`/`chat` routers, while `src/cag/`, `src/mag/`, and `src/orchestration/` remain genuinely unbuilt.

- [ ] **Step 2: Verify placeholder-scan and doc-map checks still pass**

Run: `grep -rnE '\bTBD\b|\bTODO\b|\bFIXME\b' CLAUDE.md README.md docs/ --include='*.md' | grep -vE 'docs/inputs/concepts/|docs/superpowers/'`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add docs/architecture/OVERVIEW.md
git commit -m "docs: update Phase 1 blueprint to reflect the RAG Pipeline code that now exists

Refs Story #<STORY_NUMBER>"
```

---

## After all tasks: finishing this branch

Once Task 16 is complete and the final whole-branch review (per `subagent-driven-development`) is clean:

1. Update the GitHub Story filed in Task 1 and Epic #144: check off acceptance criteria and Definition-of-Done items that are now genuinely true.
2. Since Epic #144's both Stories (Auth Foundation, RAG Pipeline) are now done, revisit Epic #144's own Definition of Done — "All linked Stories are Done," "Success criteria above are verifiably met" (a user can register, log in, upload, and get a grounded answer — Task 15's smoke test is the proof) — and close the Epic if both hold.
3. Run `/graphify` (or, absent `claude-md-sync`, a hand-authored pass against `docs/architecture/CONTEXT_GRAPH.md`'s own established convention, the same way Auth Foundation's second generation was produced) so `src/rag/` gets its own solid Module/Test File nodes alongside Identity & Access.
4. Use `superpowers:finishing-a-development-branch` — squash-merge into `develop` per Gitflow, never into `main` directly.
