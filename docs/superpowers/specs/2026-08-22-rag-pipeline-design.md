# RAG Pipeline — Design Spec

## Why this is Epic #144's second Story

`docs/superpowers/specs/2026-08-22-auth-foundation-design.md` scoped Epic #144's first half — a working, protected API with nothing behind it yet. This spec is the second half: `docs/architecture/OVERVIEW.md`'s own Phase 1 blueprint names it plainly — "a document upload endpoint, fixed-size chunking with MiniLM embeddings, a vector search endpoint, and a basic chat endpoint answering from RAG context." Everything here builds directly on Auth Foundation rather than beside it: every endpoint this sub-project adds is protected by the JWT verification Task 12 built, and every query this sub-project runs is scoped by the `get_db_session`/`set_tenant_context` wiring Auth Foundation shipped but left unconsumed — its own ledger named this as "the first thing the next sub-project must wire and test end-to-end before any real protected route ships." This is that route.

## Scope

In scope: uploading a `.txt` or `.pdf` document, extracting and chunking its text, embedding each chunk locally, storing chunks in both PostgreSQL (`Chunks.embedding`, for transactional consistency with the row) and Qdrant (the actual search path, tenant-filtered), a vector search endpoint, and a synchronous chat endpoint that retrieves the top-5 matching chunks and answers via the Claude API. Out of scope, named explicitly so the boundary doesn't drift: Hybrid Search and Reranking (`docs/architecture/RAG.md`'s own "beginner path" bundles these with chunking and claims ~70-80% of production quality from the combination — deliberately deferred to the next RAG iteration, matching this project's own precedent of narrow first slices); Parent Document Retrieval, Context Compression, Multi-Query, HyDE, Self-RAG, and CRAG (RAG.md's intermediate/advanced/expert phases — none of this sub-project's code forecloses adding them later, since `Chunks.parent_id` and `Chunks.metadata` exist in the schema unused, exactly as Auth Foundation left `get_db_session` unused until this sub-project needed it); response streaming (a synchronous JSON response matches this slice's "basic chat endpoint" framing, and no frontend exists yet to consume a stream); and MinIO object storage (`OVERVIEW.md`'s Phase 1 service list names PostgreSQL, Qdrant, and Redis — not MinIO — so the uploaded file itself is written to local disk under a per-tenant path, not to object storage that isn't part of this phase's own stack).

## Module layout

A new `src/rag/` paradigm module, following the identical hexagonal pattern `src/identity/` already established — the same domain/application/infrastructure split, the same port-interface discipline, the same test-pyramid shape:

```text
src/
├── rag/
│   ├── domain/
│   │   ├── entities.py      # Document, Chunk, SearchResult, ChatAnswer — no framework imports
│   │   ├── ports.py         # EmbeddingModel, VectorStore, ChatModel, DocumentRepository (all ABCs)
│   │   └── errors.py        # UnsupportedFileType
│   ├── application/
│   │   ├── upload_document.py    # extract -> chunk -> embed -> dual-write
│   │   ├── search_documents.py   # embed query -> Qdrant search, tenant-filtered
│   │   └── answer_question.py    # search -> build context -> one Claude call
│   └── infrastructure/
│       ├── fixed_size_chunker.py         # 512 tokens, 10% overlap
│       ├── text_extractor.py             # .txt passthrough, .pdf via pypdf
│       ├── sentence_transformers_embedder.py  # MiniLM, 384-dim
│       ├── qdrant_vector_store.py        # implements VectorStore
│       ├── postgres_document_repository.py    # implements DocumentRepository
│       └── claude_chat_model.py          # implements ChatModel
│
├── api/
│   ├── routers/
│   │   ├── documents.py     # POST /documents, POST /documents/search
│   │   └── chat.py          # POST /chat
│   └── schemas/
│       ├── documents.py     # UploadResponse, SearchRequest, SearchResponse
│       └── chat.py          # ChatRequest, ChatResponse

alembic/versions/
└── 0002_documents_chunks.py  # Documents, Chunks tables + RLS on both

docker/
└── docker-compose.yml        # + qdrant service, alongside the existing postgres/redis

storage/                      # local per-tenant file storage — gitignored, created at runtime
```

`src/rag/domain/` stays framework-free exactly as `src/identity/domain/` does — `EmbeddingModel`, `VectorStore`, and `ChatModel` are all ports the infrastructure layer implements, never imported by `application/` as concrete classes. `UploadDocument` is the one use case with real orchestration weight: it calls the chunker and extractor (both pure, framework-free infrastructure with no I/O), the embedder (a real ML model load), and two repositories (Postgres and Qdrant) in sequence — a heavier use case than anything in Auth Foundation, but still built from the same port interfaces a fake can stand in for during unit tests.

## Data model

Both tables come directly from `docs/database/DATABASE.md`'s schema table, unchanged:

| Table | Column | Notes |
|---|---|---|
| Documents | id | UUID, primary key |
| Documents | tenant_id | tenant-scoping root |
| Documents | filename | — |
| Documents | mime_type | — |
| Documents | storage_path | local disk path this sub-project actually writes to |
| Documents | chunk_count | set once chunking completes |
| Documents | status | `processing` \| `completed` \| `failed` |
| Chunks | id | UUID, primary key |
| Chunks | document_id | foreign key → Documents |
| Chunks | content | the chunk's text |
| Chunks | embedding | vector (pgvector) — written but not queried; Qdrant is the search path |
| Chunks | parent_id | self-reference; written as `NULL` this phase — Parent Document Retrieval is out of scope, not the schema |
| Chunks | metadata | JSONB; written as `{}` this phase, for the same reason |

Both tables get row-level security identical in shape to `Sessions`' — `ENABLE`/`FORCE ROW LEVEL SECURITY` plus a `tenant_isolation` policy using `current_setting('app.current_tenant_id', true)`. This is the case Auth Foundation's own hard-won ruling said RLS is *for*: `Documents`/`Chunks` are genuinely tenant-scoped data, and every request that touches them arrives already authenticated, with the tenant known from the verified JWT before any query runs — the opposite of `Users`' login-time chicken-and-egg problem. `app_user` (the non-superuser role Auth Foundation's migration created) gets `GRANT SELECT, INSERT, UPDATE, DELETE ON Documents, Chunks` in the same migration that enables the policy.

Qdrant's `documents` collection is exactly what `docs/database/DATABASE.md` already specifies: HNSW index, `m=16`, `ef_construct=128`, one point per chunk, payload carrying `tenant_id` and `document_id` so a search can be scoped to one tenant's payloads before the nearest-neighbor search runs — the mechanism `DATABASE.md` names as what "keeps the multi-tenant boundary intact at the vector layer the same way row-level security keeps it intact in PostgreSQL."

## Request flows

**Upload** (`POST /documents`, multipart form with a file): `get_current_user_claims` verifies the JWT, `get_db_session` sets tenant context from it. The handler reads the upload, extracts text (`.txt` read directly; `.pdf` via `pypdf`; anything else raises `UnsupportedFileType`, mapped to a `422`), writes the raw bytes to `storage/{tenant_id}/{document_id}/{filename}`, and calls `UploadDocument.execute()`. That use case creates the `Document` row (`status="processing"`), chunks the extracted text (512 tokens, 10% overlap — `docs/architecture/RAG.md`'s own suggested starting point), embeds each chunk via the local MiniLM model, writes chunks to Postgres and their vectors to Qdrant (payload includes `tenant_id` for the isolation `DATABASE.md` specifies), then updates the `Document` row to `status="completed"` with the real `chunk_count`. Returns `201` with the document's id, filename, and chunk count.

**Search** (`POST /documents/search`, `{"query": "...", "top_k": 5}`): same auth/tenant wiring. `SearchDocuments.execute()` embeds the query text with the same MiniLM model, then calls `VectorStore.search()`, which runs a Qdrant nearest-neighbor query filtered to the caller's `tenant_id` payload — no result from another tenant's documents is structurally reachable, matching the same "filter before the vector search runs" property RLS gives Postgres. Returns a list of `{document_id, chunk_id, content, score}`.

**Chat** (`POST /chat`, `{"question": "..."}`): same auth/tenant wiring. `AnswerQuestion.execute()` runs the same search internally (top-5), concatenates the retrieved chunks into a context block, and makes one call to `ChatModel.generate(question, context)` — `ClaudeChatModel`, using the `claude-opus-5` model id (this project's own `claude-api` skill default; overridable via a `CHAT_MODEL` environment variable, since the evaluation framework in `docs/evaluation/COMPARISON_METHODOLOGY.md` already anticipates comparing model choices here). Returns `{"answer": "...", "sources": [{"document_id": ..., "chunk_id": ..., "content": "..."}]}` — the source list is what lets a caller verify the answer is actually grounded in retrieved content, not just asserted.

## Testing strategy

Per `docs/testing/TESTING.md`'s pyramid, matching Auth Foundation's own precedent exactly: `tests/unit/` covers `FixedSizeChunker` and `TextExtractor` (pure logic, no I/O), the domain entities, and all three use cases against fakes (`FakeEmbeddingModel`, `FakeVectorStore`, `FakeChatModel`, `FakeDocumentRepository`) — fast, no Docker, no network. `tests/integration/` covers the real adapters: `QdrantVectorStore` against a real `testcontainers.qdrant.QdrantContainer` (confirmed to exist and expose a `get_client()` helper — the same fixture shape as the existing `postgres_container`/`redis_container`), `PostgresDocumentRepository` against the existing Postgres fixture extended with the new migration, and `SentenceTransformersEmbedder` against the real MiniLM model loaded once via a session-scoped fixture (mirroring the container fixtures' own scope, since loading a real embedding model is a real, if one-time, cost worth paying once per test session rather than per test).

`ClaudeChatModel` is the one adapter this plan doesn't integration-test the same way: calling the real Claude API costs real money and real tokens on every test run, which is a different category of cost than a free, local TestContainers instance. Its prompt-construction and response-parsing logic gets a unit test against a faked Anthropic client (proving the request shape and the answer/sources extraction are correct), and a single manual, non-pytest-collected smoke script — the exact same shape as Auth Foundation's own `test_docker_compose_smoke.py` — hits the real Claude API once to prove the integration genuinely works end to end, run by hand rather than on every CI pass.

## Security-control traceability

| Control | Verified by |
|---|---|
| Row-level tenant isolation on Documents/Chunks | Integration test seeding two tenants' documents, querying with one tenant's context set and no application-level filter, asserting zero cross-tenant rows — the same shape as Auth Foundation's `test_rls_tenant_isolation.py`, now proven on a second table |
| Qdrant search never returns another tenant's chunks | Integration test: seed chunks for two tenants into the same Qdrant collection, search as tenant A, assert zero tenant-B results in the response |
| Every endpoint requires a valid JWT | Integration test: each of the three new endpoints, called with no `Authorization` header, returns the domain's `401`, not a raw framework error |
| Uploaded file type is validated | Unit test: a file extension outside `.txt`/`.pdf` raises `UnsupportedFileType`, mapped to `422` |
| No raw SQL | Every `PostgresDocumentRepository` query uses SQLAlchemy's `text()` with bound parameters, matching `PostgresUserRepository`'s existing pattern exactly |

## Explicit non-goals

No Hybrid Search, Reranking, Parent Document Retrieval, Context Compression, Multi-Query, HyDE, Self-RAG, or CRAG — all nine are RAG.md's own named techniques, all explicitly deferred to a later RAG iteration, not foreclosed by anything this schema or module layout does. No response streaming. No MinIO or other object storage — local disk only. No support for file types beyond `.txt`/`.pdf`. No frontend — `OVERVIEW.md`'s Phase 1 tree stays backend-only through this sub-project too.
