# CLAUDE.md — Unified RAG × CAG × MAG AI System

> **Project:** Full-Stack Unified AI System  
> **Purpose:** Production-grade orchestration of Retrieval-Augmented Generation (RAG), Cache-Augmented Generation (CAG), and Memory-Augmented Generation (MAG) paradigms  
> **Primary Language:** Python 3.11+  
> **DL Framework:** PyTorch 2.3+ (CUDA 12.1+)  
> **Backend Architecture:** Hexagonal + CQRS  
> **Frontend Architecture:** React 19 Atomic Design  
> **Last Updated:** 2026-08-21

---

## 1. Project Identity

This system is NOT a simple chatbot. It is a **multi-paradigm AI orchestration platform** that routes queries across three distinct knowledge layers:

| Paradigm | Answers | Storage | Latency | Mutability |
|----------|---------|---------|---------|------------|
| **RAG** | "What exists OUTSIDE the model?" | Vector DB (Qdrant) + BM25 | 50-500ms | Instant sync |
| **CAG** | "What can I fit INSIDE the model's cache?" | GPU KV Cache (vLLM) | ~0ms TTFT | Batch invalidation |
| **MAG** | "What has the model LEARNED from this session?" | Redis + PostgreSQL + Neo4j | 1-10ms | Continuous writes |

**Core Insight:** No single paradigm is sufficient. The orchestration meta-layer decides which paradigm handles which piece of knowledge at which moment.

---

## 2. Architecture Overview

### 2.1 High-Level Layers
┌─────────────────────────────────────────────┐
│  LAYER 4: ORCHESTRATION (Meta-Layer)        │
│  • Paradigm Router (query classifier)       │
│  • Context Budget Allocator                 │
│  • Latency-Adaptive Fallback Cascade        │
│  • Sync Mixer (cross-paradigm consistency)  │
│  • Freshness-Aware Data Router              │
├─────────────────────────────────────────────┤
│  LAYER 3: MAG (State Layer)                 │
│  • Working Memory (context window)          │
│  • Episodic Memory (interaction logs)       │
│  • Semantic Memory (distilled facts)        │
│  • Procedural Memory (reusable workflows)   │
│  • Memory Graphs (Neo4j relationships)      │
│  • Consolidation & Evolution                │
├─────────────────────────────────────────────┤
│  LAYER 2: CAG (Cache Layer)                 │
│  • vLLM / SGLang serving                    │
│  • Prefix / Prompt Caching                  │
│  • PagedAttention block management          │
│  • KV Cache Eviction (H2O / SnapKV)         │
│  • KV Cache Compression (KIVI / KVQuant)    │
│  • Speculative Decoding                     │
│  • Cache-Aware Batching                     │
├─────────────────────────────────────────────┤
│  LAYER 1: RAG (Retrieval Layer)             │
│  • Document chunking (6 strategies)         │
│  • Embedding (sentence-transformers / BGE)  │
│  • Hybrid Search (vector + BM25)            │
│  • Multi-Query Retrieval                    │
│  • HyDE (Hypothetical Document Embeddings)  │
│  • Reranking (cross-encoder)                │
│  • Parent Document Retrieval                │
│  • Context Compression                      │
│  • Self-RAG (retrieval gate)                │
│  • CRAG (corrective validation)             │
├─────────────────────────────────────────────┤
│  FOUNDATION: LLM Core                       │
│  • vLLM serving (PyTorch-based)             │
│  • FlashAttention 2                         │
│  • Quantization (AWQ / GPTQ)                │
└─────────────────────────────────────────────┘
plain

### 2.2 Context Budget Allocation (128K default)

| Slice | Size | Content | Paradigm |
|-------|------|---------|----------|
| CAG Slice | 40% (51K) | Frozen pre-loaded docs, system prompt | CAG |
| MAG Slice | 25% (32K) | Session state, conversation history | MAG |
| RAG Slice | 20% (26K) | Dynamically retrieved chunks | RAG |
| Query Slice | 10% (13K) | Current user query + instructions | — |
| Reserve | 5% (6K) | Generation output buffer | — |

**Rule:** Slices are dynamic. If RAG is not needed, MAG expands. If MAG is empty, CAG expands.

### 2.3 Latency-Adaptive Fallback Cascade
User Query
│
▼
┌─────────────────┐  ← 10ms timeout
│  TIER 1: CAG    │     Pre-loaded static knowledge
│  Check cache    │
└─────────────────┘
│ Hit? ──→ Return immediately
│ Miss?
▼
┌─────────────────┐  ← 50ms timeout
│  TIER 2: MAG    │     Session state, user prefs
│  Check memory   │
└─────────────────┘
│ Hit? ──→ Return stateful answer
│ Miss / Insufficient?
▼
┌─────────────────┐  ← 2s timeout
│  TIER 3: RAG    │     External DB, live APIs
│  Retrieve       │
└─────────────────┘
│
▼
Return comprehensive answer
plain

---

## 3. Technology Stack (Non-Negotiable)

### 3.1 Core Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.11+ | Main backend |
| DL Framework | PyTorch | 2.3+ | Tensor ops, custom CUDA |
| LLM Serving | vLLM | 0.5+ | Production inference |
| API Framework | FastAPI | 0.111+ | REST + WebSocket |
| Validation | Pydantic v2 | — | Data models |
| Async Runtime | uvloop | — | Fast async event loop |
| Type Checking | mypy | latest | Static types |
| Lint/Format | ruff | latest | Linting + formatting |
| Package Manager | uv / poetry | latest | Dependencies |

### 3.2 Data Storage

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vector DB | Qdrant | Semantic search, HNSW indexing |
| Relational DB | PostgreSQL 16 + pgvector | Structured data + vectors |
| Cache / Session | Redis 7+ | Hot cache, pub/sub, sessions |
| Graph DB | Neo4j | Memory relationships, MAG graphs |
| Object Storage | MinIO | Document storage |
| Message Queue | RabbitMQ | Event-driven sync |
| Task Queue | Celery | Background jobs |
| Document Store | MongoDB (optional) | Unstructured logs |

### 3.3 RAG Ecosystem

| Component | Library | Purpose |
|-----------|---------|---------|
| Orchestration | LangChain >=0.2.0 / LlamaIndex >=0.10.0 | RAG pipelines |
| Agent Graphs | LangGraph >=0.1.0 | CRAG, Self-RAG workflows |
| Document Parsing | unstructured >=0.14.0 | PDF, MD, TXT parsing |
| Embeddings | sentence-transformers >=3.0 | Vector generation |
| Keyword Search | rank-bm25 >=0.2.2 | BM25 keyword search |
| Reranking | BAAI/bge-reranker-v2-m3 | Cross-encoder reranking |

### 3.4 CAG Ecosystem

| Component | Library | Purpose |
|-----------|---------|---------|
| Serving Engine | vLLM >=0.5.0 | PagedAttention, prefix caching |
| Attention Kernels | flash-attn >=2.5.0 | Optimized attention |
| Quantization | bitsandbytes, auto-gptq, optimum | Model compression |
| Custom CUDA | triton >=2.3.0 | Custom kernels |

### 3.5 MAG Ecosystem

| Component | Library | Purpose |
|-----------|---------|---------|
| Memory Framework | mem0ai >=1.0.0 | Memory layer for LLMs |
| PostgreSQL Client | psycopg >=3.1.0 | Structured storage |
| Vector Extension | pgvector >=0.2.0 | Postgres vectors |
| Graph DB | neo4j >=5.20.0 | Relationship memory |
| Async Redis | redis-py >=5.0.0 / aioredis >=2.0.0 | Hot cache |

### 3.6 Observability

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM Tracing | Langfuse | Trace LLM calls |
| Metrics | Prometheus | System metrics |
| Dashboards | Grafana | Visualization |
| Logging | structlog | Structured logging |
| APM | OpenTelemetry | Distributed tracing |

### 3.7 Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containers | Docker | Containerization |
| Orchestration | Kubernetes | Container orchestration |
| Packaging | Helm | K8s package management |
| Ingress | Traefik | Load balancing |
| TLS | cert-manager | TLS automation |
| Secrets | HashiCorp Vault | Secret management |
| IaC | Terraform | Infrastructure as code |

### 3.8 Frontend (React 19 Atomic Design)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React 18+ | Web UI |
| Styling | Tailwind CSS | Utility-first CSS |
| State | Zustand | State management |
| Streaming | SSE / WebSocket | Token streaming |

---

## 4. Development Rules (Non-Negotiable)

### 4.1 DATABASE-FIRST
- **Schema is source of truth.** Write Alembic migrations BEFORE application code.
- Test database mirrors production EXACTLY (same schema, same constraints, same indexes).
- All database access through repository pattern (Hexagonal Architecture).
- No raw SQL in handlers — use parameterized queries or ORM.
- RLS (Row-Level Security) enabled on PostgreSQL for multi-tenancy.

### 4.2 SPEC TDD (Red-Green-Refactor)
- Write `.spec.ts` / `.feature` / `test_*.py` BEFORE implementation.
- Every public function/class MUST have a corresponding test before merge.
- Unit tests: >=80% coverage.
- Integration tests: real DB, real Qdrant, real Redis (TestContainers).
- E2E tests: critical user paths only.

### 4.3 ARCHITECTURE: Hexagonal + CQRS Backend
- **Domain layer:** Pure business logic, no framework dependencies.
- **Application layer:** Use cases, orchestration.
- **Infrastructure layer:** DB, cache, HTTP, external APIs.
- **CQRS:** Separate read and write models for MAG memory operations.
- **SOLID/GRASP:** Single responsibility, dependency inversion, interface segregation.

### 4.4 ARCHITECTURE: React 19 Atomic Design Frontend
- **Atoms:** Buttons, inputs, labels (no business logic).
- **Molecules:** Search bars, form fields (simple combinations).
- **Organisms:** Chat interfaces, document uploaders (complex UI).
- **Templates:** Page layouts.
- **Pages:** Route-level components.
- All components typed with TypeScript strict mode.

### 4.5 SECURITY (20-Point Checklist)
1. API keys hidden in Vault / environment (never in code).
2. PostgreSQL RLS enabled for tenant isolation.
3. Parameterized queries ONLY (no string interpolation).
4. HTTPS everywhere (TLS 1.3).
5. Passwords hashed with Argon2id.
6. Rate limiting per user + per IP (Redis-backed).
7. Bot protection (reCAPTCHA v3 or hCaptcha).
8. Input validation at API boundary (Pydantic).
9. Output encoding for XSS prevention.
10. CORS properly configured (whitelist origins).
11. JWT with short expiry + refresh rotation.
12. Secrets rotation policy documented.
13. Dependency scanning (Snyk / Dependabot).
14. Container scanning (Trivy).
15. Network policies in K8s (deny-all default).
16. Audit logging for all auth events.
17. PII redaction in logs and memory storage.
18. Encryption at rest (PostgreSQL, Qdrant, MinIO).
19. Encryption in transit (mTLS between services).
20. DAST in CI pipeline (OWASP ZAP).

### 4.6 DOCUMENTATION (Auto-Generated)
Every PR must update or confirm the following docs:

| Document | Purpose | Auto-Gen Tool |
|----------|---------|---------------|
| `README.md` | Project overview, quickstart | Manual |
| `ARCHITECTURE.md` | C4 diagrams, component map | Manual + Mermaid |
| `API.md` | OpenAPI spec | FastAPI auto-gen |
| `TESTING.md` | Test strategy, coverage report | pytest-cov |
| `SECURITY.md` | Threat model, compliance | Manual |
| `DATABASE.md` | Schema docs, ER diagrams | tbls / ERAlchemy |
| `CHANGELOG.md` | Version history | git-cliff |
| `CONTEXT_GRAPH.md` | Domain→Context→Module→Class→Test | Mermaid (manual) |

### 4.7 CONTEXT GRAPH (Mermaid)
Maintain Mermaid diagrams mapping:
Domain → Bounded Context → Module → Class → Test File
plain
Update BEFORE any refactoring. The context graph lives in `docs/CONTEXT_GRAPH.md`.

### 4.8 GIT-NATIVE
- **Conventional Commits:** `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`.
- Every Epic, Task, ADR is a commit or referenced in commit messages.
- Git history = audit trail.
- Branch naming: `feat/123-short-desc`, `fix/456-bug-desc`, `docs/adr-007`.
- Squash merge to `main` with full commit message.

### 4.9 AI PROTOCOL (Development Workflow)
Load context → Check schema → Write spec → Implement → Test → Document → Commit
plain
**NEVER guess.** If schema or spec is missing, ASK. Do not hallucinate table structures or API contracts.

---

## 5. The Three Paradigms — Detailed

### 5.1 RAG (Retrieval-Augmented Generation)
**Purpose:** Access infinite external knowledge. Dynamic, live data.

**Pipeline Stages:**
1. **Pre-Processing:** Chunking (fixed, semantic, recursive, parent-document, sliding window, structure-aware)
2. **Pre-Retrieval:** Query expansion, HyDE, Multi-Query generation, Self-RAG gate
3. **Retrieval:** Hybrid Search (vector + BM25 via RRF)
4. **Post-Retrieval:** Reranking, Parent Document expansion, Context Compression, CRAG validation
5. **Generation:** LLM with curated context

**Key Combinations (A+ Synergy):**
- Hybrid Search + Reranking
- Multi Query + HyDE
- Reranking + CRAG
- Parent Document + Context Compression

### 5.2 CAG (Cache-Augmented Generation)
**Purpose:** Near-zero latency for static knowledge. Pre-baked / frozen context.

**Pipeline Stages:**
1. **Scheduling:** Cache-Aware Batching (group by shared prefix)
2. **Prefill:** Prefix Caching (skip redundant prefill)
3. **Storage:** PagedAttention blocks, KV Cache Compression (KIVI), Hybrid Offloading
4. **Decoding:** KV Cache Eviction (H2O / SnapKV), Speculative Decoding
5. **Architecture:** Standard Attention (NOT alternative attention — cache-based methods required)

**Key Combinations (A+ Synergy):**
- Prefix Caching + PagedAttention + Cache-Aware Batching
- Eviction + Compression + Hybrid Offloading
- Multi-Turn Caching + Eviction + Compression

**CRITICAL:** Alternative Attention (Linear, Mamba, etc.) conflicts with ALL cache-based methods. This project uses **standard attention + cache optimization** (Path A).

### 5.3 MAG (Memory-Augmented Generation)
**Purpose:** Stateful, personalized interactions. The agent's "soul."

**Memory Tiers:**
| Tier | Name | Scope | Speed | Storage |
|------|------|-------|-------|---------|
| 1 | Short-Term / Working | Current session | Fastest | LLM Context Window |
| 2 | Medium-Term / Recall | Recent interactions | Fast | Redis |
| 3 | Long-Term | All-time knowledge | Slower | PostgreSQL + Qdrant + Neo4j |

**Memory Types:**
- **Episodic:** Time-indexed experiences (interactions, tool calls, outcomes)
- **Semantic:** Distilled facts, user preferences, domain knowledge
- **Procedural:** Reusable workflows, successful patterns
- **Graph:** Relational memory (entities, relationships, temporal edges)

**Key Operations:**
- **Consolidation:** Transform episodic → semantic (Celery background job)
- **Evolution:** Update stale facts, handle contradictions, archive old data
- **Gating:** Token-budget filter before context injection
- **Retrieval:** Multi-strategy (semantic + temporal + causal + graph traversal + salience)

**Key Combinations (A+ Synergy):**
- Episodic + Semantic + Consolidation (The "Living Agent")
- Memory Hierarchy + Retrieval + Gating (The "Context Wizard")
- Episodic + Procedural + Consolidation (The "Self-Improving Agent")
- Memory Graphs + Episodic + Semantic (The "Relationship-Aware Agent")

---

## 6. Database Schema Philosophy

### 6.1 PostgreSQL (Primary Relational Store)
- **Users:** `id (UUID PK)`, `email`, `hashed_password`, `tenant_id`, `created_at`, `updated_at`
- **Sessions:** `id (UUID PK)`, `user_id (FK)`, `tenant_id`, `title`, `context_budget`, `created_at`
- **EpisodicMemory:** `id (UUID PK)`, `session_id (FK)`, `content (JSONB)`, `embedding (vector)`, `timestamp`, `salience_score`
- **SemanticMemory:** `id (UUID PK)`, `user_id (FK)`, `fact_key`, `fact_value`, `confidence`, `source`, `valid_until`, `embedding (vector)`
- **ProceduralMemory:** `id (UUID PK)`, `user_id (FK)`, `task_pattern`, `success_rate`, `last_used`, `workflow (JSONB)`
- **Documents:** `id (UUID PK)`, `tenant_id`, `filename`, `mime_type`, `storage_path`, `chunk_count`, `status`
- **Chunks:** `id (UUID PK)`, `document_id (FK)`, `content`, `embedding (vector)`, `parent_id`, `metadata (JSONB)`

### 6.2 Qdrant (Vector Store)
- Collection: `documents` — document chunks with payload metadata
- Collection: `semantic_memory` — distilled facts
- Collection: `episodic_memory` — experience embeddings
- HNSW index with `ef_construct=128`, `m=16`
- Metadata filtering enabled for tenant isolation

### 6.3 Redis (Hot Cache)
- Key pattern: `session:{session_id}:working_memory` — current turn context
- Key pattern: `user:{user_id}:preferences` — hot user prefs
- Key pattern: `cag:prefix:{hash}` — prefix cache metadata
- Pub/Sub: `memory:updates` — cross-service invalidation

### 6.4 Neo4j (Graph Memory)
- Nodes: `User`, `Session`, `Entity`, `Concept`, `Episode`, `Fact`
- Edges: `PARTICIPATED_IN`, `MENTIONS`, `RELATED_TO`, `ABSTRACTS_TO`, `TEMPORALLY_FOLLOWS`
- Index on `Entity.name` and `Entity.embedding`

---

## 7. API Design Conventions

### 7.1 REST Endpoints
- Base path: `/api/v1/`
- Resource naming: plural nouns (`/sessions`, `/documents`, `/memories`)
- Actions as sub-resources: `POST /sessions/{id}/messages`
- Pagination: cursor-based (`?cursor=xxx&limit=20`)
- Filtering: query params (`?status=active&sort=-created_at`)
- Content-Type: `application/json` or `text/event-stream` for streaming

### 7.2 WebSocket
- Path: `/ws/sessions/{session_id}`
- Protocol: JSON messages with `type`, `payload`, `timestamp`
- Heartbeat: 30s ping/pong
- Auth: JWT in query parameter (`?token=...`)

### 7.3 Response Format
```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "request_id": "uuid",
    "timestamp": "2026-08-21T01:57:00Z",
    "paradigms_used": ["CAG", "MAG"],
    "latency_ms": 45
  }
}
7.4 Error Format
JSON
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input",
    "details": { "field": "email", "issue": "required" }
  },
  "meta": { "request_id": "uuid" }
}
8. Testing Strategy
8.1 Test Pyramid
plain
        /\
       /  \  E2E (Playwright / Cypress)
      /____\  ~5% of tests
     /      \
    /________\  Integration (TestContainers)
   /          \  ~15% of tests
  /____________\  Unit (pytest)
 /              \  ~80% of tests
/________________\
8.2 Unit Tests
Every domain class/function has a test_*.py file.
Mock external dependencies (DB, LLM, vector store).
Target: >=80% line coverage.
Run: pytest tests/unit/ --cov=src --cov-report=term-missing
8.3 Integration Tests
Real PostgreSQL (TestContainers)
Real Qdrant (TestContainers)
Real Redis (TestContainers)
Test RAG pipeline end-to-end with sample documents.
Test MAG memory lifecycle (write → consolidate → retrieve).
Test CAG prefix caching with vLLM.
8.4 E2E Tests
Critical paths only:
User uploads document → asks question → gets RAG answer
User starts chat → multi-turn conversation → MAG remembers context
Repeated query on static docs → CAG cache hit → instant response
Paradigm router correctly routes query types
8.5 Performance Tests
Benchmark suite: scripts/benchmark.py
Metrics: TTFT (Time To First Token), throughput (tok/s), latency P95/P99
Load testing: k6 or Locust
Target: CAG cache hit <10ms, RAG retrieval <300ms, MAG retrieval <50ms
9. Security Model
9.1 Multi-Tenancy
Every table has tenant_id column.
PostgreSQL RLS policies enforce tenant_id filtering.
Qdrant payloads include tenant_id; filtered at query time.
Redis keys prefixed with tenant:{tenant_id}:.
9.2 Authentication
JWT access tokens (15min expiry).
Refresh tokens (7 days, stored in httpOnly cookie).
OAuth2 + OpenID Connect for enterprise SSO.
API keys for service-to-service (HashiCorp Vault rotation).
9.3 Data Protection
PII redaction before storing in MAG (use Presidio or regex patterns).
Encryption at rest: PostgreSQL (transparent data encryption), Qdrant (field-level), MinIO (server-side).
Encryption in transit: TLS 1.3 for all external, mTLS for internal K8s services.
Memory sanitization: Episodic memories auto-expire after 90 days unless consolidated.
9.4 Rate Limiting
Per-user: 100 req/min for chat, 10 req/min for document upload.
Per-IP: 1000 req/min (stricter for auth endpoints).
Redis-backed sliding window.
Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset.
10. Implementation Phases
Table
Phase	Weeks	Focus	Deliverables
1	1-2	Foundation	FastAPI scaffold, Docker, PostgreSQL+Qdrant+Redis, basic RAG
2	3-4	Advanced RAG	Hybrid search, reranking, HyDE, Self-RAG, CRAG, chunking strategies
3	5-6	CAG Infrastructure	vLLM deployment, prefix caching, PagedAttention, eviction, compression
4	7-8	MAG System	Session memory, episodic/semantic/procedural stores, consolidation worker
5	9-10	Orchestration	Paradigm router, budget allocator, latency cascade, sync mixer
6	11-12	Production	K8s, monitoring, auth, rate limiting, CI/CD, multi-tenancy
11. Key Architectural Decisions (ADRs)
ADR-001: vLLM over SGLang
Decision: Use vLLM as primary serving engine.
Rationale: vLLM has broader ecosystem, better documentation, and native prefix caching. SGLang acceptable for specific agent workflows.
Status: Accepted.
ADR-002: Qdrant over Milvus/Weaviate
Decision: Use Qdrant as vector database.
Rationale: Simpler ops, excellent Rust-based performance, strong hybrid search support, easier local development.
Status: Accepted.
ADR-003: Standard Attention + Cache Optimization (Path A)
Decision: Do NOT use alternative attention (Mamba, Linear Attention).
Rationale: This project requires prefix caching, PagedAttention, and KV cache manipulation. Alternative attention eliminates the KV cache entirely, making all CAG concepts incompatible.
Status: Accepted.
ADR-004: Hexagonal + CQRS for MAG
Decision: Separate read/write models for memory operations.
Rationale: MAG writes are frequent and async (episodic storage). Reads are complex and multi-strategy (semantic + graph + temporal). CQRS allows independent optimization.
Status: Accepted.
ADR-005: LangChain + LangGraph for RAG Orchestration
Decision: Use LangChain for RAG pipelines, LangGraph for agent workflows (CRAG, Self-RAG).
Rationale: Mature ecosystem, extensive integrations, graph-based agent loops map cleanly to CRAG/Self-RAG requirements.
Status: Accepted.
12. File Naming Conventions
Python
Modules: snake_case.py
Classes: PascalCase
Functions/variables: snake_case
Constants: SCREAMING_SNAKE_CASE
Private: _leading_underscore
Abstract base: Abstract*, Base*, I* (interface)
Tests: test_*.py or *_test.py
TypeScript / React
Components: PascalCase.tsx
Hooks: useCamelCase.ts
Utils: camelCase.ts
Styles: PascalCase.module.css or inline Tailwind
Tests: *.test.tsx / *.spec.tsx
Database
Tables: snake_case (plural)
Columns: snake_case
Indexes: idx_{table}_{column}
Constraints: fk_{table}_{ref_table}, uq_{table}_{column}
Migrations: YYYYMMDD_HHMMSS_description.py (Alembic)
13. Environment Variables (.env.example)
bash
# App
APP_ENV=development
APP_PORT=8000
APP_SECRET_KEY=change-me-in-production

# PostgreSQL
DATABASE_URL=postgresql+psycopg://user:pass@postgres:5432/ai_db
DATABASE_POOL_SIZE=20

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_POOL_SIZE=50

# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# vLLM
VLLM_URL=http://vllm:8000
VLLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
VLLM_MAX_MODEL_LEN=32768
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_QUANTIZATION=awq

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda

# Reranker
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Object Storage
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=documents

# Observability
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://langfuse:3000
PROMETHEUS_PORT=9090

# Security
JWT_SECRET=change-me
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7
14. Quick Commands
bash
# Setup
uv sync                          # Install dependencies
docker-compose -f docker/docker-compose.yml up -d  # Start infra

# Development
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run mypy src/                 # Type check
uv run pytest tests/ --cov=src   # Test

# Database
uv run alembic revision --autogenerate -m "description"
uv run alembic upgrade head

# Workers
uv run celery -A src.workers.celery_app worker -l info

# Serving
uv run vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-prefix-caching \
  --quantization awq

# API
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
15. Context Graph (Summary Mermaid)
Mermaid
Code
Preview
graph TD
    subgraph Domain["Domain: AI Orchestration"]
        subgraph RAG_Context["RAG Context"]
            R1[Chunking Module]
            R2[Embedding Module]
            R3[Retrieval Module]
            R4[Reranking Module]
            R5[Advanced: CRAG/Self-RAG]
        end
        
        subgraph CAG_Context["CAG Context"]
            C1[Cache Manager]
            C2[Eviction Policy]
            C3[Compression Engine]
            C4[Serving Client]
            C5[Offloader]
        end
        
        subgraph MAG_Context["MAG Context"]
            M1[Memory Stores]
            M2[Retrieval Strategies]
            M3[Consolidation Worker]
            M4[Evolution Engine]
            M5[Gating Controller]
        end
        
        subgraph Orchestration["Orchestration Context"]
            O1[Paradigm Router]
            O2[Budget Allocator]
            O3[Sync Mixer]
            O4[Latency Cascade]
        end
    end
    
    Client[React Frontend] --> O1
    O1 --> R3
    O1 --> C4
    O1 --> M2
    O1 --> O2
    O2 --> RAG_Context
    O2 --> CAG_Context
    O2 --> MAG_Context
    R3 --> Qdrant[(Qdrant)]
    C4 --> vLLM[(vLLM)]
    M1 --> Redis[(Redis)]
    M1 --> Postgres[(PostgreSQL)]
    M1 --> Neo4j[(Neo4j)]
This document is the single source of truth for AI assistants working on this codebase. When in doubt, check the schema, check the spec, then implement. Never guess.
plain

---

This `CLAUDE.md` synthesizes all five uploaded documents into a single, authoritative project context file. It captures:

- **The three paradigms** (RAG, CAG, MAG) with their specific concepts, synergies, and compatibility rules
- **The orchestration meta-layer** (router, budget allocator, sync mixer, latency cascade)
- **The full technology stack** from the architecture document
- **Your non-negotiable rules** (database-first, spec TDD, hexagonal/CQRS, security 20-point, documentation, context graphs, git-native, AI protocol)
- **Concrete schema guidance** for PostgreSQL, Qdrant, Redis, and Neo4j
- **API conventions, testing strategy, security model, and implementation phases**