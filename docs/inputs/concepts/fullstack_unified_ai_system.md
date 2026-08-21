# Full-Stack Architecture for Unified RAG × CAG × MAG System

> **Goal:** Production-grade Git repository implementing all three paradigms (RAG, CAG, MAG) with unified orchestration  
> **Primary Language:** Python 3.11+  
> **Deep Learning Framework:** PyTorch 2.3+ (with CUDA 12.1+)

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Core Technology Stack](#2-core-technology-stack)
3. [Per-Paradigm Technology Breakdown](#3-per-paradigm-technology-breakdown)
4. [Project Repository Structure](#4-project-repository-structure)
5. [Implementation Phases](#5-implementation-phases)
6. [Infrastructure & DevOps](#6-infrastructure--devops)
7. [Hardware Requirements](#7-hardware-requirements)
8. [Alternative Technology Choices](#8-alternative-technology-choices)

---

## 1. Architecture Overview

### Is PyTorch a Good Option?

**Yes — PyTorch is essential, but not sufficient alone.**

```
PyTorch = The engine (tensor ops, model weights, custom CUDA kernels)
Serving Framework = The car (vLLM, TGI, SGLang — all built ON PyTorch)
Your Code = The driver (orchestration logic, business rules)
```

**What PyTorch gives you:**
- Custom KV cache manipulation (eviction, compression algorithms)
- Fine-tuning embedding models and rerankers
- Building custom attention mechanisms
- Research flexibility

**What PyTorch alone CANNOT give you:**
- Production-grade batching (you need vLLM/SGLang)
- PagedAttention (memory management)
- Prefix caching at serving scale
- Tensor parallelism across GPUs

**Verdict:** Use PyTorch as the foundation, but build on top of **vLLM** or **SGLang** for serving. Use PyTorch directly only for custom research components.

---

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Web UI      │  │ Mobile App  │  │ API Clients │  │ SDK / CLI           │ │
│  │ (React/Vue) │  │ (React      │  │ (HTTP/gRPC) │  │ (Python/TS)         │ │
│  │             │  │  Native)    │  │             │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         └─────────────────┴─────────────────┴────────────────────┘            │
│                                    │                                         │
│                              ┌─────┴─────┐                                   │
│                              │  Load     │                                   │
│                              │ Balancer  │                                   │
│                              │ (Nginx/   │                                   │
│                              │  Traefik) │                                   │
│                              └─────┬─────┘                                   │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────┐
│                         API GATEWAY LAYER                                    │
│                              ┌─────┴─────┐                                   │
│                              │  FastAPI  │  ← Main orchestration API        │
│                              │  (Python) │                                   │
│                              └─────┬─────┘                                   │
│         ┌──────────────────────────┼──────────────────────────┐              │
│         │                          │                          │              │
│    ┌────┴────┐              ┌─────┴─────┐              ┌──────┴──────┐      │
│    │  Auth   │              │  Rate     │              │  Request    │      │
│    │ (JWT/   │              │  Limiter  │              │  Router     │      │
│    │  OAuth) │              │ (Redis)   │              │  (Paradigm  │      │
│    └─────────┘              └───────────┘              │  Classifier)│      │
│                                                        └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                                     │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐  │
│  │         Context Budget Allocator │  Sync Mixer                        │  │
│  │  ┌─────────┬─────────┬─────────┐│  ┌──────────────────────────────┐  │  │
│  │  │ CAG 40% │ MAG 25% │ RAG 20% ││  │ • Invalidate cascade         │  │  │
│  │  │ Query   │ Reserve │         ││  │ • Event deduplication        │  │  │
│  │  │  10%    │   5%    │         ││  │ • Consistency window         │  │  │
│  │  └─────────┴─────────┴─────────┘│  └──────────────────────────────┘  │  │
│  └─────────────────────────────────┼─────────────────────────────────────┘  │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────┐
│                         PARADIGM LAYERS                                      │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │      CAG         │  │      MAG         │  │         RAG              │   │
│  │  (Cache Layer)   │  │ (State Layer)    │  │    (Retrieval Layer)     │   │
│  │                  │  │                  │  │                          │   │
│  │ • vLLM / SGLang  │  │ • Redis (hot)    │  │ • Qdrant / Milvus        │   │
│  │ • Prefix Caching │  │ • PostgreSQL     │  │ • sentence-transformers  │   │
│  │ • PagedAttention │  │   (warm)         │  │ • rank-bm25              │   │
│  │ • Speculative    │  │ • Neo4j (graphs) │  │ • BGE reranker           │   │
│  │   Decoding       │  │ • Mem0 / Zep     │  │ • LangChain / LlamaIndex │   │
│  │                  │  │                  │  │ • LangGraph (agents)     │   │
│  └────────┬─────────┘  └────────┬─────────┘  └────────────┬─────────────┘   │
│           │                     │                         │                 │
│           └─────────────────────┴─────────────────────────┘                 │
│                                     │                                       │
│                              ┌──────┴──────┐                                │
│                              │   LLM Core  │                                │
│                              │  (vLLM /    │                                │
│                              │  Transformers│                               │
│                              │  + PyTorch) │                                │
│                              └─────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────┐
│                      DATA & STORAGE LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Qdrant /    │  │  PostgreSQL  │  │    Redis     │  │   Object Store  │  │
│  │  Milvus      │  │  + pgvector  │  │   (Cache +   │  │   (MinIO / S3)  │  │
│  │  (Vectors)   │  │  (Relational │  │   Session)   │  │   (Documents)   │  │
│  │              │  │   + Vectors) │  │              │  │                 │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │    Neo4j     │  │   Kafka /    │  │  Prometheus  │  │   Grafana /     │  │
│  │  (Graph DB)  │  │   RabbitMQ   │  │  + Grafana   │  │   Langfuse      │  │
│  │              │  │  (Events)    │  │  (Metrics)   │  │   (Observability)│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Technology Stack

### 2.1 Programming Language & Runtime

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Primary Language** | Python | 3.11+ | Main backend language |
| **Type Checking** | mypy | latest | Static type safety |
| **Linting** | ruff | latest | Fast Python linter |
| **Formatting** | ruff / black | latest | Code formatting |
| **Task Runner** | just / Makefile | — | Command automation |
| **Package Manager** | uv / poetry | latest | Dependency management |

### 2.2 Deep Learning & LLM Framework

| Component | Technology | Alternative | Purpose |
|-----------|-----------|-------------|---------|
| **DL Framework** | **PyTorch** 2.3+ | JAX (advanced) | Tensor ops, custom CUDA |
| **CUDA Toolkit** | CUDA 12.1+ | ROCm (AMD) | GPU acceleration |
| **LLM Serving** | **vLLM** 0.5+ | SGLang, TGI | Production inference |
| **Transformers** | Hugging Face Transformers | — | Model loading, tokenization |
| **Quantization** | bitsandbytes, AutoGPTQ, AWQ | TensorRT-LLM | Model compression |
| **Attention** | FlashAttention 2 | xFormers | Fast attention kernels |
| **Distributed** | PyTorch Distributed | Ray | Multi-GPU training |

### 2.3 Web Framework & API

| Component | Technology | Alternative | Purpose |
|-----------|-----------|-------------|---------|
| **API Framework** | **FastAPI** | Litestar, Django Ninja | REST API + WebSocket |
| **Async Runtime** | uvloop | — | Fast async event loop |
| **Validation** | Pydantic v2 | — | Data validation |
| **Auth** | FastAPI-Users, Auth0 | Keycloak | Authentication |
| **Documentation** | OpenAPI / Swagger | — | Auto API docs |
| **gRPC** | grpcio | — | Internal service comms |

### 2.4 Data Storage

| Component | Technology | Alternative | Purpose |
|-----------|-----------|-------------|---------|
| **Vector DB** | **Qdrant** | Milvus, Weaviate, Chroma | Semantic search |
| **Relational DB** | **PostgreSQL** 16+ | — | Structured data, pgvector |
| **Cache / Session** | **Redis** 7+ | KeyDB | Hot cache, sessions, pub/sub |
| **Graph DB** | **Neo4j** | Memgraph | Memory relationships |
| **Object Storage** | **MinIO** | AWS S3, Ceph | Document storage |
| **Document Store** | MongoDB | — | Unstructured logs |

### 2.5 Message Queue & Streaming

| Component | Technology | Alternative | Purpose |
|-----------|-----------|-------------|---------|
| **Message Broker** | **RabbitMQ** | Apache Kafka | Event-driven sync |
| **Task Queue** | **Celery** | RQ, Dramatiq | Background jobs |
| **Streaming** | Kafka | Redis Streams | Real-time data pipeline |

### 2.6 Observability

| Component | Technology | Alternative | Purpose |
|-----------|-----------|-------------|---------|
| **LLM Observability** | **Langfuse** | LangSmith, Phoenix | Trace LLM calls |
| **Metrics** | **Prometheus** | InfluxDB | System metrics |
| **Dashboards** | **Grafana** | — | Visualization |
| **Logging** | structlog | loguru | Structured logging |
| **APM** | OpenTelemetry | Datadog | Distributed tracing |

### 2.7 Infrastructure

| Component | Technology | Alternative | Purpose |
|-----------|-----------|-------------|---------|
| **Container** | **Docker** | Podman | Containerization |
| **Orchestration** | **Kubernetes** | Docker Swarm | Container orchestration |
| **Helm Charts** | Helm | Kustomize | K8s package management |
| **Ingress** | **Traefik** | Nginx Ingress | Load balancing |
| **Cert Manager** | cert-manager | — | TLS automation |
| **Secrets** | HashiCorp Vault | AWS Secrets Manager | Secret management |
| **IaC** | Terraform | Pulumi | Infrastructure as code |

### 2.8 Frontend (Optional)

| Component | Technology | Alternative | Purpose |
|-----------|-----------|-------------|---------|
| **Framework** | **React** 18+ | Vue, Svelte | Web UI |
| **Styling** | Tailwind CSS | — | Utility-first CSS |
| **State** | Zustand | Redux | State management |
| **Streaming** | SSE / WebSocket | — | Token streaming |
| **Chat UI** | Custom | Chainlit, Streamlit | Quick demo UIs |

---

## 3. Per-Paradigm Technology Breakdown

### 3.1 RAG Stack (Retrieval-Augmented Generation)

```
RAG Pipeline:
  Document Ingestion
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Chunking: LangChain / LlamaIndex / unstructured           │
  │  • Fixed-size, semantic, recursive, markdown-aware         │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Embedding: sentence-transformers / BGE / OpenAI           │
  │  • all-MiniLM-L6-v2 (fast), BAAI/bge-large-en (quality)    │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Vector Store: Qdrant / Milvus                             │
  │  • HNSW indexing, metadata filtering, hybrid search        │
  └─────────────────────────────────────────────────────────────┘
    ↓
  Query Time:
  ┌─────────────────────────────────────────────────────────────┐
  │  Query Processing:                                         │
  │  • Query Expansion: LLM-generated variants                 │
  │  • HyDE: hypothetical document generation                  │
  │  • Multi-Query: diverse phrasing generation                │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Hybrid Search:                                            │
  │  • Vector: Qdrant semantic search                          │
  │  • Keyword: rank-bm25 / Elasticsearch                      │
  │  • Fusion: RRF (Reciprocal Rank Fusion)                    │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Reranking:                                                │
  │  • Cross-Encoder: BAAI/bge-reranker-v2-m3                  │
  │  • LLM-based: GPT-4 / local LLM scoring                    │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Post-Processing:                                          │
  │  • Parent Document Retrieval: map chunks → parents         │
  │  • Context Compression: LLM summarization / filtering      │
  │  • CRAG: quality validation loop                           │
  │  • Self-RAG: retrieval decision gate                       │
  └─────────────────────────────────────────────────────────────┘
    ↓
  LLM Generation
```

**Key Libraries:**
```python
# Core RAG
langchain>=0.2.0           # Orchestration framework
llama-index>=0.10.0        # Alternative RAG framework
unstructured>=0.14.0       # Document parsing
sentence-transformers>=3.0 # Embeddings
qdrant-client>=1.9.0       # Vector DB client
rank-bm25>=0.2.2           # BM25 keyword search

# Advanced RAG
langgraph>=0.1.0           # Agent graphs (CRAG, Self-RAG)
langchain-huggingface>=0.0 # HF integration
llama-index-embeddings-huggingface
llama-index-vector-stores-qdrant
```

---

### 3.2 CAG Stack (Cache-Augmented Generation)

```
CAG Pipeline:
  Document Store (all static docs)
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Pre-Processing:                                           │
  │  • Load documents into model context                       │
  │  • Compute KV cache for entire corpus                      │
  │  • Store frozen KV cache in GPU VRAM                       │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Serving Infrastructure:                                   │
  │  • vLLM: PagedAttention, Prefix Caching, Continuous Batch  │
  │  • Tensor parallelism across GPUs                          │
  │  • Quantization: AWQ / GPTQ for model compression          │
  └─────────────────────────────────────────────────────────────┘
    ↓
  Query Time:
  ┌─────────────────────────────────────────────────────────────┐
  │  Cache Management:                                         │
  │  • Prefix Cache Hit: skip prefill (~0ms TTFT)              │
  │  • KV Cache Eviction: H2O / SnapKV (dynamic)               │
  │  • KV Cache Compression: KIVI / KVQuant                    │
  │  • Block Management: PagedAttention block tables           │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Acceleration:                                             │
  │  • Speculative Decoding: draft model + verification        │
  │  • Medusa: multiple draft heads                            │
  │  • FlashAttention 2: optimized attention kernels           │
  └─────────────────────────────────────────────────────────────┘
    ↓
  LLM Generation (rapid response)
```

**Key Libraries:**
```python
# Core Serving
vllm>=0.5.0                # Primary serving engine (PyTorch-based)
sglang>=0.2.0              # Alternative serving engine
flash-attn>=2.5.0          # Fast attention kernels

# Quantization
bitsandbytes>=0.43.0       # 8-bit / 4-bit quantization
auto-gptq>=0.7.0           # GPTQ quantization
optimum>=1.20.0            # HuggingFace optimization

# Custom KV Cache (PyTorch)
torch>=2.3.0               # Custom cache manipulation
triton>=2.3.0              # Custom CUDA kernels

# Monitoring
nvidia-ml-py>=12.0         # GPU metrics
prometheus-client>=0.20    # Metrics export
```

**Important:** vLLM is built ON PyTorch. You don't choose between them — vLLM IS your PyTorch serving layer. Use PyTorch directly only when implementing custom KV cache algorithms that vLLM doesn't support.

---

### 3.3 MAG Stack (Memory-Augmented Generation)

```
MAG Pipeline:
  User Interaction
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Read Memory (State Fetch):                                │
  │  • Redis: Hot session state (working memory)               │
  │  • PostgreSQL: Warm structured memory (episodes, facts)    │
  │  • Neo4j: Relational memory (entity graphs)                │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Memory Retrieval:                                         │
  │  • Semantic: pgvector / Qdrant for similarity              │
  │  • Temporal: Time-indexed queries                          │
  │  • Graph: Neo4j traversal for relationships                │
  │  • Salience: Importance scoring                            │
  └─────────────────────────────────────────────────────────────┘
    ↓
  LLM Inference (with memory-augmented context)
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Write Memory (State Update):                              │
  │  • Episodic: Store interaction with timestamp              │
  │  • Semantic: Extract and deduplicate facts                 │
  │  • Procedural: Extract reusable workflows                  │
  │  • Graph: Update entity relationships                      │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Consolidation (Background):                               │
  │  • Celery job: periodic episode → semantic transformation  │
  │  • LLM reflection: synthesize patterns from episodes       │
  │  • Graph update: abstraction edges                         │
  └─────────────────────────────────────────────────────────────┘
    ↓
  Agent Loop (continue with updated memory)
```

**Key Libraries:**
```python
# Core Memory
redis>=5.0.0               # Hot cache, sessions, pub/sub
psycopg>=3.1.0             # PostgreSQL (with pgvector)
pgvector>=0.2.0            # Vector extension for Postgres
neo4j>=5.20.0              # Graph database
pymongo>=4.7.0             # MongoDB (optional logs)

# Memory Frameworks
mem0ai>=1.0.0              # Memory layer for LLMs
zep-python>=2.0.0          # Memory API (alternative)

# Async & Background
celery>=5.4.0              # Background task queue
redis-py>=5.0.0            # Redis client
aioredis>=2.0.0            # Async Redis

# Data Processing
pandas>=2.2.0              # Data manipulation
numpy>=1.26.0              # Numerical ops
pydantic>=2.7.0            # Data models
```

---

### 3.4 Orchestration Stack (The Meta-Layer)

```python
# Orchestration
fastapi>=0.111.0           # API framework
uvicorn>=0.30.0            # ASGI server (with uvloop)
httpx>=0.27.0              # Async HTTP client
aiohttp>=3.9.0             # Async HTTP (alternative)
tenacity>=8.3.0            # Retry logic

# Configuration
pydantic-settings>=2.3.0   # Environment config
dynaconf>=3.2.0            # Advanced config (alternative)
python-dotenv>=1.0.0       # .env files

# Validation & Parsing
instructor>=1.3.0          # Structured LLM outputs
jsonschema>=4.22.0         # JSON validation

# Testing
pytest>=8.2.0              # Testing framework
pytest-asyncio>=0.23.0     # Async tests
pytest-cov>=5.0.0          # Coverage
factory-boy>=3.3.0         # Test data
faker>=25.0.0              # Fake data

# Development
pre-commit>=3.7.0          # Git hooks
ruff>=0.4.0                # Linting + formatting
mypy>=1.10.0               # Type checking
```

---

## 4. Project Repository Structure

```
unified-ai-system/
├── 📁 .github/
│   └── workflows/
│       ├── ci.yml                    # Lint, test, type-check
│       ├── docker-build.yml          # Container builds
│       └── deploy-staging.yml        # K8s deployment
│
├── 📁 docker/
│   ├── Dockerfile.api                # FastAPI service
│   ├── Dockerfile.worker             # Celery worker
│   ├── Dockerfile.vllm               # vLLM serving
│   ├── docker-compose.yml            # Local development
│   └── docker-compose.prod.yml       # Production stack
│
├── 📁 k8s/
│   ├── helm/
│   │   └── unified-ai/
│   │       ├── Chart.yaml
│   │       ├── values.yaml
│   │       └── templates/
│   │           ├── api-deployment.yaml
│   │           ├── vllm-deployment.yaml
│   │           ├── redis-statefulset.yaml
│   │           ├── postgres-statefulset.yaml
│   │           ├── qdrant-statefulset.yaml
│   │           └── ingress.yaml
│   └── manifests/                    # Raw K8s YAML (alternative)
│
├── 📁 src/
│   ├── 📁 core/
│   │   ├── __init__.py
│   │   ├── config.py                 # Pydantic settings
│   │   ├── exceptions.py             # Custom exceptions
│   │   ├── logging.py                # Structured logging setup
│   │   └── constants.py              # System constants
│   │
│   ├── 📁 api/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app factory
│   │   ├── dependencies.py           # FastAPI dependencies
│   │   ├── 📁 routers/
│   │   │   ├── chat.py               # Chat endpoints
│   │   │   ├── documents.py          # Document upload/management
│   │   │   ├── memory.py             # Memory CRUD endpoints
│   │   │   ├── health.py             # Health checks
│   │   │   └── admin.py              # Admin endpoints
│   │   ├── 📁 middleware/
│   │   │   ├── auth.py               # JWT auth middleware
│   │   │   ├── rate_limit.py         # Rate limiting
│   │   │   └── logging.py            # Request logging
│   │   └── 📁 schemas/
│   │       ├── chat.py               # Pydantic request/response models
│   │       ├── documents.py
│   │       └── memory.py
│   │
│   ├── 📁 rag/
│   │   ├── __init__.py
│   │   ├── 📁 chunking/
│   │   │   ├── base.py               # Abstract chunker
│   │   │   ├── fixed_size.py         # Fixed-size chunking
│   │   │   ├── semantic.py           # Semantic chunking
│   │   │   ├── recursive.py          # Recursive chunking
│   │   │   └── parent_document.py    # Parent document chunking
│   │   ├── 📁 embedding/
│   │   │   ├── base.py               # Abstract embedder
│   │   │   ├── sentence_transformers.py
│   │   │   └── openai.py
│   │   ├── 📁 retrieval/
│   │   │   ├── base.py               # Abstract retriever
│   │   │   ├── vector.py             # Vector search (Qdrant)
│   │   │   ├── hybrid.py             # Hybrid search (vector + BM25)
│   │   │   ├── multi_query.py        # Multi-query retrieval
│   │   │   └── hyde.py               # HyDE retrieval
│   │   ├── 📁 reranking/
│   │   │   ├── base.py               # Abstract reranker
│   │   │   ├── cross_encoder.py      # Cross-encoder reranking
│   │   │   └── llm_reranker.py       # LLM-based reranking
│   │   ├── 📁 advanced/
│   │   │   ├── self_rag.py           # Self-RAG implementation
│   │   │   ├── crag.py               # CRAG implementation
│   │   │   ├── context_compression.py
│   │   │   └── parent_document.py    # Parent document retrieval
│   │   ├── pipeline.py               # Main RAG pipeline orchestrator
│   │   └── indexer.py                # Document indexing service
│   │
│   ├── 📁 cag/
│   │   ├── __init__.py
│   │   ├── 📁 cache/
│   │   │   ├── base.py               # Abstract cache manager
│   │   │   ├── prefix_cache.py       # Prefix caching logic
│   │   │   ├── kv_cache.py           # KV cache management
│   │   │   └── block_manager.py      # PagedAttention-style blocks
│   │   ├── 📁 eviction/
│   │   │   ├── base.py               # Abstract eviction policy
│   │   │   ├── h2o.py                # Heavy-Hitter Oracle
│   │   │   ├── snapkv.py             # SnapKV eviction
│   │   │   └── random.py             # Random eviction (baseline)
│   │   ├── 📁 compression/
│   │   │   ├── base.py               # Abstract compressor
│   │   │   ├── kivi.py               # KIVI quantization
│   │   │   ├── kvquant.py            # KVQuant implementation
│   │   │   └── low_rank.py           # SVD-based compression
│   │   ├── 📁 serving/
│   │   │   ├── vllm_client.py        # vLLM API client
│   │   │   ├── speculative.py        # Speculative decoding wrapper
│   │   │   └── batching.py           # Cache-aware batching
│   │   ├── 📁 offloading/
│   │   │   ├── base.py               # Abstract offloader
│   │   │   ├── cpu_offload.py        # CPU memory offloading
│   │   │   └── disk_offload.py       # Disk/SSD offloading
│   │   ├── preprocessor.py           # Pre-load documents into cache
│   │   └── warmup.py                 # Cache warming from analytics
│   │
│   ├── 📁 mag/
│   │   ├── __init__.py
│   │   ├── 📁 memory/
│   │   │   ├── base.py               # Abstract memory store
│   │   │   ├── episodic.py           # Episodic memory store
│   │   │   ├── semantic.py           # Semantic memory store
│   │   │   ├── procedural.py         # Procedural memory store
│   │   │   └── working.py            # Working memory (context window)
│   │   ├── 📁 storage/
│   │   │   ├── redis_store.py        # Redis hot storage
│   │   │   ├── postgres_store.py     # PostgreSQL warm storage
│   │   │   ├── neo4j_store.py        # Neo4j graph storage
│   │   │   └── qdrant_store.py       # Vector storage for memory
│   │   ├── 📁 retrieval/
│   │   │   ├── base.py               # Abstract memory retriever
│   │   │   ├── semantic.py           # Semantic similarity retrieval
│   │   │   ├── temporal.py           # Time-based retrieval
│   │   │   ├── graph.py              # Graph traversal retrieval
│   │   │   └── multi_strategy.py     # Multi-strategy fusion
│   │   ├── 📁 consolidation/
│   │   │   ├── base.py               # Abstract consolidator
│   │   │   ├── llm_reflection.py     # LLM-based reflection
│   │   │   └── pattern_extraction.py # Pattern extraction from episodes
│   │   ├── 📁 evolution/
│   │   │   ├── base.py               # Abstract evolution engine
│   │   │   ├── contradiction.py      # Contradiction detection
│   │   │   └── update_policy.py      # Update/invalidation policies
│   │   ├── 📁 gating/
│   │   │   ├── base.py               # Abstract memory gater
│   │   │   ├── token_budget.py       # Token budget allocator
│   │   │   └── relevance.py          # Relevance scoring
│   │   ├── agent_loop.py             # Main MAG agent loop
│   │   └── memory_manager.py         # Central memory coordinator
│   │
│   ├── 📁 orchestration/
│   │   ├── __init__.py
│   │   ├── router.py                 # Paradigm router (query classifier)
│   │   ├── budget_allocator.py       # Context budget allocator
│   │   ├── sync_mixer.py             # Synchronization mixer
│   │   ├── latency_cascade.py        # Latency-adaptive fallback
│   │   ├── freshness_router.py       # Freshness-aware data routing
│   │   └── unified_context.py        # Assembles unified context
│   │
│   ├── 📁 llm/
│   │   ├── __init__.py
│   │   ├── client.py                 # Unified LLM client
│   │   ├── models.py                 # Model registry
│   │   ├── prompts.py                # Prompt templates
│   │   └── tokenizer.py              # Tokenization utilities
│   │
│   ├── 📁 models/                    # SQLAlchemy / Pydantic models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── document.py
│   │   ├── session.py
│   │   └── memory_entry.py
│   │
│   ├── 📁 services/
│   │   ├── __init__.py
│   │   ├── chat_service.py           # Main chat orchestration
│   │   ├── document_service.py       # Document management
│   │   ├── memory_service.py         # Memory operations
│   │   └── analytics_service.py      # Usage analytics
│   │
│   └── 📁 workers/
│       ├── __init__.py
│       ├── celery_app.py             # Celery app factory
│       ├── consolidation_worker.py   # MAG consolidation tasks
│       ├── indexing_worker.py        # RAG indexing tasks
│       ├── cache_warmup_worker.py    # CAG cache warming
│       └── sync_worker.py            # Cross-paradigm sync
│
├── 📁 tests/
│   ├── 📁 unit/
│   │   ├── test_rag_chunking.py
│   │   ├── test_rag_retrieval.py
│   │   ├── test_mag_memory.py
│   │   ├── test_cag_cache.py
│   │   └── test_orchestration.py
│   ├── 📁 integration/
│   │   ├── test_api_endpoints.py
│   │   ├── test_rag_pipeline.py
│   │   ├── test_mag_agent.py
│   │   └── test_full_stack.py
│   ├── 📁 fixtures/
│   │   └── sample_documents/
│   └── conftest.py                   # pytest fixtures
│
├── 📁 scripts/
│   ├── setup_dev.sh                  # Development environment setup
│   ├── seed_data.py                  # Database seeding
│   ├── benchmark.py                  # Performance benchmarking
│   └── migrate.py                    # Custom migrations
│
├── 📁 docs/
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment.md
│   └── development.md
│
├── 📁 notebooks/
│   ├── 01_rag_experiments.ipynb
│   ├── 02_cag_benchmarks.ipynb
│   ├── 03_mag_prototypes.ipynb
│   └── 04_unified_demo.ipynb
│
├── .env.example                      # Environment variable template
├── .env.local                        # Local dev secrets (gitignored)
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml                    # Project metadata + dependencies
├── uv.lock / poetry.lock            # Lock file
├── README.md
├── LICENSE
└── Makefile / justfile              # Common commands
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goal:** Working API with basic RAG

**Deliverables:**
- [ ] FastAPI project scaffold with Docker
- [ ] PostgreSQL + Qdrant + Redis running via docker-compose
- [ ] Document upload endpoint (PDF, TXT, MD)
- [ ] Basic chunking (fixed-size) + embedding (MiniLM)
- [ ] Vector search endpoint
- [ ] LLM integration (OpenAI API or local via vLLM)
- [ ] Simple chat endpoint with RAG context

**Tech Focus:** FastAPI, Qdrant, sentence-transformers, PyTorch (for vLLM)

---

### Phase 2: Advanced RAG (Weeks 3-4)
**Goal:** Production-grade RAG with all 9 concepts

**Deliverables:**
- [ ] Hybrid search (vector + BM25)
- [ ] Reranking (cross-encoder)
- [ ] Multi-query retrieval
- [ ] HyDE implementation
- [ ] Parent document retrieval
- [ ] Context compression
- [ ] Self-RAG gate
- [ ] CRAG validation loop
- [ ] Chunking strategy selector

**Tech Focus:** LangChain/LlamaIndex, rank-bm25, BGE models, LangGraph

---

### Phase 3: CAG Infrastructure (Weeks 5-6)
**Goal:** High-performance serving with caching

**Deliverables:**
- [ ] vLLM deployment with prefix caching
- [ ] Document pre-loading into KV cache
- [ ] Cache warming pipeline from RAG analytics
- [ ] PagedAttention block management
- [ ] Speculative decoding setup
- [ ] KV cache eviction (H2O or SnapKV)
- [ ] Quantization (AWQ/GPTQ) for model compression
- [ ] Benchmarking suite (TTFT, throughput, memory)

**Tech Focus:** vLLM, FlashAttention, AutoGPTQ, CUDA profiling

---

### Phase 4: MAG System (Weeks 7-8)
**Goal:** Stateful agent with memory

**Deliverables:**
- [ ] Session memory (Redis)
- [ ] Episodic memory (PostgreSQL)
- [ ] Semantic memory (pgvector)
- [ ] Memory retrieval (semantic + temporal)
- [ ] Memory consolidation worker (Celery)
- [ ] Contradiction detection
- [ ] Memory gating (token budget)
- [ ] Graph memory (Neo4j) — optional

**Tech Focus:** Redis, PostgreSQL+pgvector, Neo4j, Celery, Mem0/Zep

---

### Phase 5: Orchestration (Weeks 9-10)
**Goal:** Unified system with intelligent routing

**Deliverables:**
- [ ] Paradigm router (query classifier)
- [ ] Context budget allocator
- [ ] Latency-adaptive fallback cascade
- [ ] Sync mixer (cross-paradigm consistency)
- [ ] Freshness-aware data routing
- [ ] State-aware RAG
- [ ] Cache-warmed RAG
- [ ] Unified agent loop

**Tech Focus:** FastAPI async, structured LLM outputs (Instructor), event-driven architecture

---

### Phase 6: Production Hardening (Weeks 11-12)
**Goal:** Deployable, observable, scalable

**Deliverables:**
- [ ] Kubernetes deployment (Helm charts)
- [ ] Prometheus + Grafana monitoring
- [ ] Langfuse integration for LLM tracing
- [ ] Rate limiting + auth
- [ ] Load testing (k6 / Locust)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Documentation
- [ ] Multi-tenant isolation

**Tech Focus:** Kubernetes, Helm, Prometheus, Langfuse, GitHub Actions

---

## 6. Infrastructure & DevOps

### 6.1 Local Development (Docker Compose)

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ai_db
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
      - VLLM_URL=http://vllm:8000"
    depends_on:
      - postgres
      - redis
      - qdrant

  vllm:
    build:
      context: .
      dockerfile: docker/Dockerfile.vllm
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: >
      --model meta-llama/Meta-Llama-3-8B-Instruct
      --tensor-parallel-size 1
      --max-model-len 32768
      --enable-prefix-caching
      --quantization awq

  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    depends_on:
      - redis
      - postgres

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ai_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
```

### 6.2 Production (Kubernetes)

```yaml
# Simplified production topology
# 3x API pods (FastAPI)
# 2x vLLM pods (GPU nodes)
# 3x Qdrant pods (stateful set)
# 1x PostgreSQL primary + 1x replica
# 3x Redis Cluster nodes
# 2x Celery workers
# 1x Neo4j
# Ingress via Traefik
# Monitoring: Prometheus + Grafana + Langfuse
```

### 6.3 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run mypy src/
      - run: uv run pytest tests/ --cov=src --cov-report=xml
      - run: uv run pytest tests/ -m integration

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: ./docker/Dockerfile.api
          push: ${{ github.ref == 'refs/heads/main' }}
          tags: ghcr.io/your-org/unified-ai-api:latest
```

---

## 7. Hardware Requirements

### 7.1 Development (Single Machine)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 4090 (24GB) | RTX 4090 / A6000 (48GB) |
| **RAM** | 32GB | 64GB |
| **Storage** | 500GB SSD | 1TB NVMe SSD |
| **CPU** | 8 cores | 16 cores |
| **OS** | Ubuntu 22.04 | Ubuntu 22.04 |

**Model for dev:** Llama-3-8B-Instruct (AWQ 4-bit fits in 8GB VRAM)

### 7.2 Production (Per Node)

| Component | API Node | GPU Node (vLLM) |
|-----------|----------|-----------------|
| **GPU** | — | 2× A100 80GB or 4× L40S |
| **RAM** | 32GB | 128GB |
| **Storage** | 500GB SSD | 2TB NVMe |
| **CPU** | 8 cores | 32 cores |
| **Network** | 10Gbps | 25Gbps (NVLink if multi-GPU) |

**Model for prod:** Llama-3-70B-Instruct (tensor-parallel across 2-4 GPUs)

### 7.3 Cloud Estimates (AWS)

| Environment | Instance Type | Monthly Cost |
|-------------|--------------|--------------|
| **Dev** | g5.xlarge (1× A10G) | ~$500 |
| **Staging** | g5.12xlarge (4× A10G) | ~$4,500 |
| **Production** | p4d.24xlarge (8× A100) | ~$30,000 |

---

## 8. Alternative Technology Choices

### 8.1 If You Prefer Go / Rust

| Python Component | Go Alternative | Rust Alternative |
|-----------------|---------------|-----------------|
| FastAPI | Gin / Echo | Axum / Actix |
| Celery | Asynq | None (use tokio) |
| Qdrant client | Official Go client | Official Rust client |
| PyTorch | — | tch-rs (limited) |

**Verdict:** Python is strongly recommended for this project due to the ML ecosystem. Go/Rust only for specific high-performance microservices.

### 8.2 If You Prefer Cloud-Managed

| Self-Hosted | Cloud Alternative |
|-------------|-------------------|
| Qdrant | Pinecone, Weaviate Cloud |
| PostgreSQL | AWS RDS, Cloud SQL |
| Redis | AWS ElastiCache, Redis Cloud |
| Neo4j | Neo4j Aura |
| vLLM | Together AI, Fireworks, Groq |
| Kubernetes | AWS EKS, GCP GKE |
| MinIO | AWS S3, GCS |

### 8.3 If You Want Simpler Stacks

| Full Stack | Simplified Alternative |
|-----------|----------------------|
| LangChain | LiteLLM + custom code |
| vLLM | Ollama (local only) |
| Qdrant + BM25 | Chroma (all-in-one) |
| Neo4j | NetworkX (in-memory) |
| Celery + RabbitMQ | APScheduler + Redis |
| Kubernetes | Docker Compose + systemd |

---

## Summary: The Essential Stack

If you only remember 10 technologies:

| # | Technology | Role |
|---|-----------|------|
| 1 | **PyTorch** | Deep learning foundation |
| 2 | **vLLM** | LLM serving (built on PyTorch) |
| 3 | **FastAPI** | API framework |
| 4 | **Qdrant** | Vector database |
| 5 | **PostgreSQL + pgvector** | Structured + vector storage |
| 6 | **Redis** | Hot cache + sessions |
| 7 | **LangChain / LlamaIndex** | RAG orchestration |
| 8 | **Celery** | Background tasks |
| 9 | **Docker + Kubernetes** | Deployment |
| 10 | **Prometheus + Grafana** | Monitoring |

**PyTorch is essential but not sufficient.** Use it for:
- Custom model fine-tuning
- Custom KV cache research
- Building novel attention mechanisms

**Use vLLM (built on PyTorch) for:**
- Production serving
- PagedAttention
- Prefix caching
- Speculative decoding

This stack gives you a production-grade, scalable, observable unified RAG×CAG×MAG system.
