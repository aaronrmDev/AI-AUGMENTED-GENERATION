# Architecture Overview

This document is the detailed architecture and technology-stack reference behind CLAUDE.md's rules — the layered system design, the orchestration meta-layer that ties RAG, CAG, and MAG together, the context-budget math, the fallback cascade, three synthesis techniques this project commits to by name, the full technology stack, what that stack costs to run, where it's fine to substitute a different tool, and the concrete module layout this codebase is heading toward once implementation begins. Everything here traces to CLAUDE.md's own rules or to the two source documents it was synthesized from — `docs/inputs/concepts/unified_rag_cag_mag_architecture.md` (the orchestration-concepts document) and `docs/inputs/concepts/fullstack_unified_ai_system.md` (the stack-and-repository-layout document) — and where a claim isn't something either of those states outright, that's said plainly rather than left to look like settled fact.

## The layered architecture: three parallel paradigms under one coordinator, all resting on the LLM core

```text
LAYER 4: ORCHESTRATION (Meta-Layer)
  Paradigm Router · Context Budget Allocator · Latency-Adaptive Fallback Cascade · Sync Mixer · Freshness-Aware Data Router
────────────────────────────────────────────────────────────────
LAYER 3: MAG (State Layer)          — session memory, highly stateful, writes continuously
LAYER 2: CAG (Cache Layer)          — GPU-resident cache, pre-baked, invalidates in batches
LAYER 1: RAG (Retrieval Layer)      — external index, stateless, updates instantly
────────────────────────────────────────────────────────────────
FOUNDATION: LLM Core (inference engine + context window, shared by every layer above)
```

Both source documents draw this same four-layer stack, numbered identically (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md` §4.2, "Layer Architecture"; the original project brief's §2.1). The foundation earns its position for a literal reason the source states directly: the LLM's context window and inference engine are "shared by all layers above" it — CAG's frozen KV cache lives inside that engine's GPU memory, RAG's retrieved chunks get injected into that same context window, and MAG's session state gets read back into it every turn. Nothing above the foundation can operate without it, so it sits underneath everything else by necessity, not by convention.

The three paradigm layers above the foundation are drawn as a vertical stack — RAG at Layer 1, CAG at Layer 2, MAG at Layer 3 — but they are not dependent on each other the way layers usually are in a stack diagram: RAG doesn't need CAG to function, and CAG doesn't need MAG. `fullstack_unified_ai_system.md`'s own system diagram (§1, "High-Level Architecture") actually draws them side by side as three parallel "PARADIGM LAYERS" boxes sitting under a single Orchestration layer, rather than stacked on top of one another — which is the more literal picture of how they relate. Read against the numbering, though, the vertical order does track something real: it follows the same progression the paradigm comparison table uses for "Operational State" — RAG is "Completely Stateless," CAG is "Pre-baked / Frozen," and MAG is "Highly Stateful & Mutating" (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md` §1). Read bottom-to-top, the stack goes from the paradigm with the least memory of its own (RAG, which looks nothing up until asked) to the one frozen in place between updates (CAG) to the one that rewrites itself on every turn (MAG). That reading is this document's own inference from the source's comparison table, not something either source states as the explicit rationale for the diagram's order — but it's the only account of the ordering that's consistent with both diagrams at once.

Orchestration sits above all three for a reason the source is explicit about: it is the "meta-layer" that "decides which paradigm handles which piece of knowledge at which moment" (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md` §1). That decision is structurally impossible to make from inside any one paradigm layer — RAG's index has no way of knowing whether MAG's session state already answers the query, and CAG's frozen cache has no way of knowing whether RAG's index just got fresher. Only a layer positioned above all three, with visibility into all three, can route a query, divide the context budget, decide which paradigm to try first, and keep their answers from contradicting each other. The next section walks through the five components that do that job.

## The orchestration meta-layer's five components

Each of the five pieces below owns one specific decision. What they have in common is that the decision belongs at this layer specifically because it requires visibility across all three paradigms — something no single paradigm layer has on its own.

### Paradigm Router — deciding which paradigm answers a query

The router classifies an incoming query along several axes before it goes anywhere: how fresh the answer needs to be (real-time favors RAG, static favors CAG, session-only favors MAG), whether it references prior conversation (a MAG signal), how tight the latency budget is, and how complex the question is. A query like "What's our refund policy?" is static and asked often, so it routes to CAG; "What changed in the policy today?" needs data CAG's frozen cache can't have, so it routes to RAG; "Continue where we left off yesterday" is pure state retrieval with no external-knowledge component at all, so it routes to MAG alone; a query like "Compare today's sales with last month" needs both live external data and the session's own context, so it routes to RAG and MAG together (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 1). When the classifier isn't confident which paradigm applies, the source's answer is to run candidates in parallel and merge the results rather than guess. This decision has to sit above the three paradigms because none of them can see the others' contents to know whether they're even the right one to ask — CAG doesn't know if its cached answer is now stale, and RAG doesn't know a cheaper answer already exists in MAG.

### Context Budget Allocator — deciding how much context window each paradigm gets

Even a large context window runs out if CAG tries to preload an entire corpus, MAG tries to hold fifty turns of history, and RAG tries to inject twenty retrieved chunks, all at once, on every request — the model ends up "lost in the middle" of a context stuffed with everything and organized by nothing (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 3). The allocator is the one component with a global view of the total budget and each paradigm's current need, so it's the only place that can arbitrate the fight over space before it happens. Its actual numbers are covered in their own section below.

### Latency-Adaptive Fallback Cascade — deciding what order paradigms get tried in, and when to give up and move on

CAG, MAG, and RAG have wildly different latency profiles — CAG answers from a frozen cache in effectively no time, MAG's session lookups typically take 1–10ms, and RAG's external retrieval typically takes 50–500ms, with the cascade giving it up to a full 2s before giving up. Left to their own devices none of the three paradigms would know when to stop waiting on themselves and hand off to something faster or more thorough; a paradigm layer has no view of what the other tiers would cost the caller, so it can't make that trade-off itself. The cascade sits above all three specifically to enforce a shared timeout budget and decide, layer by layer, whether to return what it has or escalate. Its tiers and timeouts are covered in detail further down.

### Sync Mixer — deciding how three different update rhythms stay reconciled

RAG's index updates the instant a source document changes; CAG's cache invalidates in scheduled or event-driven batches; MAG writes to its memory tables on every turn (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 4). None of the three paradigms has a channel to tell the other two that the underlying fact they're both holding just changed — RAG's re-indexing has no built-in path to CAG's cache or MAG's session state. The mixer exists because that coordination has to happen somewhere, and the only place with visibility into all three stores at once is the orchestration layer. Its tiebreak rule for when the three paradigms actually disagree is covered in its own section below.

### Freshness-Aware Data Router — deciding, at ingestion, which paradigm should own a piece of data in the first place

This is a different decision from the Paradigm Router's, and it happens at a different time: the Paradigm Router decides, per query, which paradigm to consult; the Freshness-Aware Data Router decides, per data source, which paradigm should hold that data at all, based on how fast it changes. Stock prices and live scores change by the second and are "too volatile for cache," so they go to RAG only; company policies and manuals change monthly or quarterly and are "stable enough to pre-load," so they go to CAG; user session state changes every turn and "must be mutable per interaction," so it goes to MAG (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 9). This is a system-wide policy decision made once per data source, not a per-request routing decision, which is why it belongs at the orchestration layer rather than inside any single paradigm — a paradigm layer only sees the data it's already been given; deciding where new data should land in the first place requires seeing across all three.

## Slicing the context window: the budget numbers and why they're shaped this way

The default allocation, against a 128K-token context window, is:

| Slice | Size | Content | Paradigm |
|-------|------|---------|----------|
| **CAG Slice** | 40% (51K) | Frozen pre-loaded docs, system prompt, static knowledge | CAG |
| **MAG Slice** | 25% (32K) | Session state, conversation history, user preferences | MAG |
| **RAG Slice** | 20% (26K) | Dynamically retrieved chunks per query | RAG |
| **Query Slice** | 10% (13K) | Current user query + instructions | — |
| **Reserve** | 5% (6K) | Buffer for generation output | — |

(`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 3; the original project brief's §2.2)

The shares aren't arbitrary once you connect them back to each paradigm's own nature. CAG gets the largest slice because CAG's content is frozen and reused turn after turn at effectively zero marginal cost once it's cached — front-loading the budget with static content pays for itself across an entire session, unlike RAG's chunks, which have to be fetched and re-injected fresh on every call that needs them. MAG's 25% has to cover a session's worth of continuity — preferences, history, scratch state — without being allowed to grow unbounded, which is exactly why MAG's own architecture (covered in `docs/architecture/MAG.md`) includes consolidation and eviction rather than just letting this slice expand indefinitely. RAG gets the smallest of the three paradigm slices because it's invoked selectively through the router and the cascade rather than on every turn, so it doesn't need a standing allocation as large as CAG's or MAG's. The Query slice is close to non-negotiable — without room for the actual question and its instructions there's nothing to answer — and the Reserve exists so the model always has somewhere to put its output even if the four slices ahead of it filled the rest of the window.

The allocation is explicitly not fixed. The source's dynamic-reallocation rules give unused budget to whichever paradigm is likely to benefit most on that particular turn, rather than letting it sit idle in a slice nothing needs: if a query is simple enough that RAG isn't required, the freed RAG budget expands MAG's slice instead, since more room for conversational context costs nothing and improves continuity when retrieval wasn't going to be used anyway. If a session is brand new, MAG has no state to contribute, so the CAG or RAG slice expands to absorb what would otherwise be dead space. If CAG comes back with a cache miss, the RAG slice temporarily grows to compensate for the knowledge CAG couldn't supply. And if MAG's own state grows too large on its own, the source's answer isn't to keep expanding its slice indefinitely but to trigger compression or eviction instead (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 3). The original project brief's own condensed version of this rule stated the two most common cases directly: "if RAG is not needed, MAG expands; if MAG is empty, CAG expands" — both are specific instances of the same general principle, that a paradigm's slice grows when it has something to offer and another paradigm's slice has nothing to spend its budget on that turn.

## The latency-adaptive fallback cascade in detail

| Tier | Paradigm | Timeout | What happens at this tier |
|------|----------|---------|----------------------------|
| 1 | CAG | 10ms | Cache checked; a hit returns immediately from pre-loaded static knowledge |
| 2 | MAG | 50ms | Session state / conversation history checked; a hit returns a stateful answer |
| 3 | RAG | 2s | External retrieval always runs (no hit/miss — this tier does the full lookup); returns a comprehensive answer |

(`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 5; the original project brief's §2.3)

The cascade tries the cheapest paradigm first and only pays for a more expensive one when the cheaper tiers genuinely can't answer. A query enters at CAG: if the cache holds the answer, the response goes back to the user with no further work, because paying for a MAG lookup or a RAG round-trip on top of a cache hit would be pure waste. A CAG miss falls through to MAG, which checks session state and conversation history; a hit there returns a stateful answer built from what the session already knows. Only when both of the fast tiers miss, or what they found isn't sufficient, does the request fall through to RAG's full retrieval — embedding, database lookup, ranking, network transfer — which is slower by design because it's doing genuinely more work: reaching outside the system for knowledge neither the cache nor the session has.

The timeouts aren't just performance targets, they're guards against a slow tier stalling a request that a faster tier could have handled adequately: if the CAG lookup itself is slow, the cascade skips it rather than waiting; if MAG's state turns out to be complex enough that resolving it exceeds its 50ms budget, the cascade moves on to RAG rather than blocking there. The source also describes smarter fallback behavior than a strict miss-and-retry: a partial CAG match can be supplemented with a RAG call rather than discarded outright; MAG state that looks stale can be invalidated and refetched via RAG instead of trusted as-is; and if RAG itself is running slow, the cascade can return whatever CAG or MAG already has as a best-effort answer while the RAG result updates asynchronously in the background (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 5). None of these timeout numbers or fallback behaviors have been measured against a running system yet — there is no serving stack in this repository to benchmark against — so they should be read as the targets the architecture commits to, not results already achieved.

## Three named synthesis techniques this system commits to

The three paradigms don't just coexist — the source document names three specific ways they're meant to work together, beyond the orchestration components above. CLAUDE.md's current architecture rules use these ideas without naming them; naming them here makes them things this project can refer to directly.

### Tiered Knowledge Hot-Cold Architecture

Knowledge has a temperature, in the sense that how often something gets looked up should determine where it's stored. The source lays out three tiers: hot data — system prompts, static docs, code repositories, textbooks, FAQs — gets accessed on the order of a thousand times an hour and belongs in CAG, where lookups cost close to nothing because the content is already sitting in the frozen cache. Warm data — user preferences, session state, conversation history, an agent's scratchpad — gets accessed on the order of ten times an hour and belongs in MAG, in RAM or a fast database, at 1–10ms latency. Cold data — an enterprise knowledge base, live databases, real-time APIs, external web data, documents that are rarely touched — gets accessed roughly once a day or purely on demand and belongs in RAG's vector database or external index, at 50–500ms latency (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 2). A customer support bot illustrates the placement cleanly: its product manual, FAQ, and return policy are hot and sit in CAG; a user's ticket history and current issue state are warm and sit in MAG; the latest forum posts and real-time inventory are cold and stay in RAG.

The architecture isn't static once data is placed, though. New data gets classified by its expected access pattern at ingestion and placed accordingly — hot into CAG preload, warm into MAG tables, cold into the RAG index — but the source specifies two ongoing corrections on top of that initial placement: promotion, where cold data that turns out to be accessed far more often than expected gets moved up to warm or hot, and demotion, where hot data that turns out to be accessed rarely gets moved back down to save the GPU VRAM it was occupying. Invalidation happens at whichever tier the changed data currently lives in. The underlying principle is to match the storage tier to the actual access pattern rather than to how important the data seems in the abstract — hot data belongs in GPU memory, warm in RAM, cold on disk or over the network, and getting that placement wrong in either direction wastes either latency or VRAM.

### State-Aware RAG

Standard RAG retrieves based only on the text of the current query, which throws away everything the system already knows about the user through MAG. The source's example makes the gap concrete: a user who's spent ten turns discussing Python data science, and whose MAG state records a preference for matplotlib, a dislike of Plotly, and habitual use of pandas, asks "How do I visualize this?" A retrieval pipeline with no access to that state has nothing to go on but four words and returns a generic visualization article that name-drops five different libraries. State-Aware RAG instead reads MAG's state first — the user's preferences, the conversation history, the current task context — and uses it to rewrite the query before retrieval ever runs: "How do I fix this?" becomes, for a user MAG knows is deploying FastAPI on Docker at an intermediate skill level, something closer to "how to fix FastAPI Docker deployment issues for an intermediate developer." Retrieval then runs against that enriched query, ranking gets a boost for results matching the user's known stack and skill level, and whatever comes back gets written back into MAG so the next turn has it too (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 6). For the visualization example, that pipeline retrieves matplotlib tutorials and pandas plotting documentation specifically, instead of the generic five-library overview standard RAG would have returned. The source's own framing for the relationship is direct: MAG makes RAG personal, and RAG in turn gives MAG something external to be personal about.

### Cache-Warmed RAG

RAG pays the full cost of embedding, database lookup, ranking, and network transfer on every single query, even though real query traffic tends not to spread evenly across a document set — the source's working assumption is that roughly 80% of queries hit the same 20% of documents, a Pareto pattern (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 7). Cache-Warmed RAG turns that skew into a shortcut: an analytics process tracks which documents RAG retrieves most often, the most frequently retrieved documents get pre-loaded into CAG's frozen KV cache, and at query time the system checks CAG first — an instant answer on a hit, a normal RAG fall-through on a miss. The cache gets periodically re-warmed as the analytics shift, and even on a cache miss the source notes that the pre-loaded documents can still improve RAG's own ranking by providing context the retriever wouldn't otherwise have. The worked example is a support bot where 80% of questions cluster around return policy, shipping, and account setup — those three documents get pre-loaded into CAG, and the result is that 80% of traffic gets answered in close to zero time while only the remaining 20% pays RAG's full retrieval cost. That 80/20 split is the source's own stated assumption about query distribution, not a number this project has measured against its own corpus — the technique only pays off to the degree the assumption actually holds, which is exactly what the analytics step in its mechanism exists to check.

## The sync mixer's tiebreak rule: RAG wins

When the three paradigms' different sync rhythms leave them disagreeing about the same underlying fact — RAG's index already updated, CAG's cache hasn't invalidated yet, MAG's session state still remembers the old value — the sync mixer's conflict-resolution step has one explicit rule: RAG, as the external source, wins as the source of truth (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 4). The source's worked example is a price drop from $100 to $80: RAG's index updates the moment the new document is ingested; the mixer detects the change and flags the corresponding CAG cache entry for invalidation; CAG actually invalidates on its next batch cycle, which the source's example puts at within five minutes — an illustrative figure from that example, not a fixed SLA this project has committed to; and MAG's session state, if it was holding the old price, gets corrected via notification once CAG's invalidation lands. The end state is that every paradigm converges on the same $80 answer, but only because there was an explicit, stated rule for which paradigm's data to trust while the others catch up — without it, the three would just keep giving three different answers to the same question for as long as their sync rhythms stayed out of step.

## The full technology stack

The stack below is what the original project brief's §3.1–§3.8 already committed to, carried over here with one correction: React is **18+** throughout. The original project brief's header for its frontend section named React major version 19, while its own stack table in that very same section already listed "React 18+" — and neither source document specified version 19 anywhere else. That was an internal contradiction in the original file, not a deliberate version choice, and it's corrected here to React 18+ consistently, since this is where that table now lives. Five of the choices below already have their reasoning recorded separately as ADRs — vLLM over SGLang, Qdrant over Milvus/Weaviate, standard attention over alternative attention, Hexagonal+CQRS for MAG, and LangChain/LangGraph for RAG orchestration — in `docs/decisions/adr/`; this section is the inventory, those files are the "why."

`fullstack_unified_ai_system.md` frames PyTorch's role in one useful metaphor worth carrying forward: PyTorch is the engine — tensor operations, model weights, custom CUDA kernels — a serving framework like vLLM is the car built on top of that engine, and this project's own code is the driver. PyTorch alone doesn't give production-grade batching, PagedAttention, prefix caching at serving scale, or tensor parallelism across GPUs; vLLM does, because vLLM is built on PyTorch specifically to add those capabilities. The stack below reflects that split: PyTorch is used directly only for custom KV-cache research and fine-tuning, and vLLM is the actual serving layer everything else talks to.

### Core backend

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

### Served models and hardware target

The stack table above and the CAG ecosystem table below both assume NVIDIA/CUDA, because that's what `fullstack_unified_ai_system.md` assumes throughout — Triton, `flash-attn`, and the CUDA toolkit version are all NVIDIA-specific. This project's actual initial deployment target is different: **vLLM on ROCm**, running on an AMD Radeon 7900 XTX (24GB VRAM), because that's the hardware actually available for this project's own experimentation. This is stated here as an explicit reconciliation, the same way the React 18+ correction earlier in this section and `docs/governance/GIT_WORKFLOW.md`'s squash-versus-merge ruling are — vLLM does have official ROCm support, so the serving-engine choice (ADR-0001) doesn't change, but the CUDA-specific kernel entries in the CAG ecosystem table below (`triton`, `flash-attn`) need ROCm equivalents that haven't been selected or benchmarked yet, which is flagged here as design intent, not yet verified, rather than left as a silent assumption that CUDA is available.

Two models are served locally against this hardware, and three more are called through their APIs as reference points rather than self-hosted — the full reasoning for why each one is in which category, and the citations behind every figure below, live in `docs/evaluation/COMPARISON_METHODOLOGY.md`; this table is the inventory, that document is the "why."

| Model | Role | Context | Notes |
|---|---|---|---|
| Gemma 4 (12B / 26B-A4B / 31B) | Self-hosted, vLLM on ROCm | 256K native | Apache 2.0; chosen first, ahead of the context comparison |
| Qwen3.8-27B | Self-hosted, vLLM on ROCm | 262,144 native, extensible to 1M (YaRN) | Apache 2.0; single-GPU by design; stronger of the two on coding benchmarks and native context |
| DeepSeek V4 (V4-Pro / V4-Flash) | API reference | 1M, 384K max output | MIT-licensed weights, but not realistically self-hostable on one 24GB card |
| Claude (Sonnet 5 / Opus 5) | API reference, primary qualitative judge | 1M | Not one of the models being ablated — see `docs/evaluation/qualitative-rubric.md` |
| Gemini (3.1 Pro / 3.6 Flash) | API reference, secondary qualitative judge | ~1.05M | Independent model family from both the self-hosted models and the primary judge |

### Data storage

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

### RAG ecosystem

| Component | Library | Purpose |
|-----------|---------|---------|
| Orchestration | LangChain >=0.2.0 / LlamaIndex >=0.10.0 | RAG pipelines |
| Agent Graphs | LangGraph >=0.1.0 | CRAG, Self-RAG workflows |
| Document Parsing | unstructured >=0.14.0 | PDF, MD, TXT parsing |
| Embeddings | sentence-transformers >=3.0 | Vector generation |
| Keyword Search | rank-bm25 >=0.2.2 | BM25 keyword search |
| Reranking | BAAI/bge-reranker-v2-m3 | Cross-encoder reranking |

### CAG ecosystem

| Component | Library | Purpose |
|-----------|---------|---------|
| Serving Engine | vLLM >=0.5.0 | PagedAttention, prefix caching |
| Attention Kernels | flash-attn >=2.5.0 | Optimized attention |
| Quantization | bitsandbytes, auto-gptq, optimum | Model compression |
| Custom CUDA | triton >=2.3.0 | Custom kernels |

### MAG ecosystem

| Component | Library | Purpose |
|-----------|---------|---------|
| Memory Framework | mem0ai >=1.0.0 | Memory layer for LLMs |
| PostgreSQL Client | psycopg >=3.1.0 | Structured storage |
| Vector Extension | pgvector >=0.2.0 | Postgres vectors |
| Graph DB | neo4j >=5.20.0 | Relationship memory |
| Async Redis | redis-py >=5.0.0 / aioredis >=2.0.0 | Hot cache |

### Observability

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM Tracing | Langfuse | Trace LLM calls |
| Metrics | Prometheus | System metrics |
| Dashboards | Grafana | Visualization |
| Logging | structlog | Structured logging |
| APM | OpenTelemetry | Distributed tracing |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containers | Docker | Containerization |
| Orchestration | Kubernetes | Container orchestration |
| Packaging | Helm | K8s package management |
| Ingress | Traefik | Load balancing |
| TLS | cert-manager | TLS automation |
| Secrets | HashiCorp Vault | Secret management |
| IaC | Terraform | Infrastructure as code |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React 18+ | Web UI |
| Styling | Tailwind CSS | Utility-first CSS |
| State | Zustand | State management |
| Streaming | SSE / WebSocket | Token streaming |

## Hardware and cost: what this stack actually requires to run

None of this is in CLAUDE.md today, and it matters for the same reason the stack tables above do: a technology choice is only meaningful once you know what it costs to actually run. The figures below are `fullstack_unified_ai_system.md`'s own estimates (§7) — nothing here has been benchmarked against a real deployment of this project, since no deployment exists yet.

A single development machine needs, at minimum, an RTX 4090 with 24GB of VRAM, 32GB of system RAM, a 500GB SSD, and 8 CPU cores, with 64GB of RAM and an RTX 4090 or A6000 (48GB) recommended if the budget allows it. At that scale, the model that actually fits is Llama-3-8B-Instruct quantized to AWQ 4-bit, which the source notes fits in as little as 8GB of VRAM — comfortably inside even the minimum GPU spec. Production hardware looks different by node role: an API node needs no GPU at all — 32GB RAM, a 500GB SSD, 8 CPU cores, and 10Gbps networking cover it — while a GPU node running vLLM needs two A100 80GB GPUs or four L40S GPUs, 128GB RAM, 2TB of NVMe storage, 32 CPU cores, and 25Gbps networking with NVLink if multiple GPUs are involved. At that scale the model changes too: Llama-3-70B-Instruct, tensor-parallel across two to four GPUs, replaces the 8B model used for development.

Cloud estimates from the same source (AWS, monthly, approximate): a dev environment on a `g5.xlarge` (one A10G GPU) runs around $500/month; staging on a `g5.12xlarge` (four A10G GPUs) runs around $4,500/month; and a production deployment on a `p4d.24xlarge` (eight A100 GPUs) runs around $30,000/month. The jump between those three numbers is almost entirely GPU count and class, not anything else in the stack — which is consistent with the hardware tables above, where the API-serving side of this architecture stays cheap and commodity while the GPU-serving side is where the real cost lives.

## When you might deviate from this stack

The tables above are the default, not the only option this project's source material considers. `fullstack_unified_ai_system.md` §8 lays out three specific directions a deviation might go, each for a different reason.

**If you'd rather build in Go or Rust:** FastAPI's rough equivalents are Gin or Echo in Go and Axum or Actix in Rust; Celery's nearest Go equivalent is Asynq, with Rust offering no direct match beyond building on `tokio` directly (the source spells this out explicitly, listing Celery's Rust alternative as "None (use tokio)"); Qdrant has an official client in both languages. PyTorch does not follow that same both-languages pattern: its Go column in the source table is empty, and reading that against the table's own convention — an explicit "None" where no equivalent truly exists — an empty cell means no Go binding exists for PyTorch at all, not merely that one wasn't listed. Rust fares only a little better, with a binding (`tch-rs`) the source itself calls limited. The source's own verdict is blunt about when this is worth it: Python is strongly recommended for this project because of its ML ecosystem, and Go or Rust make sense only for specific, narrowly-scoped high-performance microservices carved out of the larger system — not as a wholesale replacement for the Python stack above.

**If you'd rather not self-host:** every self-hosted component in the stack tables has a managed equivalent — Qdrant can become Pinecone or Weaviate Cloud, PostgreSQL can become AWS RDS or Cloud SQL, Redis can become AWS ElastiCache or Redis Cloud, Neo4j can become Neo4j Aura, vLLM's serving role can move to Together AI, Fireworks, or Groq, Kubernetes can become AWS EKS or GCP GKE, and MinIO can become AWS S3 or GCS. This trades operational burden for a recurring cost and less control over the underlying infrastructure — the right call when a team would rather not run its own database and cache fleet, and the wrong call when the cost or data-locality requirements make self-hosting worth the operational overhead.

**If you want something simpler than the full stack:** LangChain can be replaced by LiteLLM plus custom orchestration code, vLLM can be replaced by Ollama for local-only serving, the Qdrant-plus-BM25 hybrid-search pair can collapse into Chroma as an all-in-one store, Neo4j can be replaced by NetworkX running in memory, Celery-plus-RabbitMQ can be replaced by APScheduler plus Redis, and Kubernetes can be replaced by Docker Compose plus systemd. Every substitution in this list trades production capability — scale, durability, operational maturity — for less setup and faster iteration, which makes this the right direction for early experimentation or a proof of concept, not for the production-grade system this stack is otherwise built for.

## The Phase 1 module blueprint

`fullstack_unified_ai_system.md` §4 lays out a concrete `src/` module structure for this system, organized around the same three paradigms plus the orchestration layer that coordinates them. The Auth Foundation sub-project (August 2026) established the authentication infrastructure alongside these paradigm modules: `src/api/` and `src/identity/` exist, along with `tests/unit/`, `tests/integration/`, and the Docker configuration needed to run them locally (`docker/Dockerfile.api` and `docker/docker-compose.yml`). The RAG Pipeline sub-project that followed it has since built the first of the three paradigm modules: `src/rag/` now exists, with its domain, application, and infrastructure layers in place, and `src/api/routers/` has grown a `documents` router and a `chat` router alongside the `auth` router Auth Foundation added — the endpoints a client actually calls to upload a document and get a grounded answer back. The MAG Memory sub-project has since built the second: `src/mag/` now exists, with its own domain, application, and infrastructure layers in place across seven completed batches (Foundation through Combinations) covering episodic, semantic, and procedural memory, consolidation, memory evolution, memory gating, and memory graphs — see `docs/architecture/MAG.md` for the concepts and `evaluation/reports/mag-*.md` for each batch's own live-measured narrative report. `src/cag/` has three batches in: a domain-layer module (`src/cag/domain/attention_compatibility.py`) encoding ADR-0003's alternative-attention compatibility split; a second batch adding the domain spine (`ports.py`'s two compressor interfaces, `entities.py`'s `CompressedKV`, `compression_metrics.py`) plus five real KV Cache Compression algorithms under `src/cag/infrastructure/` (KIVI, KVQuant, PALU, MiniCache, ShadowKV); and a third batch adding `src/cag/application/` (`SpeculativeDecode`, the propose-verify-accept loop) plus three Speculative Decoding candidate generators (Medusa, Lookahead Decoding, Prompt Lookup Decoding) and the `HFTargetModel`/`ngram_search.py` infrastructure they share — see `docs/architecture/CAG.md` for the concepts and `evaluation/reports/cag-*.md` for each batch's own measured report. The techniques needing real GPU/vLLM serving to honestly measure (Eviction, PagedAttention, Cache-Aware Batching, Hybrid Offloading, Multi-Turn Caching, and the Combinations) remain queued behind that infrastructure being set up. `src/orchestration/` has its first real content: a cross-paradigm RAG+CAG synthesis batch built `domain/` (`entities.py`'s `CacheHit`/`SyncConflict`/`TierDecision`, `ports.py`'s `AccessFrequencyTracker`/`FrozenCache`, `sync_mixer.py`'s RAG-wins reconciliation), `application/` (`WarmCache`, `TieringPolicy`, `SyncCycle`, `CacheWarmedRetrieve`), and `infrastructure/` (`InMemoryAccessFrequencyTracker`, `HFFrozenCache` — a real, CPU-sized proxy for CAG's GPU-resident cache using the same real-`distilgpt2` methodology CAG Batch B's own integration test proved out), implementing Cache-Warmed RAG, the CAG↔RAG hot/cold tiering boundary, and the Sync Mixer's RAG-vs-CAG tiebreak — see `evaluation/reports/rag-cag-synthesis.md` for the live-measured report. A second cross-paradigm batch, RAG+MAG synthesis, has since built `domain/`'s `WarmEntry` entity and `UserScopedAccessFrequencyTracker`/`WarmStore` ports (deliberately separate from the RAG+CAG ports above, since MAG's warm tier is inherently personal — every MAG table is `user_id`-keyed — unlike CAG's tenant-wide shared cache), generalized `sync_mixer.reconcile` to take a plain content hash so both pairings' sync mechanisms share it, and added `application/`'s `StateAwareRetrieve`, `MagTieringPolicy`, and `MagSyncCycle` plus `infrastructure/`'s `InMemoryUserScopedAccessFrequencyTracker` and `SemanticMemoryWarmStore` (wrapping this project's existing `PostgresSemanticMemoryRepository`/`QdrantSemanticMemoryIndex`), implementing State-Aware RAG, the RAG↔MAG warm/cold tiering boundary, and the Sync Mixer's RAG-vs-MAG tiebreak — see `evaluation/reports/rag-mag-synthesis.md` for the live-measured report. `src/orchestration/router.py`, `budget_allocator.py`, and `latency_cascade.py` (the Paradigm Router, Context Budget Allocator, and Latency-Adaptive Fallback Cascade) remain unbuilt, as does the CAG+MAG cross-paradigm pairing. `src/workers/`, the Kubernetes configuration under `k8s/`, the additional Docker images (`Dockerfile.worker` and `Dockerfile.vllm`) and production Docker Compose variant (`docker-compose.prod.yml`), and the experiment notebooks under `notebooks/` remain pending as well. The blueprint below documents the full target structure; what follows is both a record of what's been built so far and a guide for what remains to be built.

```text
src/
├── rag/
│   ├── chunking/       # base.py, fixed_size.py, semantic.py, recursive.py, parent_document.py
│   ├── embedding/      # base.py, sentence_transformers.py, openai.py
│   ├── retrieval/      # base.py, vector.py, hybrid.py, multi_query.py, hyde.py
│   ├── reranking/      # base.py, cross_encoder.py, llm_reranker.py
│   ├── advanced/       # self_rag.py, crag.py, context_compression.py, parent_document.py
│   ├── pipeline.py     # Main RAG pipeline orchestrator
│   └── indexer.py      # Document indexing service
│
├── cag/
│   ├── cache/          # base.py, prefix_cache.py, kv_cache.py, block_manager.py
│   ├── eviction/       # base.py, h2o.py, snapkv.py, random.py
│   ├── compression/    # base.py, kivi.py, kvquant.py, low_rank.py
│   ├── serving/        # vllm_client.py, speculative.py, batching.py
│   ├── offloading/     # base.py, cpu_offload.py, disk_offload.py
│   ├── preprocessor.py # Pre-load documents into cache
│   └── warmup.py       # Cache warming from analytics
│
├── mag/
│   ├── memory/         # base.py, episodic.py, semantic.py, procedural.py, working.py
│   ├── storage/        # redis_store.py, postgres_store.py, neo4j_store.py, qdrant_store.py
│   ├── retrieval/      # base.py, semantic.py, temporal.py, graph.py, multi_strategy.py
│   ├── consolidation/  # base.py, llm_reflection.py, pattern_extraction.py
│   ├── evolution/      # base.py, contradiction.py, update_policy.py
│   ├── gating/         # base.py, token_budget.py, relevance.py
│   ├── agent_loop.py       # Main MAG agent loop
│   └── memory_manager.py   # Central memory coordinator
│
├── orchestration/
│   ├── router.py            # Paradigm router (query classifier)
│   ├── budget_allocator.py  # Context budget allocator
│   ├── sync_mixer.py        # Synchronization mixer
│   ├── latency_cascade.py   # Latency-adaptive fallback
│   ├── freshness_router.py  # Freshness-aware data routing
│   └── unified_context.py   # Assembles unified context
│
└── workers/
    ├── celery_app.py             # Celery app factory
    ├── consolidation_worker.py   # MAG consolidation tasks
    ├── indexing_worker.py        # RAG indexing tasks
    ├── cache_warmup_worker.py    # CAG cache warming
    └── sync_worker.py            # Cross-paradigm sync

tests/
├── unit/          # test_rag_chunking.py, test_rag_retrieval.py, test_mag_memory.py, test_cag_cache.py, test_orchestration.py
├── integration/   # test_api_endpoints.py, test_rag_pipeline.py, test_mag_agent.py, test_full_stack.py
├── fixtures/      # sample_documents/
└── conftest.py

docker/
├── Dockerfile.api
├── Dockerfile.worker
├── Dockerfile.vllm
├── docker-compose.yml
└── docker-compose.prod.yml

k8s/
└── helm/
    └── unified-ai/
        ├── Chart.yaml
        ├── values.yaml
        └── templates/  # api-deployment.yaml, vllm-deployment.yaml, redis-statefulset.yaml,
                         # postgres-statefulset.yaml, qdrant-statefulset.yaml, ingress.yaml

notebooks/
├── 01_rag_experiments.ipynb
├── 02_cag_benchmarks.ipynb
├── 03_mag_prototypes.ipynb
└── 04_unified_demo.ipynb
```

The `src/api/` and `src/identity/` directories (which now exist after the Auth Foundation sub-project) are not part of the tree diagram above — they were added because authentication isn't one of the three RAG/CAG/MAG paradigms the blueprint is organized around, and needed their own home outside the diagram's paradigm-centric structure. They provide the authentication and API infrastructure that all three paradigms depend on, rather than implementing one paradigm themselves, and were established during Phase 0 foundation work per `docs/superpowers/specs/2026-08-22-auth-foundation-design.md`.

Every module directory in the paradigm and worker layers maps directly onto something already named earlier in this document: `orchestration/` holds one file per meta-layer component described above, `rag/`, `cag/`, and `mag/` each hold a subdirectory per pipeline stage described in the paradigm's own architecture doc, and `workers/` holds the background jobs — consolidation, indexing, cache warmup, cross-paradigm sync — that the orchestration and memory sections above assume are running asynchronously. This structure is the target for Phase 1 specifically: the source's own implementation roadmap (`fullstack_unified_ai_system.md` §5) scopes Phase 1 to "Working API with basic RAG" over its first two weeks — a FastAPI scaffold running under Docker, PostgreSQL, Qdrant, and Redis wired up via `docker-compose`, a document upload endpoint, fixed-size chunking with MiniLM embeddings, a vector search endpoint, and a basic chat endpoint answering from RAG context. That's a small slice of the full tree above — mostly `rag/`, the earliest pieces of `orchestration/`, and the `docker/` files needed to run it locally — with the rest of the structure filling in across the later phases as CAG infrastructure, MAG's memory system, and the remaining orchestration components get built out. The RAG Pipeline sub-project has now delivered most of that slice: `src/rag/infrastructure/` carries a fixed-size chunker, a `sentence-transformers` embedder, a Qdrant-backed vector store, and a Claude-backed chat model, and `src/rag/application/` wires those into the upload, search, and answer-question use cases the `documents` and `chat` routers expose — the upload endpoint, fixed-size chunking with MiniLM embeddings, vector search, and basic RAG-grounded chat that the roadmap called for. `src/orchestration/` itself still doesn't exist — there's no router, budget allocator, or fallback cascade yet, only the one paradigm module calling directly into its own use cases — and CAG and MAG haven't started at all; this section still records the plan the remaining work will follow.
