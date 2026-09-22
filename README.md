# AI-Augmented Generation

## One question, three answers

Where should a given piece of knowledge live so a language model can use it well? Most systems pick one answer and force everything through it. This one doesn't: it builds three separate, complementary answers to that same question, and a layer above them that decides which one applies, per query, every time.

- **RAG** reaches outside the model for knowledge that changes on its own schedule — a return policy, a product manual, whatever's in a document store.
- **CAG** freezes knowledge that barely changes into the model's own GPU-resident KV cache, so reusing it on the next ten thousand queries costs almost nothing.
- **MAG** keeps track of what a single conversation itself has learned, so the system gets more useful the longer it talks to the same user.

No paradigm is sufficient alone, and this repository doesn't treat any of them as the "real" system with the other two bolted on. All three are built out fully, tested against real infrastructure, and measured against each other — the sections below walk through why that split holds up, then show the evidence that it does.

## Why one paradigm was never going to be enough

Picture three questions the same support bot might get in one conversation. "What's our return policy?" barely changes month to month — the right move is to load it once and keep answering from that frozen copy for free, not re-fetch it every time. "What changed in the policy today?" needs data a frozen cache can't have by definition — the right move is to go look. "Continue where we left off yesterday" needs no external knowledge at all — the right answer is already sitting in the session's own memory. A system built around only one of RAG, CAG, or MAG answers one of those three well and the other two badly. This project answers all three, and adds a coordinator whose only job is deciding which paradigm to trust on a given turn — because none of the three can see what the other two already know.

| Paradigm | Answers | Storage | Latency | Mutability |
|----------|---------|---------|---------|------------|
| **RAG** | "What exists outside the model?" | Vector DB (Qdrant) + BM25 | 50–500ms | Instant sync |
| **CAG** | "What can fit inside the model's cache?" | GPU KV cache (vLLM) | Near-zero TTFT on a hit | Batch invalidation |
| **MAG** | "What has this session learned?" | Redis + PostgreSQL + Neo4j | 1–10ms | Continuous writes |

## The layer that decides, so no paradigm has to guess

Above all three sits an orchestration meta-layer with five jobs, each requiring visibility no single paradigm has on its own:

1. **Paradigm Router** — classifies each incoming query and picks which paradigm, or combination, should answer it.
2. **Context Budget Allocator** — splits a 128K-token context window across the three paradigms turn by turn, instead of every paradigm grabbing space independently and stuffing the model's context with everything at once.
3. **Latency-Adaptive Fallback Cascade** — tries the cheapest paradigm first (CAG, a 10ms budget) and falls through to the next only when the cheap tier genuinely can't answer (MAG at 50ms, then RAG at up to 2s).
4. **Sync Mixer** — resolves what happens when the three paradigms' different update rhythms leave them disagreeing about the same underlying fact.
5. **Freshness-Aware Data Router** — decides, once, at ingestion time, which paradigm should own a new piece of data based on how fast that data actually changes.

The full reasoning, the exact budget math, and every measured trade-off behind these five components live in [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md).

## What's actually built — a paper trail, not a pitch

Every batch of work below shipped through the same discipline: a design spec, an implementation plan, real tests written before the code they cover, and a live-measured report proving the technique does what it claims. Nothing in this table is aspirational.

| Milestone | What it covers | Issues closed |
|---|---|---|
| RAG: Individual Techniques | Six chunking strategies, hybrid search + reranking, parent-document retrieval, HyDE, CRAG, Self-RAG | 25 |
| RAG: Combinations | Composing the individual RAG techniques together | 14 |
| CAG: Individual Techniques | All nine CAG techniques — eviction, compression, offloading, speculative decoding, prefix caching, and more | 37 |
| CAG: Combinations | Long-context and real-time-chat pipelines built from those nine | 6 |
| MAG: Individual Techniques | Episodic/semantic/procedural memory, six retrieval strategies, memory graphs, gating | 31 |
| MAG: Combinations | The full memory lifecycle composed end to end | 8 |
| RAG+CAG: Cross-Paradigm Synergy | Cache-Warmed RAG and the RAG↔CAG hot/cold tiering boundary | 8 |
| RAG+MAG: Cross-Paradigm Synergy | State-Aware RAG and the RAG↔MAG warm/cold tiering boundary | 8 |
| CAG+MAG: Cross-Paradigm Synergy | The CAG↔MAG hot/warm tiering boundary and its own sync tiebreak | 6 |
| Orchestration: Meta-Layer | All five meta-layer components above, built and live-measured | 31 |
| Unified API | HTTP access to the meta-layer: session-scoped answers (shipped), hardening against abuse (shipped), freshness-routed ingestion (shipped), streaming answers (shipped) | 13 closed, 0 open |

That's 187 closed issues across eleven milestones, tracked the same way from the first line of code to the latest release.

### The proof, in numbers

- **1,226 unit tests** across 164 files, and **309 integration tests** across 70 files run against real PostgreSQL, Qdrant, Redis, and Neo4j via TestContainers — not mocks standing in for the parts that are actually hard to get right.
- **52 narrative evaluation reports** under [evaluation/reports/](evaluation/reports/), each one a real baseline-versus-treatment comparison against a live judge model, not a unit test dressed up as a benchmark.
- Every cross-paradigm synthesis technique — Cache-Warmed RAG, State-Aware RAG, and the three tiering/sync boundaries between all three paradigms — is measured against real infrastructure, including a genuine GPU-served vLLM instance for the CAG-side measurements.
- The Unified API's session-scoped endpoint (`POST /sessions`, `GET /sessions`, `POST /sessions/{session_id}/answers`) serves the full per-query orchestration path over HTTP, with object-level authorization enforced on every session lookup.
- Freshness-routed ingestion is served the same way: `POST /data-sources` hands a tenant- or user-scoped source to this project's first background-worker subsystem, Celery against a Redis broker under `src/workers/`, and `GET /data-sources/jobs/{task_id}` polls the result, denying a caller who isn't the job's own tenant or user the same 404 a missing task id gets.
- The per-query path streams, too: `POST /sessions/{session_id}/answers/stream` reports routing, retrieval, and budget progress as Server-Sent Events before streaming the generated answer token-by-token, sharing the same object-level authorization and the same `chat:{user_id}`/global quota as the JSON endpoint it sits beside.

### Hardened, not just functional

The most recent batch of work (Task #193) closed five findings a security review had recorded but not yet fixed: a global hourly chat quota shared across every account, so creating more accounts can't buy more budget; a request body-size ceiling enforced before any parsing happens; trusted-proxy-aware client-address resolution for auth's rate limiter; a security response-header baseline (HSTS, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`) on every response; and structured logging for authorization denials and rate-limit refusals that never logs a token or a request body. Each one shipped with a test proving it — the full checklist and what's still ahead on the security side is [docs/security/SECURITY.md](docs/security/SECURITY.md).

### Release history

| Version | Date | What it shipped |
|---|---|---|
| `v1.0.0` | 2026-09-03 | The first tagged release: a full RAG pipeline, a full MAG memory system, three of CAG's nine techniques, and all three cross-paradigm synthesis pairings |
| `v1.0.1`–`v1.0.5` | 2026-09-03 – 2026-09-06 | Five documentation-accuracy hotfixes |
| `v1.1.0` | 2026-09-15 | CAG's remaining six techniques, the full orchestration meta-layer, and the Unified API's session-scoped per-query endpoint |
| `v1.2.0` | 2026-09-15 | The Unified API hardening batch above, plus a documentation pass that stopped naming an unbuilt frontend stack as if it were a decision this project had made |
| `v1.2.1` | 2026-09-15 | Rewrote this README as a top-down project narrative |
| `v1.3.0` | 2026-09-16 | Freshness-routed ingestion (`POST /data-sources`, `GET /data-sources/jobs/{task_id}`), served through this project's first background-worker subsystem |

## Where to go next

[CLAUDE.md](CLAUDE.md) states the rules that don't bend in this codebase — database-first schema changes, spec-first testing, the hexagonal/CQRS backend split, the security checklist — and points at the document behind each one. [docs/README.md](docs/README.md) indexes the full documentation tree: every architecture deep-dive, the ADRs behind each technology choice, the database schema, and the original source material all of it was synthesized from. Start at whichever one matches what you're trying to do — both are written for a human reading this front door and an AI agent about to make a change alike.

## What's honestly still ahead

The Unified API milestone is now fully built: session-scoped answers, abuse hardening, freshness-routed ingestion, and streaming answers all ship as HTTP endpoints, proven against real infrastructure and a real httpx streaming client rather than a frontend that doesn't exist yet. What's left is a frontend (no framework has been chosen — the technology choice is deliberately deferred until that work actually begins, rather than named in advance for code that doesn't exist) and the Kubernetes and production-deployment layer. None of that is hidden in the docs that describe it; where the line between built and not-yet-built matters, this project says so plainly rather than letting intent read as fact.
