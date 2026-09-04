# CLAUDE.md — Unified RAG × CAG × MAG AI System

This repository is building a full-stack AI system that treats retrieval, caching, and memory as three separate, complementary answers to the same underlying question — where should a given piece of knowledge live so the model can use it well — rather than picking one paradigm and forcing everything through it. RAG reaches outside the model for knowledge that changes on its own schedule; CAG freezes knowledge that barely changes into the model's own KV cache so reusing it costs almost nothing; MAG keeps track of what a session itself has learned, so the system gets more useful the longer it talks to the same user. An orchestration layer sits above all three and decides, per query and per data source, which paradigm actually answers — because no single paradigm can see what the other two already know. This file is the entry point for anyone or anything working in this codebase: it states the rules that don't bend, points at the document that explains each one in full, and maps the rest of `docs/` so nothing has to be found by guessing. A full RAG pipeline, a full MAG memory system, three of CAG's nine techniques, and all three cross-paradigm synthesis pairings (RAG+CAG, RAG+MAG, CAG+MAG) are built, tested — 591 unit tests and 205 integration tests, all passing against real infrastructure (PostgreSQL, Qdrant, Redis, Neo4j via TestContainers, and real small HF models where a technique's own correctness depends on a real forward pass) — and live-measured, with a narrative report for every batch under `evaluation/reports/`. What's still ahead is real too, and named plainly rather than left implicit: the orchestration meta-layer's own Paradigm Router, Context Budget Allocator, and Latency-Adaptive Fallback Cascade; the six CAG techniques that need real GPU/vLLM serving to measure honestly (Eviction, PagedAttention, Cache-Aware Batching, Hybrid Offloading, Multi-Turn Caching, and CAG's own Combinations); and the frontend, background workers, and Kubernetes deployment layers. Where the line between built and not-yet-built matters, the linked deep-dive docs say so explicitly rather than letting intent read as fact.

## The three paradigms

| Paradigm | Answers | Storage | Latency | Mutability |
|----------|---------|---------|---------|------------|
| **RAG** | "What exists outside the model?" | Vector DB (Qdrant) + BM25 | 50–500ms | Instant sync |
| **CAG** | "What can fit inside the model's cache?" | GPU KV cache (vLLM) | Near-zero TTFT on a hit | Batch invalidation |
| **MAG** | "What has this session learned?" | Redis + PostgreSQL + Neo4j | 1–10ms | Continuous writes |

No paradigm is sufficient alone. Deciding which one handles which piece of knowledge, at which moment, is the orchestration layer's whole job — the five components that make that decision, the 128K context budget's split into five slices (one each for CAG, MAG, and RAG, plus a query slice and a generation reserve), and the latency-adaptive fallback cascade that tries CAG first, then MAG, then RAG, all live in [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md).

## Non-negotiable directives

Nine rules hold regardless of which part of the system a change touches. Each one is summarized here; the linked doc has the full reasoning, the schema, or the checklist behind it.

### Database-first

Schema is the source of truth: an Alembic migration lands before the application code that depends on it, the test database mirrors production's schema, constraints, and indexes exactly, and every database access goes through a repository — never raw SQL in a handler, never a string-interpolated query. The full schema across PostgreSQL, Qdrant, Redis, and Neo4j is in [docs/database/DATABASE.md](docs/database/DATABASE.md); read it before writing a migration or a repository method.

### Spec-first testing

Every public function or class gets a test — `test_*.py`, a `.spec.ts`, or a `.feature` file — written before the implementation it covers, not after. The reasoning behind the test pyramid's roughly 80/15/5 split across unit, integration, and end-to-end tests, and why MAG's write-then-read cycle and CAG's cache-hit correctness specifically need real dependencies rather than mocks, is in [docs/testing/TESTING.md](docs/testing/TESTING.md).

### Hexagonal architecture with CQRS

The backend keeps a domain layer free of framework dependencies, an application layer that orchestrates use cases, and an infrastructure layer that talks to the database, cache, and external APIs — and MAG's memory operations specifically split into separate CQRS read and write models, because memory writes are frequent and asynchronous while memory reads are complex and multi-strategy. The reasoning for that split is [docs/decisions/adr/0004-hexagonal-cqrs-for-mag.md](docs/decisions/adr/0004-hexagonal-cqrs-for-mag.md); the module layout it produces is in [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md).

### Atomic-design frontend

React components build bottom-up: atoms with no business logic, molecules that combine a few atoms, organisms like the chat interface or the document uploader, templates, then route-level pages, all typed under TypeScript strict mode. The frontend's place in the stack (React 18+, Tailwind, Zustand, SSE/WebSocket streaming) is recorded in [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md).

### Security is a checklist, not a vibe

Twenty specific controls are non-negotiable — parameterized queries, Argon2id password hashing, PostgreSQL row-level security for tenant isolation, short-lived JWTs, encryption at rest and in transit, dependency and container scanning, and more — and each one has a concrete test, scan, or configuration review that verifies it rather than a claim that it's handled. The full checklist and what each control actually prevents is in [docs/security/SECURITY.md](docs/security/SECURITY.md).

### Documentation stays in sync with the repository

The `docs/` tree mapped at the bottom of this file is meant to track the system it describes: a change that touches a documented area — a new module, a changed decision, a renamed file — updates the corresponding doc in the same pass rather than leaving it for later. [docs/README.md](docs/README.md) is the index to that tree; the autolearning mandate below is the backstop that catches drift when this rule gets missed anyway.

### The context graph

Domain → Bounded Context → Module → Class → Test File is meant to be mapped as a Mermaid diagram maintained via `graphify`, so the relationships between this codebase's pieces are queryable instead of something every session reconstructs from scratch by re-reading source files. That diagram is generated output, and it now lives at [docs/architecture/CONTEXT_GRAPH.md](docs/architecture/CONTEXT_GRAPH.md), now in its third generation: real Module, Class, and Test File nodes exist for every bounded context with real code behind it — RAG, CAG, MAG, Orchestration, Identity & Access, and the Comparison & Measurement Infrastructure context the evaluation harness earned this generation — verified directly against the repository via `graphify`'s own AST extraction rather than recalled from memory. Only the orchestration meta-layer's own router/allocator/cascade files, the six GPU/vLLM-blocked CAG techniques, and the frontend and worker layers remain the dashed, genuinely-unbuilt nodes the diagram's own "What isn't here yet" section names directly. [docs/governance/SKILL_ROUTING.md](docs/governance/SKILL_ROUTING.md) covers when to reach for `graphify` beyond a full regeneration like this one.

### Git-native workflow

Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`) using the template at [.gitmessage](.gitmessage), and git history is meant to be the audit trail for every epic, task, and decision — which only holds if a commit message is written to be read later, not just to satisfy a hook. Day-to-day branch naming sits inside a fuller Gitflow branch model — `main`, `develop`, `feature/*`, `release/*`, `hotfix/*` — adopted alongside this project's existing squash-merge-to-main convention rather than in place of it: a Gitflow feature branch is named `feature/123-short-desc`, which supersedes the `feat/123-short-desc` form this section named before Gitflow was adopted, while an ad hoc `fix/456-bug-desc` branch outside that model is unaffected, since Gitflow has no `fix/*` type for it to collide with. The full branch lifecycle and the ruling that reconciles Gitflow's merge-commit convention with squash-merging are in [docs/governance/GIT_WORKFLOW.md](docs/governance/GIT_WORKFLOW.md).

### Board automation

The six GitHub Project boards (`rag-management`, `cag-management`, `mag-management`, and the three cross-paradigm boards) run a six-stage Status field — Backlog → Ready for Dev → In Progress → Code Review → QA/Testing → Done — with issue creation, closing, and PR merges driving that field automatically rather than by hand-moved cards. Two of the six stage transitions (PR-opened → In Progress, review-requested → Code Review) sit outside what GitHub's Projects v2 API exposes for configuration and were left as a documented manual step rather than silently skipped. What was automated, what was verified by testing rather than assumed, and the exact manual steps left are in [docs/governance/KANBAN_AUTOMATION.md](docs/governance/KANBAN_AUTOMATION.md).

### The AI protocol

Load context, check the schema, write the spec, implement, test, document, commit — in that order, every time. If a schema or a spec is missing, the answer is to ask, never to guess or hallucinate a table structure or an API contract; [docs/README.md](docs/README.md) is where to start looking before asking.

## Skill routing

This environment has skills installed for the shapes of work this project already knows it will hit repeatedly, and reaching for the matching one is meant to be the default, not a step reserved for obviously large tasks. Creative or architectural work — a new feature, a new subsystem, any change to how components fit together — starts with `superpowers:brainstorming` before any code or design doc gets written, because a requirements question is always cheaper to answer before implementation starts than to unwind afterward, and every batch this project has actually built (RAG, CAG, MAG, and all three cross-paradigm pairings) went through a design spec first for exactly that reason. Security or audit work — hardening an auth flow, checking a control against the OWASP lists, verifying a data-protection claim — routes to `fullstack-e2e-security-engineer` rather than an ad hoc read-through. Any GitHub-process work, including filing one single task or bug, routes to `github-project-management-orchestrator`, which is explicitly meant to be invoked even for single work items so the Epic → Story → Task → Bug hierarchy stays consistent. Schema or database-contract design routes to `data-contract-architect` or `db-architect-holistic`, both of which produce a schema and its API contract as one package rather than as two passes that can drift apart. Questions about how this codebase's pieces connect — how a concept document maps to the architecture layer it describes, or "how does this all fit together" — route to `graphify` rather than re-reading and re-summarizing source files by hand. Bringing a `feature/*`, `release/*`, or `hotfix/*` branch up to date with the target it was cut from — a `feature/*` branch onto `develop`, a `release/*` or `hotfix/*` branch onto `main` — routes to `smart-rebase` rather than a freehand `git rebase`, because resolving a rebase conflict by hand means reconstructing two people's intent from a diff alone, and a wrong resolution pushed to a shared branch rewrites history in a way that isn't cleanly undoable. The full reasoning behind each of these rules, and why routing beats freehanding even for small tasks, is in [docs/governance/SKILL_ROUTING.md](docs/governance/SKILL_ROUTING.md).

## Token economy

This repository's own source material was already unusually large before a line of application code existed, and has only grown since — the concept documents under `docs/inputs/concepts/` run to hundreds of lines apiece, the architecture docs synthesized from them aren't much shorter, and a full `src/` implementation with its own test suite and evaluation reports now sits alongside all of it. Check `/usage` periodically during a long or research-heavy session rather than waiting for an automatic compaction to land mid-thought, and once a file has already been read or a decision already made earlier in the conversation, trust that context instead of re-reading it defensively "just to be sure." When the same procedural knowledge would otherwise need re-explaining across multiple separate conversations, that repetition is the signal to package it as a skill with `skill-creator` instead of writing it out again each time. Full policy in [docs/governance/TOKEN_ECONOMY.md](docs/governance/TOKEN_ECONOMY.md).

## Templates, never freehand

Commits, issues, and pull requests are never written freehand in this repository. Commits use the [.gitmessage](.gitmessage) template and Conventional Commits format; issues use the templates under `.github/ISSUE_TEMPLATE/` (bug, epic, story, task); pull requests use [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md). A hand-written issue "just this once" is exactly how a hierarchy or a field convention drifts out of agreement with itself — the templates exist so every entry into the system passes through the same gate.

## Keep this file honest

After any significant piece of work — standing up a new module, changing an architectural decision, adding a doc the map below doesn't yet account for — invoke the `claude-md-sync` skill to check this file and the docs it points to against the repository's actual current state. It never writes to `CLAUDE.md` or anything under `docs/` on its own; it produces an itemized list of proposed edits, each naming a file, a location, and the piece of git history that motivates the change, for a human to review and approve one item at a time. Full contract in [docs/governance/AUTOLEARNING.md](docs/governance/AUTOLEARNING.md).

## Documentation map

One line per documentation file under `docs/` — `docs/superpowers/` is deliberately excluded, here and from `docs/README.md`'s own map, because its two files are SDD process scaffolding documenting how this documentation set got built rather than documentation of the system itself — grouped the same way [docs/README.md](docs/README.md) groups them; that index explains the folder structure itself and why it's organized by concern, and this is the flat reference for finding a specific file fast.

### architecture/

- [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md) — the layered system design, the orchestration meta-layer, context-budget slicing, the fallback cascade, the full technology stack with cost estimates, and the Phase 1 module blueprint, kept current against what's actually built after every batch.
- [docs/architecture/RAG.md](docs/architecture/RAG.md) — the five-stage RAG pipeline, six chunking strategies, HyDE, reranking, CRAG, Self-RAG, and the technique compatibility matrix, with every concept's real implementation, real tests, and real measured numbers pointed to directly.
- [docs/architecture/CAG.md](docs/architecture/CAG.md) — the nine CAG techniques mapped onto the serving pipeline: eviction, compression, offloading, speculative decoding, and the attention-compatibility split ADR-0003 establishes; three of the nine (alternative-attention compatibility, KV cache compression, speculative decoding) are built and live-measured, the rest wait on GPU/vLLM serving.
- [docs/architecture/MAG.md](docs/architecture/MAG.md) — the three-tier memory hierarchy, episodic and semantic memory, six retrieval strategies, memory graphs, gating, and evolution — all seven batches built, tested, and live-measured, including the places real measurement corrected what the concept source assumed.
- [docs/architecture/CONTEXT_GRAPH.md](docs/architecture/CONTEXT_GRAPH.md) — the Domain → Bounded Context → Module → Class → Test File graph generated via `graphify`, with real Module/Class/Test File nodes now populated for every bounded context that has real code.

### decisions/adr/

- [docs/decisions/adr/0001-vllm-over-sglang.md](docs/decisions/adr/0001-vllm-over-sglang.md) — why vLLM is the primary serving engine over SGLang.
- [docs/decisions/adr/0002-qdrant-over-milvus.md](docs/decisions/adr/0002-qdrant-over-milvus.md) — why Qdrant is the vector database over Milvus and Weaviate.
- [docs/decisions/adr/0003-standard-attention-cache-optimization.md](docs/decisions/adr/0003-standard-attention-cache-optimization.md) — why this project uses standard attention with cache optimization instead of alternative attention.
- [docs/decisions/adr/0004-hexagonal-cqrs-for-mag.md](docs/decisions/adr/0004-hexagonal-cqrs-for-mag.md) — why MAG's memory operations split into separate CQRS read and write models.
- [docs/decisions/adr/0005-langchain-langgraph-rag-orchestration.md](docs/decisions/adr/0005-langchain-langgraph-rag-orchestration.md) — superseded by ADR-0006; the original, never-followed decision to use LangChain for RAG pipelines and LangGraph for CRAG and Self-RAG, kept as the historical record.
- [docs/decisions/adr/0006-hand-rolled-rag-orchestration.md](docs/decisions/adr/0006-hand-rolled-rag-orchestration.md) — why RAG orchestration is hand-rolled Python under this project's own hexagonal layering instead, recorded honestly as a decision made after the fact with no original rationale to reconstruct.
- [docs/decisions/adr/TEMPLATE.md](docs/decisions/adr/TEMPLATE.md) — the blank Context/Decision/Consequences template for recording a new architectural decision.

### database/

- [docs/database/DATABASE.md](docs/database/DATABASE.md) — the schema across PostgreSQL, Qdrant, Redis, and Neo4j, and how the four stores stay in sync without a shared transaction.

### security/

- [docs/security/SECURITY.md](docs/security/SECURITY.md) — the 20-point security checklist, what each control actually prevents, and how it gets verified.
- [docs/security/SECRETS_MANAGEMENT.md](docs/security/SECRETS_MANAGEMENT.md) — the real protocol for this project's API keys: where they live, rotation, leak response, and the working pre-commit hook (`.githooks/pre-commit`) that blocks a commit containing one.

### testing/

- [docs/testing/TESTING.md](docs/testing/TESTING.md) — the test pyramid, why MAG and CAG specifically need real-dependency integration tests, and the performance targets tied to the fallback cascade's own timeouts.

### governance/

- [docs/governance/WRITING_STANDARDS.md](docs/governance/WRITING_STANDARDS.md) — the style standard this file and everything else under `docs/` is held to: connected prose over labeled checklists, and a citation discipline for every claim.
- [docs/governance/TOKEN_ECONOMY.md](docs/governance/TOKEN_ECONOMY.md) — the context-budget policy summarized above.
- [docs/governance/SKILL_ROUTING.md](docs/governance/SKILL_ROUTING.md) — the full skill-routing table summarized above.
- [docs/governance/AUTOLEARNING.md](docs/governance/AUTOLEARNING.md) — the `claude-md-sync` contract summarized above.
- [docs/governance/GIT_WORKFLOW.md](docs/governance/GIT_WORKFLOW.md) — the Gitflow branch model (main/develop/feature/release/hotfix) and how it reconciles with this project's squash-merge-to-main convention.
- [docs/governance/INTELLIGENT_REBASE.md](docs/governance/INTELLIGENT_REBASE.md) — the `smart-rebase` contract summarized above: reading both sides of a rebase conflict and reasoning about intent before proposing a resolution, and the hard rule that only a human ever finalizes the rebase or force-pushes.
- [docs/governance/KANBAN_AUTOMATION.md](docs/governance/KANBAN_AUTOMATION.md) — what the GitHub API can and can't configure on the six project boards' Workflows panel, the six-stage Status field expansion actually applied, and the exact manual steps left for the transitions the API can't reach.

### evaluation/

- [docs/evaluation/COMPARISON_METHODOLOGY.md](docs/evaluation/COMPARISON_METHODOLOGY.md) — the self-versus-self ablation design for proving RAG/CAG/MAG's actual effect, and the model roster (self-hosted Gemma 4 and Qwen3.8-27B; API-based DeepSeek V4, Claude, and Gemini) behind the "Served models" table in `docs/architecture/OVERVIEW.md`.
- [docs/evaluation/quantitative-template.md](docs/evaluation/quantitative-template.md) — the reusable token/latency/task-success comparison table.
- [docs/evaluation/qualitative-rubric.md](docs/evaluation/qualitative-rubric.md) — the four-dimension judge rubric for reproducible quality scoring.

### inputs/concepts/

- [docs/inputs/concepts/advanced_rag_concepts.md](docs/inputs/concepts/advanced_rag_concepts.md) — the source extraction behind `docs/architecture/RAG.md`.
- [docs/inputs/concepts/advanced_cag_concepts.md](docs/inputs/concepts/advanced_cag_concepts.md) — the source extraction behind `docs/architecture/CAG.md`.
- [docs/inputs/concepts/advanced_mag_concepts.md](docs/inputs/concepts/advanced_mag_concepts.md) — the source extraction behind `docs/architecture/MAG.md`.
- [docs/inputs/concepts/fullstack_unified_ai_system.md](docs/inputs/concepts/fullstack_unified_ai_system.md) — the stack-and-layout source behind `docs/architecture/OVERVIEW.md`'s technology tables and module blueprint.
- [docs/inputs/concepts/unified_rag_cag_mag_architecture.md](docs/inputs/concepts/unified_rag_cag_mag_architecture.md) — the orchestration-concepts source behind `docs/architecture/OVERVIEW.md`'s layered design and fallback cascade.
