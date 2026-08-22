# Documentation Index

This tree is organized by concern, not by document type: the deep-dive architecture docs, the decision records that justify specific technology choices, the database schema, the security model, and the testing strategy each get their own folder, and the source material each one was synthesized from lives in one place too, rather than being duplicated next to every doc that draws from it. The payoff is that a reader chasing one question — why does CAG use standard attention instead of Mamba, say — finds the deep-dive explanation in `docs/architecture/CAG.md`, the specific decision and its trade-offs in `docs/decisions/adr/0003-standard-attention-cache-optimization.md`, and the original source concept in `docs/inputs/concepts/advanced_cag_concepts.md`. Those are three different folders and three different levels of the same question, not three versions of the same answer at different lengths, and keeping the depth together by concern rather than scattering it across parallel "reference" and "source" hierarchies is what makes it possible to follow that chain at all.

Two folders don't map onto "explain a design decision" the way the rest do, and are worth calling out for what they actually govern instead. `docs/governance/` holds the rules for how this documentation set itself gets written and how a session working in this repository is expected to behave — the writing standard every other file here is held to, the token-budget discipline, the skill-routing table, and the contract for the staleness-check skill this project intends to build. `docs/inputs/concepts/` holds the five original source documents that everything under `docs/architecture/` was synthesized from; they're kept in the repository rather than discarded once the deep-dive docs existed, because this project's own citation discipline expects every claim in the architecture layer to trace back to something concrete, and these five files are that something for most of it.

What follows is one link per file below, grouped by folder, stating what each one covers and who would reach for it.

## architecture/

Each file here expands a compressed section of CLAUDE.md back out to its full reasoning, tracing every claim back to the source concept documents under `docs/inputs/concepts/` or to the ADRs under `docs/decisions/adr/`.

- [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md) — the layered system design (RAG, CAG, and MAG plus the orchestration meta-layer above them), the five orchestration components, the 128K context-budget slicing, the latency-adaptive fallback cascade, three named cross-paradigm synthesis techniques, the full technology stack with hardware and cloud-cost estimates, deviation options, and the Phase 1 module blueprint. Reach for this for the system-wide picture or the stack tables.
- [docs/architecture/RAG.md](docs/architecture/RAG.md) — the five-stage RAG pipeline in full: six chunking strategies and their tradeoffs, the distinction between Query Expansion and Multi-Query Retrieval, HyDE's exact mechanic, the reranker taxonomy, CRAG and Self-RAG, the zero-conflict compatibility matrix, and the four-phase implementation roadmap. Reach for this when extending retrieval, chunking, or reranking.
- [docs/architecture/CAG.md](docs/architecture/CAG.md) — the nine CAG techniques mapped onto the serving pipeline's five stages: eviction algorithms, compression methods, the Hybrid Memory Offloading tier, multi-turn caching, speculative decoding variants, PagedAttention versus vAttention, and the corrected alternative-attention compatibility split that ADR-0003 establishes. Reach for this when working on caching, serving, or KV-cache behavior.
- [docs/architecture/MAG.md](docs/architecture/MAG.md) — the nine MAG concepts: the three-tier memory hierarchy, episodic memory's five defining properties, semantic memory and consolidation, six retrieval strategies, memory graphs and spreading activation over Neo4j, six gating strategies, the four memory-evolution operations, and procedural memory. Reach for this when working on memory, session state, or the Neo4j graph schema.
- [docs/architecture/CONTEXT_GRAPH.md](docs/architecture/CONTEXT_GRAPH.md) — the Domain → Bounded Context → Documentation Concern graph this `docs/` tree resolves to today, generated via `graphify` over the full documentation set, plus the Module → Class → Test File levels it names but leaves unpopulated until Phase 1 implementation begins. Reach for this to see how the paradigm docs, ADRs, and source concepts actually cite each other without tracing every reference by hand.

## decisions/adr/

Each ADR records one architectural decision as Context, Decision, and Consequences — the forces at play, what was chosen, and what was traded away to choose it.

- [docs/decisions/adr/0001-vllm-over-sglang.md](docs/decisions/adr/0001-vllm-over-sglang.md) — why vLLM is the primary serving engine over SGLang, and where SGLang remains an acceptable choice for specific agent workflows.
- [docs/decisions/adr/0002-qdrant-over-milvus.md](docs/decisions/adr/0002-qdrant-over-milvus.md) — why Qdrant is the vector database over Milvus and Weaviate, and what large-scale distributed-deployment capability that choice gives up.
- [docs/decisions/adr/0003-standard-attention-cache-optimization.md](docs/decisions/adr/0003-standard-attention-cache-optimization.md) — the commitment to standard attention with cache optimization (Path A) over alternative attention (Linear, Mamba), including the precise four-of-eight CAG-technique compatibility split this project relies on.
- [docs/decisions/adr/0004-hexagonal-cqrs-for-mag.md](docs/decisions/adr/0004-hexagonal-cqrs-for-mag.md) — why MAG's memory operations split into separate CQRS read and write models within the project's Hexagonal Architecture, and the synchronization cost that split carries.
- [docs/decisions/adr/0005-langchain-langgraph-rag-orchestration.md](docs/decisions/adr/0005-langchain-langgraph-rag-orchestration.md) — why LangChain handles linear RAG pipeline assembly while LangGraph is reserved specifically for CRAG and Self-RAG's graph-based control flow.
- [docs/decisions/adr/TEMPLATE.md](docs/decisions/adr/TEMPLATE.md) — the blank Context/Decision/Consequences template for recording any future architectural decision as a new ADR.

## database/

- [docs/database/DATABASE.md](docs/database/DATABASE.md) — the schema across all four stores: PostgreSQL's seven tables (identity and session tracking, MAG's long-term memory, the RAG document pipeline), Qdrant's three collections and HNSW index parameters, Redis's key patterns and pub/sub channel, Neo4j's nodes and edges, and how the orchestration layer keeps the four stores in sync without a shared transaction. Reach for this before writing a migration or a repository method.

## security/

- [docs/security/SECURITY.md](docs/security/SECURITY.md) — CLAUDE.md's 20-point security checklist rewritten as what each control actually prevents and the concrete test, scan, or configuration review that will verify it once there's a system to check. Reach for this before implementing auth, tenant isolation, rate limiting, or any data-protection control.

## testing/

- [docs/testing/TESTING.md](docs/testing/TESTING.md) — why the test pyramid is shaped 80/15/5 for this system, why MAG's write-then-read cycle and CAG's cache-hit correctness specifically need real-dependency integration tests rather than mocks, the four end-to-end critical paths, and the performance targets tied to the fallback cascade's own timeouts. Reach for this before writing the first test in a new area.

## governance/

- [docs/governance/WRITING_STANDARDS.md](docs/governance/WRITING_STANDARDS.md) — the binding style standard every file under `docs/` and CLAUDE.md itself is held to: connected prose over labeled checklists, tables reserved for genuinely tabular content, and a citation discipline requiring every claim about behavior or performance to trace to a test, a source document, or an explicit "design intent, not yet verified" flag.
- [docs/governance/TOKEN_ECONOMY.md](docs/governance/TOKEN_ECONOMY.md) — the policy for managing context budget in a repository whose source documents run to hundreds of lines apiece: checking `/usage` before compaction forces the issue, trusting context already established earlier in a session instead of defensively re-reading, and the threshold at which repeated procedural work earns its own skill.
- [docs/governance/SKILL_ROUTING.md](docs/governance/SKILL_ROUTING.md) — which installed skill owns which recurring shape of work (brainstorming for anything creative or architectural, the security engineer for audits, the GitHub PM orchestrator for issue and board work, the schema architects for database and API-contract design, graphify for "how does this connect" questions) and why reaching for the matching skill is meant to be the default rather than a freehand attempt.
- [docs/governance/AUTOLEARNING.md](docs/governance/AUTOLEARNING.md) — the contract for `claude-md-sync`, the not-yet-built skill that will check CLAUDE.md and its linked docs against the repository's actual state: when it runs, what counts as a mismatch worth reporting, the itemized-proposal format it must produce, and the hard rule that it only ever proposes edits for a human to approve, never applies them itself.
- [docs/governance/GIT_WORKFLOW.md](docs/governance/GIT_WORKFLOW.md) — the full Gitflow branch model (`main`, `develop`, `feature/*`, `release/*`, `hotfix/*`), the ruling that reconciles Gitflow's merge-commit convention with this project's existing squash-merge-to-main rule, versioning and tagging on `main`, and what adopting the model means for a repository where only `main` exists today.
- [docs/governance/INTELLIGENT_REBASE.md](docs/governance/INTELLIGENT_REBASE.md) — the contract for `smart-rebase`, the skill that helps bring a `feature/*`, `release/*`, or `hotfix/*` branch up to date with its Gitflow target: what "intelligent" means (reading both sides of a conflict and reasoning about intent before proposing a resolution), what it does when it isn't confident, and the hard rule that it only ever proposes resolutions for a human to approve, never finalizes a rebase or force-pushes on its own.

## evaluation/

These files are the ablation-study methodology used to prove, rather than assert, that RAG/CAG/MAG earn their complexity — self-hosted model versus itself, with each paradigm switched on and off, not model-versus-model.

- [docs/evaluation/COMPARISON_METHODOLOGY.md](docs/evaluation/COMPARISON_METHODOLOGY.md) — why the comparison is self-versus-self, the model roster (Gemma 4 and Qwen3.8-27B self-hosted; DeepSeek V4, Claude, and Gemini as API reference/judge models) with cited specs, and how a run's results feed back into a GitHub issue's Definition of Done. Reach for this before running any comparison.
- [docs/evaluation/quantitative-template.md](docs/evaluation/quantitative-template.md) — the reusable token-count/latency/task-success table, one filled-in copy per model per Story.
- [docs/evaluation/qualitative-rubric.md](docs/evaluation/qualitative-rubric.md) — the four-dimension judge rubric (coherence, relevance, completeness, groundedness) that makes a quality comparison reproducible instead of a one-time impression.

## inputs/concepts/

These five files are the original source material every deep-dive under `docs/architecture/` was synthesized from, kept in the repository rather than discarded so that any claim in the architecture docs can be traced back to where it came from.

- [docs/inputs/concepts/advanced_rag_concepts.md](docs/inputs/concepts/advanced_rag_concepts.md) — the source extraction behind `docs/architecture/RAG.md`: all nine RAG concepts, their pairwise compatibility matrix, and the implementation roadmap.
- [docs/inputs/concepts/advanced_cag_concepts.md](docs/inputs/concepts/advanced_cag_concepts.md) — the source extraction behind `docs/architecture/CAG.md`: all nine CAG concepts, their compatibility matrix, and the implementation roadmap.
- [docs/inputs/concepts/advanced_mag_concepts.md](docs/inputs/concepts/advanced_mag_concepts.md) — the source extraction behind `docs/architecture/MAG.md`: all nine MAG concepts, their compatibility matrix, and the implementation roadmap.
- [docs/inputs/concepts/fullstack_unified_ai_system.md](docs/inputs/concepts/fullstack_unified_ai_system.md) — the stack-and-repository-layout source behind `docs/architecture/OVERVIEW.md`'s technology tables, Phase 1 module blueprint, hardware requirements, and alternative-stack options.
- [docs/inputs/concepts/unified_rag_cag_mag_architecture.md](docs/inputs/concepts/unified_rag_cag_mag_architecture.md) — the orchestration-concepts source behind `docs/architecture/OVERVIEW.md`'s layered architecture, five meta-layer components, context-budget slicing, latency cascade, and named synthesis techniques.
