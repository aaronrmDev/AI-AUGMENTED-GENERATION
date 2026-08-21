# CLAUDE.md Governance Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic root `CLAUDE.md` with a lean, pointer-based entry point plus a categorized `docs/` knowledge base that restores the technical depth dropped from the five source concept documents, establishes a humanized writing standard, and adds template and autolearning governance.

**Architecture:** A `docs/` tree organized by concern (architecture, decisions, database, security, testing, governance) replaces one long file. Root `CLAUDE.md` shrinks to identity, condensed rules, and a pointer map. A new on-demand skill (`claude-md-sync`) lets the agent check staleness later. This is a documentation-only change — no application code, no `src/` scaffolding, no live GitHub configuration.

**Tech Stack:** Markdown, Git, Claude Code skills (`skill-creator`, `github-project-management-orchestrator`, `graphify`). No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-08-21-claude-md-governance-architecture-design.md`

## Global Constraints

- **Writing standard applies to every new/rewritten doc:** prose paragraphs that let a reader work out what/who/why/when/where/how/how-much naturally — never a labeled checklist walking through those categories. Tables/bullets only for genuinely tabular data (versions, schema fields, comparisons). Assume a competent reader new to this codebase; define terms on first use; use a concrete example for dense concepts. Any claim about behavior, performance, or a tradeoff must cite a test, cite the source concept docs, or be flagged "design intent, not yet verified." (Full detail is authored in Task 2 as `docs/governance/WRITING_STANDARDS.md`; every later content task must follow it.)
- **No placeholders in any deliverable file:** no "TBD", no empty sections, no unresolved links.
- **No `.claude/settings.json` changes** anywhere in this plan — token-economy guidance is documentation only.
- **No `src/`, `tests/`, `docker/`, `k8s/` scaffolding** — the Phase 1 module blueprint is documented in `OVERVIEW.md`, not created as directories.
- **No live GitHub configuration** — `.github/` templates are created as files only; there is no remote to configure against.
- **Issue and PR templates are never hand-authored** — they are produced by invoking the `github-project-management-orchestrator` skill (Task 7).
- **The `claude-md-sync` skill must never write or commit changes on its own** — it only proposes edits for human approval.
- **Every path referenced anywhere (root CLAUDE.md's doc map, `docs/README.md`) must resolve to a real file** — verified explicitly in Task 20.
- **ADR-003's corrected compatibility claim** (alternative attention conflicts with only 4 of 8 CAG techniques — Eviction, Compression, PagedAttention, Speculative Decoding; compatible with Prefix Caching, Hybrid Offloading, Multi-Turn Caching, Cache-Aware Batching) must be stated identically everywhere it appears (Task 1's ADR-003 file and Task 14's `CAG.md`).
- **React version is corrected to "React 18+"** everywhere it's mentioned (source docs never say React 19 — this was an unattributed error in the current file).

---

## Phase A — Governance and Structure (sequential)

### Task 1: Extract and correct the five ADRs

**Files:**
- Create: `docs/decisions/adr/0001-vllm-over-sglang.md`
- Create: `docs/decisions/adr/0002-qdrant-over-milvus.md`
- Create: `docs/decisions/adr/0003-standard-attention-cache-optimization.md`
- Create: `docs/decisions/adr/0004-hexagonal-cqrs-for-mag.md`
- Create: `docs/decisions/adr/0005-langchain-langgraph-rag-orchestration.md`
- Create: `docs/decisions/adr/TEMPLATE.md`
- Read: `CLAUDE.md` (current §11, "Key Architectural Decisions (ADRs)")
- Read: `docs/inputs/concepts/advanced_cag_concepts.md` (for the ADR-003 compatibility matrix correction)

**Interfaces:**
- Produces: five ADR files at the exact paths above, each following Context → Decision → Consequences prose structure (no labeled bullet fields — write connected paragraphs). Also produces `TEMPLATE.md`, a blank ADR template other future ADRs will copy.

- [ ] **Step 1: Read the current ADR section**

Read `CLAUDE.md` section 11 in full (the five ADRs: vLLM over SGLang, Qdrant over Milvus/Weaviate, Standard Attention + Cache Optimization, Hexagonal + CQRS for MAG, LangChain + LangGraph for RAG Orchestration).

- [ ] **Step 2: Read the CAG compatibility matrix**

Read `docs/inputs/concepts/advanced_cag_concepts.md` in full and locate the compatibility matrix covering Alternative Attention (Mamba, Linear Attention) against the eight CAG techniques (Prefix Caching, PagedAttention, Eviction, Compression, Hybrid Offloading, Multi-Turn Caching, Speculative Decoding, Cache-Aware Batching). Confirm which four are marked incompatible (⚠️/❌) and which four are marked compatible (✅).

- [ ] **Step 3: Write `docs/decisions/adr/0001-vllm-over-sglang.md`**

Rewrite the vLLM-over-SGLang ADR as prose: a paragraph of Context (why a serving engine decision was needed), a paragraph of Decision (vLLM is primary; SGLang acceptable for specific agent workflows), and a paragraph of Consequences (what this locks in — prefix caching, ecosystem, docs). Status: Accepted.

- [ ] **Step 4: Write `docs/decisions/adr/0002-qdrant-over-milvus.md`**

Same prose structure for the Qdrant decision (simpler ops, Rust performance, hybrid search support, local dev ease vs. Milvus/Weaviate).

- [ ] **Step 5: Write `docs/decisions/adr/0003-standard-attention-cache-optimization.md`**

Same prose structure, but with the **corrected** compatibility claim from Step 2: state plainly that alternative attention is incompatible with four specific CAG techniques (name them) because they depend on manipulating a KV cache that alternative attention architectures don't produce, and compatible with the other four (name them) because those techniques operate above the attention mechanism itself. Do not claim it conflicts with "all" cache-based methods — that is the defect being fixed.

- [ ] **Step 6: Write `docs/decisions/adr/0004-hexagonal-cqrs-for-mag.md`**

Same prose structure for the Hexagonal + CQRS decision (MAG writes are frequent/async episodic storage; reads are complex/multi-strategy; CQRS allows independent optimization).

- [ ] **Step 7: Write `docs/decisions/adr/0005-langchain-langgraph-rag-orchestration.md`**

Same prose structure for the LangChain/LangGraph decision (mature ecosystem, graph-based agent loops map to CRAG/Self-RAG).

- [ ] **Step 8: Write `docs/decisions/adr/TEMPLATE.md`**

A blank template with the same three-paragraph structure (Context / Decision / Consequences) as placeholder prompts, e.g. "Context: what decision needed to be made and why. Decision: what was decided, stated plainly. Consequences: what this commits the project to, including tradeoffs accepted." Include a one-line header format: `# ADR-NNNN: <title>` plus a `Status:` line (Proposed / Accepted / Superseded).

- [ ] **Step 9: Verify no placeholder text remains**

Confirm none of the six new files contain "TBD" or unresolved brackets, and that ADR-0003 explicitly names all four incompatible and all four compatible techniques.

- [ ] **Step 10: Commit**

```bash
git add docs/decisions/adr/
git commit -m "docs: extract ADRs from CLAUDE.md into standalone files, fix ADR-003 compatibility claim"
```

---

### Task 2: Write the writing standard

**Files:**
- Create: `docs/governance/WRITING_STANDARDS.md`

**Interfaces:**
- Produces: `docs/governance/WRITING_STANDARDS.md`, which every subsequent content-writing task in this plan must read and follow.

- [ ] **Step 1: Write the document**

Cover, in prose (this document is itself the first proof of the standard it describes):

- The core rule: every section describing a real thing (feature, decision, component) must let a reader work out what it is, who it's for, why it exists, when it applies, where it lives, how it works, and what it costs — through natural paragraphs, never a labeled checklist walking through those categories one by one. Give a short "before/after" example: show one paragraph written the labeled-checklist way and the same content rewritten as connected prose, so the rule is unambiguous rather than abstract.
- When tables/bullets are still the right tool (version numbers, schema fields, side-by-side comparisons) versus when they're a crutch (explaining how something works or why a decision was made).
- The tone rule: assume a competent reader new to this codebase, define unfamiliar terms on first use, use a concrete example or analogy for dense concepts, and write the way you'd answer a colleague's real question.
- The citation-discipline rule: any claim about behavior, performance, or a tradeoff must trace to a test, a citation to the source concept documents, or an explicit "design intent, not yet verified" flag — never stated as settled fact if it isn't.
- A short closing note that this standard applies to every document under `docs/` and to `CLAUDE.md` itself, including future edits.

- [ ] **Step 2: Verify no placeholder text remains and the before/after example is concrete**

Re-read the file; confirm the before/after example genuinely demonstrates the rule (not a generic description of it) and no section is left abstract without a worked example.

- [ ] **Step 3: Commit**

```bash
git add docs/governance/WRITING_STANDARDS.md
git commit -m "docs: add humanized writing standard for all project documentation"
```

---

### Task 3: Write the token-economy policy

**Files:**
- Create: `docs/governance/TOKEN_ECONOMY.md`
- Read: `docs/governance/WRITING_STANDARDS.md` (follow it)

**Interfaces:**
- Produces: `docs/governance/TOKEN_ECONOMY.md`

- [ ] **Step 1: Write the document**

Cover in prose, following the writing standard from Task 2:

- Why token economy matters for this project specifically: the codebase's own source material (five ~900-1000 line concept documents) is large enough that careless re-reading burns context fast, and this project will grow.
- The `/usage` habit: check context usage periodically during long or research-heavy sessions rather than waiting for an automatic compaction to hit; treat rising usage as a signal to wrap up a sub-task and commit progress rather than pushing further into one giant context.
- Preferring compaction over re-reading: once information has been established in a conversation (a file's contents, a decision already made), don't re-read the same file defensively — trust the existing context or ask for a targeted diff instead of a full re-read.
- When to reach for `skill-creator`: when a piece of reusable procedural knowledge (a recurring check, a recurring generation task) would otherwise have to be re-explained inline across multiple conversations, that's the signal to package it as a skill instead — it pays for itself once it saves more tokens across future invocations than it cost to build. Reference `claude-md-sync` (Task 11) as the concrete example already in this repo.
- Explicitly state this is a documented policy only — no `.claude/settings.json` permission changes accompany it.

- [ ] **Step 2: Verify no placeholder text remains**

- [ ] **Step 3: Commit**

```bash
git add docs/governance/TOKEN_ECONOMY.md
git commit -m "docs: add token-economy policy"
```

---

### Task 4: Write the skill-routing policy

**Files:**
- Create: `docs/governance/SKILL_ROUTING.md`
- Read: `docs/governance/WRITING_STANDARDS.md` (follow it)

**Interfaces:**
- Produces: `docs/governance/SKILL_ROUTING.md`

- [ ] **Step 1: Write the document**

Cover in prose, following the writing standard:

- Creative or architectural work (new features, new subsystems, anything that changes how components fit together) starts with `superpowers:brainstorming` before any implementation — explain why: it front-loads the questions that are expensive to answer after code exists.
- Security or audit work (hardening, OWASP review, pentest-style checks, anything touching auth/data-protection) routes to `fullstack-e2e-security-engineer`.
- Any task, issue, board, or GitHub-process work routes to `github-project-management-orchestrator` — explain that this includes template creation itself (as this plan's own Task 7 does).
- Schema design or database-contract work routes to `data-contract-architect` or `db-architect-holistic`.
- Mapping, graphing, or "how does this all connect" questions route to `graphify`.
- Close with the general principle this all serves: specialized skills exist because they encode standards this project has committed to following (item 3 from the original request — never freehand what a standard process already covers) — routing to them is the default, not an exception.

- [ ] **Step 2: Verify no placeholder text remains**

- [ ] **Step 3: Commit**

```bash
git add docs/governance/SKILL_ROUTING.md
git commit -m "docs: add skill-routing policy"
```

---

### Task 5: Write the autolearning contract

**Files:**
- Create: `docs/governance/AUTOLEARNING.md`
- Read: `docs/governance/WRITING_STANDARDS.md` (follow it)

**Interfaces:**
- Produces: `docs/governance/AUTOLEARNING.md`, which Task 11 (building the actual `claude-md-sync` skill) must implement faithfully.

- [ ] **Step 1: Write the document**

Cover in prose, following the writing standard:

- What staleness means for this repo: CLAUDE.md and its linked docs describe intended structure, rules, and architecture; as real commits land, that description can drift from what's actually true. Explain this is especially likely once Phase 1 code work starts.
- When `claude-md-sync` runs: on-demand, either invoked directly or when the agent judges that a significant piece of work just finished (explain what "significant" means here — a new module, a changed architectural decision, a new doc — not every small commit).
- What it checks: recent git history since the last known sync point, the current `CLAUDE.md`, and the current `docs/` tree, looking for concrete mismatches (a rule that no longer matches practice, a doc map entry pointing at a moved/renamed file, an architecture description that no longer matches the code).
- What it produces: a specific, itemized list of proposed edits with file paths and reasoning — never a vague "some things might be stale."
- The hard constraint, stated plainly: it never writes or commits anything itself. A human reviews and approves every proposed edit before it's applied. Explain why this matters — an automated process rewriting its own governing document without review is exactly the kind of silent drift this mechanism exists to prevent.

- [ ] **Step 2: Verify no placeholder text remains**

- [ ] **Step 3: Commit**

```bash
git add docs/governance/AUTOLEARNING.md
git commit -m "docs: add autolearning contract for the claude-md-sync skill"
```

---

### Task 6: Add the commit message template

**Files:**
- Create: `.gitmessage`

**Interfaces:**
- Produces: `.gitmessage` at the repo root, wired into local git config.

- [ ] **Step 1: Write `.gitmessage`**

```
# <type>(<scope>): <subject>  (max 72 chars, imperative mood, no trailing period)
#
# <body>
# Explain WHAT changed and WHY, not how — the diff already shows how.
# Wrap at 72 chars. Leave a blank line between subject and body.
#
# <footer>
# Reference issues/ADRs if relevant, e.g. "Refs ADR-0003" or "Closes #12".
#
# --- Conventional Commit types (from CLAUDE.md) ---
# feat:     a new feature
# fix:      a bug fix
# docs:     documentation only changes
# test:     adding or correcting tests
# refactor: code change that neither fixes a bug nor adds a feature
# perf:     a performance improvement
# chore:    tooling, dependencies, or maintenance
```

- [ ] **Step 2: Wire it into git config**

```bash
git config commit.template .gitmessage
```

This is a local, reversible repo-level config change (not global) — confirm it applied:

```bash
git config --get commit.template
```

Expected output: `.gitmessage`

- [ ] **Step 3: Commit**

```bash
git add .gitmessage
git commit -m "chore: add conventional-commit message template"
```

---

### Task 7: Generate GitHub issue and PR templates via github-project-management-orchestrator

**Files:**
- Create: `.github/ISSUE_TEMPLATE/epic.md`
- Create: `.github/ISSUE_TEMPLATE/story.md`
- Create: `.github/ISSUE_TEMPLATE/task.md`
- Create: `.github/ISSUE_TEMPLATE/bug.md`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`

**Interfaces:**
- Produces: the five files above.

- [ ] **Step 1: Invoke the skill**

Invoke the `github-project-management-orchestrator` skill (via the Skill tool) and ask it specifically to produce standards-compliant GitHub issue templates for an Epic → Story → Task → Bug hierarchy plus a pull request template, appropriate for a pre-code repository that has no GitHub remote yet (so the templates must be self-contained files, not live-configured Projects V2 fields). Do not hand-author these templates — follow whatever field structure and conventions the skill specifies.

- [ ] **Step 2: Verify the five files exist and contain no placeholder text**

```bash
ls .github/ISSUE_TEMPLATE/ .github/PULL_REQUEST_TEMPLATE.md
```

Confirm none of the five files contain unresolved "TBD" or bracketed instructions meant for the skill itself rather than for a future issue/PR author.

- [ ] **Step 3: Commit**

```bash
git add .github/
git commit -m "docs: add GitHub issue and PR templates via github-project-management-orchestrator"
```

---

### Task 8: Write the database documentation

**Files:**
- Create: `docs/database/DATABASE.md`
- Read: `CLAUDE.md` (current §6, "Database Schema Philosophy")
- Read: `docs/governance/WRITING_STANDARDS.md` (follow it)

**Interfaces:**
- Produces: `docs/database/DATABASE.md`

- [ ] **Step 1: Read the current database section**

Read `CLAUDE.md` §6 in full (PostgreSQL tables, Qdrant collections, Redis key patterns, Neo4j nodes/edges).

- [ ] **Step 2: Write the document**

Rewrite as prose following the writing standard: explain, for each store (PostgreSQL, Qdrant, Redis, Neo4j), what it's responsible for holding and why that store specifically (not a different one) was chosen for that data, walking through the actual schema/collection/key-pattern details from the source material as part of the explanation rather than as a bare list. Keep genuinely tabular content (exact column names and types, exact key patterns) as tables — this is one of the cases where a table is the right tool — but wrap each table in prose that explains its purpose and reasoning.

- [ ] **Step 3: Verify no placeholder text remains**

- [ ] **Step 4: Commit**

```bash
git add docs/database/DATABASE.md
git commit -m "docs: extract and humanize database documentation"
```

---

### Task 9: Write the security documentation

**Files:**
- Create: `docs/security/SECURITY.md`
- Read: `CLAUDE.md` (current §4.5 "SECURITY (20-Point Checklist)" and §9 "Security Model")
- Read: `docs/governance/WRITING_STANDARDS.md` (follow it)

**Interfaces:**
- Produces: `docs/security/SECURITY.md`

- [ ] **Step 1: Read the current security sections**

Read `CLAUDE.md` §4.5 and §9 in full.

- [ ] **Step 2: Write the document**

Rewrite as prose following the writing standard. For each of the 20 checklist points, explain not just the rule but what happens if it's skipped (the concrete failure mode) and how it would actually be verified once code exists (a test, a scan tool like Trivy/Snyk, a manual review step) — this is the "everything tested, everything proved" citation discipline applied to security specifically. Cover multi-tenancy, authentication, data protection, and rate limiting from §9 the same way — as explained mechanisms, not restated bullet lists.

- [ ] **Step 3: Verify no placeholder text remains and every one of the 20 points has a stated verification mechanism**

- [ ] **Step 4: Commit**

```bash
git add docs/security/SECURITY.md
git commit -m "docs: extract and humanize security documentation"
```

---

### Task 10: Write the testing documentation

**Files:**
- Create: `docs/testing/TESTING.md`
- Read: `CLAUDE.md` (current §8, "Testing Strategy")
- Read: `docs/governance/WRITING_STANDARDS.md` (follow it)

**Interfaces:**
- Produces: `docs/testing/TESTING.md`

- [ ] **Step 1: Read the current testing section**

Read `CLAUDE.md` §8 in full (test pyramid, unit/integration/E2E/performance strategy).

- [ ] **Step 2: Write the document**

Rewrite as prose following the writing standard: explain why the test pyramid is shaped the way it is for this specific project (why unit tests dominate, why integration tests need real TestContainers instead of mocks — tie this to MAG's stateful nature and CAG's cache-correctness requirements, not just "it's best practice"), and walk through what unit/integration/E2E/performance testing each concretely means here before presenting the commands as a table.

- [ ] **Step 3: Verify no placeholder text remains**

- [ ] **Step 4: Commit**

```bash
git add docs/testing/TESTING.md
git commit -m "docs: extract and humanize testing documentation"
```

---

### Task 11: Build the claude-md-sync skill

**Files:**
- Create: `.claude/skills/claude-md-sync/SKILL.md` (and any supporting files `skill-creator` generates)
- Read: `docs/governance/AUTOLEARNING.md` (the contract this skill must implement)

**Interfaces:**
- Consumes: the contract defined in Task 5's `docs/governance/AUTOLEARNING.md`.
- Produces: a working project-level skill invocable as `claude-md-sync`.

- [ ] **Step 1: Invoke skill-creator**

Invoke the `skill-creator` skill (via the Skill tool) to create a new project-level skill named `claude-md-sync`. Provide it the full contract from `docs/governance/AUTOLEARNING.md`: on-demand trigger, inputs (recent git history, current `CLAUDE.md`, current `docs/` tree), required output (a specific itemized list of proposed edits with file paths and reasoning), and the hard constraint that it must never write or commit changes itself — only propose them for human approval. Confirm with `skill-creator`'s own process that the resulting skill is installed as a project skill (under `.claude/skills/`, committed to the repo), not a personal skill.

- [ ] **Step 2: Smoke-test the skill**

Invoke `claude-md-sync` once against the current repo state (which at this point in the plan is mid-rewrite) and confirm it produces a coherent, specific list of observations rather than an error or a vague summary. This is a smoke test, not a correctness audit — the real test is Task 20's final verification pass, once the rewrite is complete.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/claude-md-sync/
git commit -m "feat: add claude-md-sync skill for on-demand documentation staleness checks"
```

---

## Phase B — Content Depth Restoration (independent, parallelizable)

Each task in this phase sources from a different, non-overlapping subset of the five concept documents and can be executed independently of the other three.

### Task 12: Write the architecture overview

**Files:**
- Create: `docs/architecture/OVERVIEW.md`
- Read: `CLAUDE.md` (current §2 "Architecture Overview" and §3 "Technology Stack")
- Read: `docs/inputs/concepts/unified_rag_cag_mag_architecture.md` (full)
- Read: `docs/inputs/concepts/fullstack_unified_ai_system.md` (full)
- Read: `docs/governance/WRITING_STANDARDS.md` (follow it)

**Interfaces:**
- Produces: `docs/architecture/OVERVIEW.md`, which root `CLAUDE.md` (Task 18) points to for all architecture/stack depth.

- [ ] **Step 1: Read all source material**

Read both concept documents in full, plus `CLAUDE.md` §2-§3.

- [ ] **Step 2: Write the document**

Cover, as connected prose following the writing standard (tables only where genuinely tabular, e.g. the stack version table and the context-budget percentages):

- The four-layer architecture (Orchestration, MAG, CAG, RAG, over an LLM-core foundation) and why each layer sits where it does relative to the others.
- The orchestration meta-layer's five components (Paradigm Router, Context Budget Allocator, Latency-Adaptive Fallback Cascade, Sync Mixer, Freshness-Aware Data Router) — explain what each decides and why the decision belongs there rather than elsewhere.
- The context budget allocation table (CAG 40%, MAG 25%, RAG 20%, Query 10%, Reserve 5%) and the dynamic-resizing rule (if RAG isn't needed, MAG expands; if MAG is empty, CAG expands) — explain the reasoning, not just the numbers.
- The latency-adaptive fallback cascade (CAG 10ms → MAG 50ms → RAG 2s) and what happens at each tier.
- From `unified_rag_cag_mag_architecture.md`: name and explain the **Tiered Knowledge Hot-Cold Architecture** (CAG=hot, MAG=warm, RAG=cold, with promotion/demotion rules), **State-Aware RAG** (MAG enriching RAG queries), **Cache-Warmed RAG** (80/20 Pareto pre-loading), and the sync mixer's tiebreak rule (RAG wins as source of truth on conflict). These are currently unnamed/implicit in CLAUDE.md — name them explicitly here.
- The full technology stack (current CLAUDE.md §3.1-§3.8: core backend, data storage, RAG ecosystem, CAG ecosystem, MAG ecosystem, observability, infrastructure, frontend) — corrected to **"React 18+"** consistently, since neither source document mentions React 19.
- From `fullstack_unified_ai_system.md` §3: the hardware/cost tables and the alternative-stack options (Go/Rust, cloud-managed, simplified) that are currently entirely missing from CLAUDE.md — include them as a "when you might deviate from this stack" discussion, not just a raw table dump.
- From `fullstack_unified_ai_system.md` §4: the concrete module blueprint (`src/rag/{chunking,embedding,retrieval,reranking,advanced}/`, `src/cag/{cache,eviction,compression,serving,offloading}/`, `src/mag/{memory,storage,retrieval,consolidation,evolution,gating}/`, `src/orchestration/` with named files, `src/workers/`, `tests/{unit,integration,fixtures}/`, `docker/`, `k8s/helm/`, `notebooks/`) — presented explicitly as the **Phase 1 plan**, with a clear statement that these directories do not exist yet and will be created when Phase 1 implementation begins, not before.

- [ ] **Step 3: Verify no placeholder text remains and every named technique (Tiered Knowledge Hot-Cold Architecture, State-Aware RAG, Cache-Warmed RAG) is explained, not just mentioned**

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/OVERVIEW.md
git commit -m "docs: write architecture overview with full stack and orchestration depth"
```

---

### Task 13: Write the RAG deep-dive

**Files:**
- Create: `docs/architecture/RAG.md`
- Read: `CLAUDE.md` (current §5.1 "RAG (Retrieval-Augmented Generation)")
- Read: `docs/inputs/concepts/advanced_rag_concepts.md` (full)
- Read: `docs/governance/WRITING_STANDARDS.md` (follow it)

**Interfaces:**
- Produces: `docs/architecture/RAG.md`

- [ ] **Step 1: Read all source material**

Read `docs/inputs/concepts/advanced_rag_concepts.md` in full and `CLAUDE.md` §5.1.

- [ ] **Step 2: Write the document**

Cover, as connected prose following the writing standard, at minimum:

- The five pipeline stages (Pre-Processing, Pre-Retrieval, Retrieval, Post-Retrieval, Generation) and the techniques within each, carried forward from current CLAUDE.md §5.1 but explained rather than listed.
- The chunking strategy options (fixed, semantic, recursive, parent-document, sliding window, structure-aware) — explain the tradeoff each one makes, not just its name.
- The reranker taxonomy: cross-encoder vs. bi-encoder vs. LLM-based rerankers, with their actual speed/cost tradeoffs as documented in the source.
- The Multi-Query vs. Query-Expansion distinction as the source document draws it — these are easy to conflate; explain precisely how they differ.
- HyDE's actual mechanic: embedding the *hypothetical answer* the model generates, not the original question — explain why that produces better retrieval than embedding the question directly.
- The quantified per-technique impact figures documented in the source (e.g. recall/precision/token-count deltas) — use the source's actual numbers, not invented ones.
- The phased implementation roadmap (the source's beginner-to-expert progression) as a narrative path, not a bare list.
- The high-synergy technique combinations (at minimum: Hybrid Search + Reranking, Multi-Query + HyDE, Reranking + CRAG, Parent Document + Context Compression) with the reasoning for why each pair compounds, drawn from the source's compatibility analysis.

- [ ] **Step 3: Verify no placeholder text remains and the HyDE mechanic is stated correctly (hypothetical answer, not question)**

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/RAG.md
git commit -m "docs: write RAG deep-dive restoring depth from advanced_rag_concepts.md"
```

---

### Task 14: Write the CAG deep-dive

**Files:**
- Create: `docs/architecture/CAG.md`
- Read: `CLAUDE.md` (current §5.2 "CAG (Cache-Augmented Generation)")
- Read: `docs/inputs/concepts/advanced_cag_concepts.md` (full)
- Read: `docs/decisions/adr/0003-standard-attention-cache-optimization.md` (Task 1 — for the corrected compatibility claim, must match exactly)
- Read: `docs/governance/WRITING_STANDARDS.md` (follow it)

**Interfaces:**
- Consumes: the corrected ADR-003 compatibility claim from Task 1 (four incompatible, four compatible CAG techniques against alternative attention) — this document's compatibility discussion must state the same four-and-four split, not a different or vaguer version.
- Produces: `docs/architecture/CAG.md`

- [ ] **Step 1: Read all source material**

Read `docs/inputs/concepts/advanced_cag_concepts.md` in full, `CLAUDE.md` §5.2, and `docs/decisions/adr/0003-standard-attention-cache-optimization.md`.

- [ ] **Step 2: Write the document**

Cover, as connected prose following the writing standard, at minimum:

- The five pipeline stages (Scheduling, Prefill, Storage, Decoding, Architecture) carried forward from current CLAUDE.md §5.2, explained rather than listed.
- The named eviction algorithms beyond H2O/SnapKV documented in the source (at minimum: NACL, MorphKV, HASHEVICT, InfiniPot) — what problem each solves and how it differs from H2O/SnapKV.
- The named compression methods beyond KIVI/KVQuant documented in the source (at minimum: PALU, MiniCache, ShadowKV) — the mechanism each uses (e.g. SVD low-rank, cross-layer SLERP).
- The Hybrid Memory Offloading concept in full, including its named strategies documented in the source (at minimum: InfiniGen, LayerKV, INF2, KVPR, Oneiros) — this is currently an unsupported bullet in CLAUDE.md with zero content behind it; give it real substance here.
- The multi-turn caching methods documented in the source (at minimum: RocketKV-MT, KVzip, MemServe, SGLang's approach).
- The speculative decoding variants documented in the source (at minimum: Medusa, Lookahead/Prompt-Lookup Decoding).
- vAttention as an alternative to PagedAttention, and how it differs.
- The alternative-attention compatibility discussion, using the exact four-incompatible/four-compatible split from `docs/decisions/adr/0003-standard-attention-cache-optimization.md` — do not restate the old "conflicts with everything" claim.
- The high-synergy technique combinations (at minimum: Prefix Caching + PagedAttention + Cache-Aware Batching; Eviction + Compression + Hybrid Offloading; Multi-Turn Caching + Eviction + Compression) with reasoning.

- [ ] **Step 3: Verify no placeholder text remains and the compatibility claim matches ADR-0003 exactly**

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/CAG.md
git commit -m "docs: write CAG deep-dive restoring depth from advanced_cag_concepts.md"
```

---

### Task 15: Write the MAG deep-dive

**Files:**
- Create: `docs/architecture/MAG.md`
- Read: `CLAUDE.md` (current §5.3 "MAG (Memory-Augmented Generation)")
- Read: `docs/inputs/concepts/advanced_mag_concepts.md` (full)
- Read: `docs/governance/WRITING_STANDARDS.md` (follow it)

**Interfaces:**
- Produces: `docs/architecture/MAG.md`

- [ ] **Step 1: Read all source material**

Read `docs/inputs/concepts/advanced_mag_concepts.md` in full and `CLAUDE.md` §5.3.

- [ ] **Step 2: Write the document**

Cover, as connected prose following the writing standard, at minimum:

- The three memory tiers (Short-Term/Working, Medium-Term/Recall, Long-Term) and the four memory types (Episodic, Semantic, Procedural, Graph) carried forward from current CLAUDE.md §5.3, explained rather than listed.
- All six memory-gating strategies documented in the source (at minimum: Top-K, Token Budget, Hierarchical Assembly, Recency-Weighted, Task-Specific Filtering, Dynamic Re-ranking) — CLAUDE.md currently only mentions token-budget filtering; explain what each strategy optimizes for and when you'd pick it over the others.
- All four evolution operations documented in the source (Update, Invalidate, Refine, Archive) — CLAUDE.md currently drops Refine; explain what distinguishes Refine from a plain Update.
- Spreading activation as a graph-traversal retrieval strategy — currently unmentioned in CLAUDE.md; explain how it works and why it's useful for the Neo4j-backed graph memory specifically.
- The required properties of episodic memory as documented in the source — read the source carefully to find and enumerate the complete set (CLAUDE.md currently states none of them); explain why each property is required rather than just listing them.
- The high-synergy technique combinations (at minimum: Episodic + Semantic + Consolidation; Memory Hierarchy + Retrieval + Gating; Episodic + Procedural + Consolidation; Memory Graphs + Episodic + Semantic) with reasoning, carried forward and expanded from current CLAUDE.md.

- [ ] **Step 3: Verify no placeholder text remains and all six gating strategies plus all four evolution operations (including Refine) are present**

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/MAG.md
git commit -m "docs: write MAG deep-dive restoring depth from advanced_mag_concepts.md"
```

---

## Phase C — Final Assembly (sequential, depends on Phases A and B)

### Task 16: Write the docs index

**Files:**
- Create: `docs/README.md`
- Read: every file created in Phase A and Phase B (for accurate one-line descriptions)

**Interfaces:**
- Consumes: the final file set from Tasks 1-15.
- Produces: `docs/README.md`

- [ ] **Step 1: Write the document**

A short prose introduction explaining that this tree is organized by concern rather than by document type, so related depth stays together. Follow it with one line per file, grouped by folder (`architecture/`, `decisions/adr/`, `database/`, `security/`, `testing/`, `governance/`, `inputs/concepts/`). Write each line as a markdown link followed by its explanation, e.g. `- [docs/architecture/RAG.md](docs/architecture/RAG.md) — covers the RAG pipeline in depth; reach for this when extending retrieval, chunking, or reranking.` (the link target must be the exact relative path so Task 20's automated check can find it), stating what the file covers and who'd reach for it. Deliberately omit `docs/architecture/CONTEXT_GRAPH.md` from this map for now — it doesn't exist yet (it's created in Task 19) and Task 19 is responsible for adding its line to this file once it does.

- [ ] **Step 2: Verify every file mentioned actually exists**

```bash
for f in docs/architecture/OVERVIEW.md docs/architecture/RAG.md docs/architecture/CAG.md docs/architecture/MAG.md \
         docs/decisions/adr/0001-vllm-over-sglang.md docs/decisions/adr/0002-qdrant-over-milvus.md \
         docs/decisions/adr/0003-standard-attention-cache-optimization.md docs/decisions/adr/0004-hexagonal-cqrs-for-mag.md \
         docs/decisions/adr/0005-langchain-langgraph-rag-orchestration.md docs/decisions/adr/TEMPLATE.md \
         docs/database/DATABASE.md docs/security/SECURITY.md docs/testing/TESTING.md \
         docs/governance/WRITING_STANDARDS.md docs/governance/TOKEN_ECONOMY.md docs/governance/SKILL_ROUTING.md \
         docs/governance/AUTOLEARNING.md; do
  test -f "$f" && echo "OK  $f" || echo "MISSING $f"
done
```

Expected: every line prints `OK`. If any print `MISSING`, stop and fix before proceeding — a broken doc map is exactly what this task exists to prevent.

- [ ] **Step 3: Commit**

```bash
git add docs/README.md
git commit -m "docs: add docs tree index"
```

---

### Task 17: Rewrite root CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (full rewrite)
- Read: all files created in Tasks 1-16

**Interfaces:**
- Consumes: the final file set from Tasks 1-16 (every pointer in this file must reference one of those real paths).
- Produces: the rewritten root `CLAUDE.md`, roughly 200-250 lines.

- [ ] **Step 1: Write the document**

Following `docs/governance/WRITING_STANDARDS.md`, structure the new root file as:

1. An opening prose paragraph explaining what this project is and why it exists — a reader should be oriented before hitting any rule.
2. The three-paradigm identity table (RAG/CAG/MAG: what each answers, storage, latency, mutability) — the one table that stays inline.
3. Condensed non-negotiable directives (one derived from each of current §4.1-§4.9: database-first, spec TDD, hexagonal+CQRS backend, atomic-design frontend, the 20-point security checklist, auto-generated documentation, context graph maintenance, git-native workflow, the AI protocol), each as one to three sentences plus a pointer to its deep doc (`docs/security/SECURITY.md`, `docs/testing/TESTING.md`, etc.).
4. A skill-routing summary (the operational rules from `docs/governance/SKILL_ROUTING.md`, stated directly: brainstorming-first for creative work, `fullstack-e2e-security-engineer` for security work, `github-project-management-orchestrator` for any task/issue/board work, `data-contract-architect`/`db-architect-holistic` for schema work, `graphify` for mapping) plus a pointer to the full doc.
5. A token-economy summary (two to three sentences) plus a pointer to `docs/governance/TOKEN_ECONOMY.md`.
6. The template mandate: commits use `.gitmessage`, issues/PRs use the `.github/` templates — never freehand.
7. The autolearning mandate: invoke `claude-md-sync` after significant work, per `docs/governance/AUTOLEARNING.md`.
8. A doc map: one line per file under `docs/`, written the same way as Task 16's `docs/README.md` — a markdown link with the exact relative path as its target, e.g. `- [docs/security/SECURITY.md](docs/security/SECURITY.md) — the 20-point checklist and threat model.` — matching what `docs/README.md` says so the two never disagree, and so Task 20's automated link check can find every path.

- [ ] **Step 2: Verify every pointer resolves and the file has no leftover content from the old version that contradicts the new structure**

Re-run the existence check from Task 16 Step 2 against every path referenced in the new `CLAUDE.md`, plus confirm the old §1-§15 structure (the previous single-file version) has been fully replaced, not left alongside the new content.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: rewrite root CLAUDE.md as lean pointer-based entry point"
```

---

### Task 18: Rewrite the project README

**Files:**
- Modify: `README.md`
- Read: `CLAUDE.md` (rewritten, Task 17)

**Interfaces:**
- Produces: the rewritten `README.md`, replacing the current one-line "TBD" stub.

- [ ] **Step 1: Write the document**

A short, human-facing project overview following the writing standard: what this project is, who it's for, and where to go next (point to `CLAUDE.md` for AI-agent/contributor rules, and `docs/README.md` for the full documentation tree). This is the first thing anyone lands on — keep it welcoming and short, not a restatement of the architecture.

- [ ] **Step 2: Verify no placeholder text remains ("TBD" is specifically what's being replaced)**

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: write real project README, replacing TBD stub"
```

---

### Task 19: Generate the context graph via graphify

**Files:**
- Create: `docs/architecture/CONTEXT_GRAPH.md`
- Modify: `docs/README.md` (add the line Task 16 deliberately omitted)
- Modify: `CLAUDE.md` (add `docs/architecture/CONTEXT_GRAPH.md` to the doc map from Task 17, if not already present)
- Read: the entire `docs/` tree and `CLAUDE.md` (final state, all prior tasks complete)

**Interfaces:**
- Consumes: the complete, final documentation tree from Tasks 1-18.
- Produces: `docs/architecture/CONTEXT_GRAPH.md`; updates the doc maps in `docs/README.md` and `CLAUDE.md` to reference it.

- [ ] **Step 1: Invoke graphify**

Invoke the `graphify` skill (via the Skill tool) over `docs/` and `CLAUDE.md` to build the project's persistent knowledge graph.

- [ ] **Step 2: Derive the context graph document**

Using graphify's output, write `docs/architecture/CONTEXT_GRAPH.md` as a Mermaid diagram mapping Domain → Bounded Context → Module → Class → Test File per the project's original context-graph convention — at this stage (pre-code) this will primarily map Domain → Documentation Concern rather than reaching all the way to Class/Test File, since no code exists yet. Note explicitly in the document that the Module/Class/Test File levels will populate once Phase 1 implementation begins.

- [ ] **Step 3: Add the doc-map lines that were deliberately deferred**

Add one line for `docs/architecture/CONTEXT_GRAPH.md` to `docs/README.md`'s architecture group, as a markdown link matching the format of the other lines there (e.g. `- [docs/architecture/CONTEXT_GRAPH.md](docs/architecture/CONTEXT_GRAPH.md) — ...`), and confirm `CLAUDE.md`'s doc map (Task 17) also lists it the same way — add it there too if it's missing.

- [ ] **Step 4: Verify no placeholder text remains and the diagram renders as valid Mermaid**

- [ ] **Step 5: Commit**

```bash
git add docs/architecture/CONTEXT_GRAPH.md docs/README.md CLAUDE.md
git commit -m "docs: generate context graph via graphify, link it from doc maps"
```

---

### Task 20: Final verification pass

**Files:**
- Read: every file created or modified in Tasks 1-19

**Interfaces:**
- Consumes: the complete final state of the repo.

- [ ] **Step 1: Full placeholder scan**

```bash
grep -rniE '\bTBD\b|\bTODO\b|placeholder|fill.?in.?detail' CLAUDE.md README.md docs/ --include='*.md' | grep -v -E 'docs/inputs/concepts/|docs/superpowers/'
```

Expected: no output. `docs/inputs/concepts/` (untouched raw source material) and `docs/superpowers/` (the spec and this plan, which legitimately discuss "placeholder" as a concept rather than containing one) are excluded — everything else is a real deliverable from this plan and must be clean. If anything prints, fix it before proceeding.

- [ ] **Step 2: Full link/path resolution check**

Extract every markdown-style path reference from `CLAUDE.md` and `docs/README.md` and confirm each resolves:

```bash
grep -ohE '\(docs/[^)]+\.md[^)]*\)|\(\.github/[^)]+\)|\(\.gitmessage\)' CLAUDE.md docs/README.md | tr -d '()' | sort -u | while read -r f; do
  test -f "$f" && echo "OK  $f" || echo "MISSING $f"
done
```

Expected: every line prints `OK`.

- [ ] **Step 3: Confirm the two known defects are actually fixed**

```bash
grep -n "React 19" CLAUDE.md docs/architecture/OVERVIEW.md
grep -n "conflicts with ALL cache-based" CLAUDE.md docs/decisions/adr/0003-standard-attention-cache-optimization.md docs/architecture/CAG.md
```

Expected: no matches for either command. If either matches, fix the remaining instance.

- [ ] **Step 4: Confirm ADR-0003's four-and-four split is stated identically in both places it appears**

Read `docs/decisions/adr/0003-standard-attention-cache-optimization.md` and the compatibility section of `docs/architecture/CAG.md` side by side; confirm they name the same four incompatible and four compatible techniques.

- [ ] **Step 5: Commit (only if Step 1-4 required fixes)**

```bash
git add -A
git commit -m "docs: fix issues found in final verification pass"
```

If no fixes were needed, skip this step — there's nothing to commit.

---

## Summary

20 tasks across 3 phases: Phase A (Tasks 1-11) builds the governance structure sequentially since later pieces depend on earlier ones existing. Phase B (Tasks 12-15) restores technical depth into four independent, parallelizable documents. Phase C (Tasks 16-20) assembles the final pointer structure and verifies it end-to-end.
