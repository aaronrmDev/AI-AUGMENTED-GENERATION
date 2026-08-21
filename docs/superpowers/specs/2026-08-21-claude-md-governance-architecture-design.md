# CLAUDE.md & Documentation Governance Architecture — Design Spec

- **Date:** 2026-08-21
- **Status:** Approved by repo owner (Aarón Rodríguez), ready for implementation planning
- **Author:** Claude (via superpowers:brainstorming)

## 1. Problem and context

This repository is the home of an ambitious "Unified RAG × CAG × MAG" AI system, but as of this spec it contains no application code at all — one commit, no GitHub remote, a one-line `README.md`, and a single `CLAUDE.md` file that already tries to carry the entire technical and procedural weight of the project. That file was synthesized from five source documents living in `docs/inputs/concepts/` (`advanced_rag_concepts.md`, `advanced_cag_concepts.md`, `advanced_mag_concepts.md`, `unified_rag_cag_mag_architecture.md`, `fullstack_unified_ai_system.md`), but the synthesis compressed away most of the depth: entire technique families (hybrid KV-cache offloading strategies, the six memory-gating strategies, the hot/cold tiered-knowledge model, named techniques like State-Aware RAG and Cache-Warmed RAG) were reduced to single bullets or dropped outright. The file also contains two verifiable defects — ADR-003 claims alternative attention "conflicts with ALL cache-based methods," when the CAG source document's own compatibility matrix marks only 4 of 8 techniques as conflicting, and the document header claims "React 19" while its own stack table says "React 18+."

Separately, the repo owner asked for CLAUDE.md to be improved along nine dimensions: a documented policy for minimizing token burn (skill-creator usage, `/usage`-driven compaction), explicit routing to the `fullstack-e2e-security-engineer` and `github-project-management-orchestrator` skills when relevant, a hard rule against freehand commits/tasks in favor of real templates, documentation written in humanized/pedagogically-sound prose, content that implicitly answers who/what/when/where/why/how/how-much without ever labeling itself that way, consistent divide-and-conquer folder separation, folders that hold only what genuinely belongs in them, a knowledge-graph mapping pass via the `graphify` skill, and a mechanism for CLAUDE.md to stay current as the project evolves.

This spec is the result of a three-section brainstorming pass with the repo owner (documentation architecture, root CLAUDE.md shape plus the writing standard, and templates/autolearning/graphify/verification), all three of which were approved as presented.

## 2. Goals

- Replace the single monolithic CLAUDE.md with a lean, pointer-based root file plus a categorized `docs/` knowledge base, so depth lives in focused, single-purpose files instead of one long document.
- Restore the technical depth that the current CLAUDE.md dropped from the five source documents, rewritten as humanized prose rather than compressed tables and bullets.
- Fix the two known defects (ADR-003 overstatement, React 19/18 contradiction) as part of the rewrite.
- Establish a concrete, checkable writing standard that operationalizes "humanized, pedagogically correct, implicitly covers who/what/why/when/where/how/how-much" so it can actually be followed and reviewed, not just aspired to.
- Document a token-economy policy (when to compact, when to reach for `skill-creator`) and a skill-routing policy (when to reach for the security, project-management, data-contract, and graphify skills) inside CLAUDE.md's governance layer.
- Provide real, usable templates for commits, issues, and pull requests, so nothing gets created freehand once GitHub-facing work starts.
- Build an on-demand skill (`claude-md-sync`) that lets the agent check whether CLAUDE.md and its linked docs are stale relative to the actual repo state, and propose specific, human-approved edits.
- Produce a persistent knowledge graph of the project via `graphify`, and a Mermaid context graph derived from it.

## 3. Non-goals

- No `src/`, `tests/`, `docker/`, or `k8s/` scaffolding in this pass. The concrete module layout from `fullstack_unified_ai_system.md` gets documented as the Phase 1 blueprint inside `docs/architecture/OVERVIEW.md`, but the directories themselves are not created until Phase 1 implementation actually begins — creating empty code directories now would itself be the "spread, nothing real inside" problem this spec is trying to avoid.
- No live GitHub configuration (branch protection, Projects V2 boards, actual issue/PR creation) — there is no GitHub remote yet, so `github-project-management-orchestrator` is used here only to produce the *template files*, not to configure a live project.
- No `.claude/settings.json` changes. The repo owner explicitly chose the "policy only" option for token-economy permissions — this is documented guidance, not a settings/permissions change.
- No CI-based drift detection. The repo owner chose the on-demand skill over a GitHub Action, since a Action would sit dormant without a remote.

## 4. Folder architecture

```
CLAUDE.md
README.md
.gitmessage
.github/
  ISSUE_TEMPLATE/
    epic.md
    story.md
    task.md
    bug.md
  PULL_REQUEST_TEMPLATE.md
.claude/
  skills/
    claude-md-sync/
      SKILL.md
docs/
  README.md
  architecture/
    OVERVIEW.md
    RAG.md
    CAG.md
    MAG.md
    CONTEXT_GRAPH.md
  decisions/
    adr/
      0001-vllm-over-sglang.md
      0002-qdrant-over-milvus.md
      0003-standard-attention-cache-optimization.md
      0004-hexagonal-cqrs-for-mag.md
      0005-langchain-langgraph-rag-orchestration.md
      TEMPLATE.md
  database/
    DATABASE.md
  security/
    SECURITY.md
  testing/
    TESTING.md
  governance/
    WRITING_STANDARDS.md
    TOKEN_ECONOMY.md
    SKILL_ROUTING.md
    AUTOLEARNING.md
  inputs/
    concepts/            (existing — untouched, kept as raw reference source)
  superpowers/
    specs/                (this spec lives here, per superpowers:brainstorming convention)
```

Every folder holds only content that genuinely belongs there. No directory is created as an empty placeholder for future use.

## 5. Root CLAUDE.md content plan

The root file drops from ~700 lines to an estimated 200–250 lines and becomes an entry point rather than a data dump:

1. **Opening identity paragraph** — prose, not a table. Explains what the project is and why it exists, so a reader is oriented before hitting any rule.
2. **Three-paradigm identity table** — the one table that stays inline, since RAG/CAG/MAG identity is the single fact every reader needs before anything else makes sense. Everything else currently in §2–§3 of the existing file (detailed layer diagrams, the full 3.1–3.8 technology stack tables) moves into `docs/architecture/OVERVIEW.md`.
3. **Non-negotiable directives, condensed** — each of the current §4.1–§4.9 rules gets one to three sentences plus a pointer to its deep doc (`docs/security/SECURITY.md`, `docs/testing/TESTING.md`, etc.), instead of the full explanation inline.
4. **Skill routing, inlined summary** — the operationally critical rules from `docs/governance/SKILL_ROUTING.md` stated directly: creative/architectural work starts with `superpowers:brainstorming`; security or audit work routes to `fullstack-e2e-security-engineer`; any task, issue, or board work routes to `github-project-management-orchestrator`; schema work routes to `data-contract-architect` or `db-architect-holistic`; mapping/graph work routes to `graphify`.
5. **Token-economy summary + pointer** — two to three sentences on monitoring `/usage` and preferring compaction over re-reading, pointing to `docs/governance/TOKEN_ECONOMY.md`.
6. **Template mandate** — a direct statement that commits, issues, and PRs are never freehand; they use `.gitmessage` and the `.github/` templates.
7. **Autolearning mandate** — instructs the agent to invoke `claude-md-sync` after any significant piece of work, per `docs/governance/AUTOLEARNING.md`.
8. **Doc map** — one line per file under `docs/`, so navigation never requires guessing.

## 6. Writing standard (`docs/governance/WRITING_STANDARDS.md`)

This is what actually operationalizes "humanized, psychologically and pedagogically correct, implicit 5W2H":

- Every section describing a real thing (a feature, a decision, a component) must let a reader work out **what** it is, **who** it's for or owns it, **why** it exists, **when**/under what conditions it applies, **where** it lives in the system, **how** it works, and **how much** it costs (latency, effort, resources) — but that coverage must emerge from prose that reads naturally, never from a labeled checklist walking through those categories one by one. The test: if a reader can't answer those questions from a natural reading of the paragraph, the paragraph isn't finished.
- Tables and bullets are reserved for genuinely tabular data — version numbers, schema fields, side-by-side comparisons — not for explaining how something works or why a decision was made.
- Tone assumes a competent reader who is new to *this* codebase: unfamiliar terms are defined on first use, dense concepts get a concrete example or analogy, and the writing addresses the reader the way a colleague would answer a real question, not the way a spec addresses a machine.
- "Everything tested, everything proved" becomes a citation discipline: any claim about behavior, performance, or a tradeoff must trace to something — a test that verifies it, a citation to the source concept documents, or (for anything not yet built) an explicit "design intent, not yet verified" flag. Nothing is stated as settled fact if it isn't.

This standard applies to every document in `docs/`, including the ones this spec creates.

## 7. Governance docs

- **`docs/governance/TOKEN_ECONOMY.md`** — documents monitoring context usage via `/usage`, preferring `/compact` or autocompact over re-reading already-seen material, and using `skill-creator` to build reusable skills instead of duplicating long instructions inline across conversations. Policy only; no `.claude/settings.json` changes.
- **`docs/governance/SKILL_ROUTING.md`** — the full routing table referenced in summary from root CLAUDE.md §4 above, including when each of `using-superpowers`, `fullstack-e2e-security-engineer`, `github-project-management-orchestrator`, `data-contract-architect`/`db-architect-holistic`, and `graphify` applies.
- **`docs/governance/AUTOLEARNING.md`** — describes the `claude-md-sync` skill's contract in human terms: when it runs, what it checks, and the fact that it always proposes edits for human approval rather than writing directly.

## 8. Architecture docs — content sourcing

- **`docs/architecture/OVERVIEW.md`** — absorbs the layered-architecture diagram, the orchestration meta-layer, the context-budget-allocation table, the latency-adaptive fallback cascade, the full 3.1–3.8 technology stack tables from current CLAUDE.md, and the concrete `src/` module blueprint from `fullstack_unified_ai_system.md` §4 (documented as the Phase 1 plan, not scaffolded).
- **`docs/architecture/RAG.md`** — rewritten from `advanced_rag_concepts.md` in humanized prose: the reranker taxonomy and its speed/cost tradeoffs, the Multi-Query vs. Query-Expansion distinction, HyDE's actual mechanic (embedding the hypothetical answer, not the question), the quantified per-technique impact figures, and the phased implementation roadmap — not just the four "Key Combinations" bullets currently in CLAUDE.md.
- **`docs/architecture/CAG.md`** — rewritten from `advanced_cag_concepts.md`: the named eviction algorithms beyond H2O/SnapKV (NACL, MorphKV, HASHEVICT, InfiniPot), the compression methods beyond KIVI/KVQuant (PALU, MiniCache, ShadowKV), the Hybrid Memory Offloading strategies (InfiniGen, LayerKV, INF2, KVPR, Oneiros) that are currently an unsupported bullet, multi-turn caching methods, and speculative decoding variants (Medusa, Lookahead/Prompt-Lookup Decoding).
- **`docs/architecture/MAG.md`** — rewritten from `advanced_mag_concepts.md`: all six memory-gating strategies (not just token-budget filtering), all four evolution operations including Refine, spreading activation for graph traversal, and the five required properties of episodic memory.
- **`docs/architecture/CONTEXT_GRAPH.md`** — generated via `graphify` (see §11).

The synthesis-layer concepts from `unified_rag_cag_mag_architecture.md` — the Tiered Knowledge Hot-Cold Architecture, State-Aware RAG, and Cache-Warmed RAG — get folded into `OVERVIEW.md` as named, explained techniques rather than left implicit.

## 9. ADR extraction and correction

The five ADRs currently inline in CLAUDE.md §11 move to `docs/decisions/adr/000N-*.md`, one file each, using a standard Context/Decision/Consequences prose structure (per the writing standard — no labeled bullet fields). During extraction, ADR-003 is corrected: alternative attention (Mamba, Linear Attention) is incompatible with 4 of the 8 CAG techniques (Eviction, Compression, PagedAttention, Speculative Decoding) and compatible with the other 4 (Prefix Caching, Hybrid Offloading, Multi-Turn Caching, Cache-Aware Batching) — not a blanket conflict with "ALL cache-based methods." Root CLAUDE.md keeps a one-line pointer per ADR.

## 10. Templates

- **`.gitmessage`** — Conventional Commits template (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`) matching the existing §4.8 convention, with inline comments guiding subject/body/footer. Wired via `git config commit.template .gitmessage` as part of implementation (a local, reversible config change, called out explicitly rather than done silently).
- **`docs/decisions/adr/TEMPLATE.md`** — standard ADR template, prose-based per the writing standard.
- **`.github/ISSUE_TEMPLATE/{epic,story,task,bug}.md`** and **`.github/PULL_REQUEST_TEMPLATE.md`** — produced by invoking `github-project-management-orchestrator` during implementation rather than hand-authored, per the repo's own "never from scratch" rule applied reflexively.

## 11. Autolearning skill: `claude-md-sync`

A project-level skill at `.claude/skills/claude-md-sync/`, built using `skill-creator` (per explicit instruction) rather than hand-authored. Required contract:

- **Trigger:** on-demand, invoked explicitly (e.g. `/claude-md-sync`) or by the agent's own judgment after a significant piece of work, per the root CLAUDE.md autolearning mandate.
- **Inputs:** recent git history since the last known sync point, the current `CLAUDE.md`, and the current `docs/` tree.
- **Behavior:** identify concrete places where the documented state (rules, doc map, architecture description) no longer matches the actual repo state, and produce a specific, itemized list of proposed edits with file paths and reasoning.
- **Hard constraint:** it never writes or commits changes on its own. Every proposed edit requires explicit human approval before being applied. This mirrors the brainstorming skill's own approval gate and is non-negotiable.

## 12. Graphify mapping

`graphify` runs over `docs/` and `CLAUDE.md` (and, later, `src/` once code exists) to build the project's persistent knowledge graph. `docs/architecture/CONTEXT_GRAPH.md` is derived from that graph as a Mermaid diagram, replacing the manual diagram sketch implied by current CLAUDE.md §4.7. The graph also becomes a future input to `claude-md-sync`, which can consult it to understand what's connected to what rather than relying on commit history alone.

## 13. Verification approach

Since this is documentation work on a pre-code repository, "tested and proved" applies to the documentation itself:

- Every new document gets a placeholder/TBD scan and an internal-consistency check before being considered done.
- Every pointer in root CLAUDE.md and `docs/README.md` is checked to confirm it resolves to a real file — no dead links in the doc map.
- The two known defects (ADR-003 overstatement, React 19/18 contradiction) are fixed as part of the rewrite; React 18+ is the corrected value, matching what the source document actually states.
- Content fidelity is checked against the five source documents for the architecture docs specifically, since that's where compression happened — the gap-analysis findings in this spec (§8) are the checklist for what must reappear.

## 14. Rollout sequencing

Given the volume of prose-writing work, implementation proceeds in two layers:

1. **Governance and structure** — folder skeleton, root CLAUDE.md, the four governance docs, the ADR extraction and correction, the templates, and the `claude-md-sync` skill. This layer is largely sequential since later pieces (root CLAUDE.md's doc map, the sync skill) depend on the structure existing first.
2. **Content depth restoration** — `OVERVIEW.md`, `RAG.md`, `CAG.md`, `MAG.md`. These four are independent of each other (each sources from a different subset of the five concept documents) and get executed as parallel, divide-and-conquer writing tasks rather than sequentially.

`graphify` mapping and the final link/consistency verification pass happen after both layers are complete, since they need the finished doc tree to be meaningful.
