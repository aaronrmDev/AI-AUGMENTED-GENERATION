# CAG Batch A: Alternative Attention Compatibility — Design Spec

**Scope:** #39 (story), #130 (task) — the first CAG work in this repository. See `docs/architecture/CAG.md`'s "Alternative attention: the compatibility question, corrected" section and `docs/decisions/adr/0003-standard-attention-cache-optimization.md` for the source material this batch encodes.

## What this issue actually asks for

Issue #39's Definition of Done is the same three lines every CAG/MAG issue in this backlog uses: implemented, validated against the source's stated behavior, a real result documented in the issue's comments. Unlike every other CAG technique in this backlog, Alternative Attention isn't a per-request serving technique to build — it's the Architecture-stage decision this project already made (ADR-0003, Path A: standard attention, not alternative attention) plus a specific factual claim about *why* that decision matters operationally: alternative attention is incompatible with exactly 4 of the 8 other CAG techniques this project relies on, not "every cache-based method" the way the original project brief's overstated language claimed. #130 asks specifically to verify that corrected 4-incompatible/4-compatible split against ADR-0003.

`docs/architecture/CAG.md` already restates ADR-0003's split verbatim in prose, with an explicit note that its own summary table's Alternative Attention row deliberately reproduces the source's looser "Standalone / incompatible with cache methods" framing, pointing readers to the corrected section below it. Reading both documents directly (not just CAG.md's own paraphrase of ADR-0003) confirms they already agree — this is not a live discrepancy to fix, matching what a prior investigation into this task found. What "verify" and "implement" mean here, given there is nothing already wrong to correct, is: give this decision a real, testable encoding in code, so that any *future* CAG module can query "is technique X compatible with this project's chosen attention architecture" against a single source of truth instead of every caller re-deriving the answer from prose — and prove that encoding matches ADR-0003's exact split via a test, not just an eyeball read of two markdown files agreeing with each other.

## Approach

This is the first code under `src/cag/`. `docs/architecture/OVERVIEW.md`'s own module blueprint (written before any paradigm module was built) sketches a flat `cache/`, `eviction/`, `compression/`, `serving/` layout for `cag/` — but that blueprint predates MAG's actual implementation, which established a hexagonal `domain/`/`application/`/`infrastructure/` split instead, per CLAUDE.md's project-wide hexagonal-architecture rule ("The backend keeps a domain layer free of framework dependencies, an application layer that orchestrates use cases, and an infrastructure layer that talks to the database, cache, and external APIs"). CAG follows that same real precedent rather than the superseded flat sketch, the same way MAG did.

A new domain module, `src/cag/domain/attention_compatibility.py`:

- `CAGTechnique`, an enum of the 8 KV-cache-dependent techniques ADR-0003's split is actually about: `KV_CACHE_EVICTION`, `KV_CACHE_COMPRESSION`, `PAGED_ATTENTION`, `SPECULATIVE_DECODING` (the 4 incompatible ones), `PREFIX_CACHING`, `HYBRID_OFFLOADING`, `MULTI_TURN_CACHING`, `CACHE_AWARE_BATCHING` (the 4 compatible ones). Alternative Attention itself is the axis being decided, not a 9th member of this enum.
- `is_compatible_with_alternative_attention(technique: CAGTechnique) -> bool`, a pure function encoding ADR-0003's split exactly — no framework dependency, no I/O, matching the domain layer's own constraint.

This module has no application or infrastructure layer of its own yet — there's no use case to orchestrate and no external system to talk to for a static compatibility fact. Later CAG batches that build the techniques themselves are where an application layer (e.g., a serving-configuration use case asking "which of these techniques should this deployment enable, given its attention architecture") would actually consume this module; scaffolding that caller now would be speculative given nothing calls it yet.

## Testing plan

`tests/unit/test_attention_compatibility.py`:
- One assertion per technique, naming it explicitly against ADR-0003's own text, rather than looping over "all techniques" generically — a future edit that silently changes one technique's compatibility should fail on that technique's own named test, not a generic parametrized failure.
- An invariant test asserting the split is exactly 4 incompatible / 4 compatible, matching this issue's own "4-incompatible/4-compatible" framing as a literal, checked fact rather than an implied one.
- A completeness test asserting every member of `CAGTechnique` has a defined compatibility value (guards against a future technique being added to the enum without its compatibility being decided).

No integration test and no live infrastructure — this is a pure domain-logic verification task, unlike the CPU/Ollama-feasible techniques the next few batches cover, and unlike the GPU-dependent ones deferred until vLLM-on-ROCm setup. The "real result documented" line of the Definition of Done is satisfied by pasting the test run's actual output (the confirmed 4/4 split, by name) into the issue's closing comment.

## What this batch does not do

- **Does not implement any of the 8 techniques themselves** — those are separate CAG batches (KV Cache Eviction, Compression, Hybrid Offloading, Multi-Turn Caching, Speculative Decoding, PagedAttention, Prefix Caching, Cache-Aware Batching), sequenced afterward per the CPU-feasible-first plan.
- **Does not touch CAG.md or ADR-0003** — both already state the corrected split accurately; there is nothing to fix in either document.
- **Does not add an application or infrastructure layer for CAG yet** — nothing calls this domain fact yet; scaffolding unused layers would be speculative.
- **Does not require GPU or vLLM** — this is the one CAG task in the current backlog that needs no live model or serving infrastructure at all.
