# Cross-Paradigm Batch F: CAG+MAG Synthesis — Live Measurement Report

**Scope:** #14 (story, umbrella validation), #1 (story, hot/warm tiering boundary), #5 (story, Sync Mixer tiebreak — investigate-first), #23/#28/#30 (their three tasks). See `docs/superpowers/specs/2026-09-02-cross-paradigm-cag-mag-design.md` for the design.

## What this batch is, and why it's the last cross-paradigm batch

RAG+CAG (Batch D) and RAG+MAG (Batch E) are both merged. This batch completes the cross-paradigm backlog with the one boundary neither touched: CAG↔MAG. Unlike the other two pairings, OVERVIEW.md names no dedicated technique here — #1 and #5 are both this project's own inference from the general Tiered Knowledge Hot-Cold Architecture, exactly as those issues say themselves.

This batch reuses three things completely unmodified: `FrozenCache`/`HFFrozenCache` (Batch D), `UserScopedAccessFrequencyTracker`/`InMemoryUserScopedAccessFrequencyTracker` (Batch E), and `sync_mixer.reconcile` (Batch D, generalized by Batch E) — its third and final reuse across all three cross-paradigm batches, with MAG playing the "authoritative" role this time instead of RAG.

**One new design decision:** `FrozenCache` is tenant-only, but the MAG content this batch promotes into it is personal, so reusing it unchanged would recreate the exact cross-user leak Batch E's `WarmStore` design exists to avoid. Rather than add a fourth near-duplicate port, this batch resolves it with key namespacing: `cag_mag_keys.cache_key(user_id, mag_content_key)` folds `user_id` into the `FrozenCache` key itself (that port has no `user_id` parameter at all), while `cag_mag_keys.tracker_key(mag_content_key)` stays content-only (the tracker already takes a real `user_id` argument of its own). Two different derived UUIDs, each matching the actual shape of the port it's used against.

## The investigation #5 asked for

**Can a genuine CAG-vs-MAG-only conflict — no RAG involved — actually arise?** Yes, and this batch's own tiering mechanism is what creates the opportunity: `CagMagTieringPolicy` freezes a point-in-time copy of a MAG fact into `FrozenCache`; MAG's own live data can keep changing after that (`UpdateMemory`, `RefineMemory`, `InvalidateMemory`, `ArchiveMemory` — all existing, unmodified MAG commands) while the CAG-side copy stays frozen. No RAG involvement needed; MAG's own normal write path is sufficient.

**Resolution rule, with its own reasoning since the source names none for this case: MAG wins.** RAG wins in the other two pairings because RAG is the external, ground-truth source neither CAG nor MAG can update directly. Here, CAG's hot-tier entry isn't an independent paradigm holding a competing fact — it's a cached copy of a specific MAG record, promoted BY this batch's own mechanism FROM MAG. MAG is the one paradigm that actually owns the data in this conflict; CAG's copy is derived, not authoritative, by construction.

This wasn't just reasoned about in prose — it was proven against real infrastructure (below).

## Real measured numbers

**Hot/warm tiering boundary (#1/#23):** a real MAG semantic fact ("strongly prefers matplotlib for all data visualization tasks") recorded via `RecordSemanticFact` (existing, unmodified). Driven through real access data:

> A document given 10 real accesses promoted (`TierDecision.PROMOTED`, crossing `promote_threshold=5`) into a real `HFFrozenCache` — confirmed as a genuine KV cache, not a stub. Once traffic genuinely dried up in a later window, the same content correctly demoted (`TierDecision.DEMOTED`).

**The real "near-zero TTFT on a hit" claim, measured for MAG-sourced content specifically** (not assumed to transfer from Batch D's RAG-document measurement): median cold prefill **42.6ms** vs. median warm **16.6ms** across 3 real trials — a ~2.6x margin, comfortably clearing the `1.5x` regression-detection floor Batch D's own review established. The same underlying mechanism, genuinely re-verified against a different content source.

**Per-user isolation, proven against real infrastructure:** a second real user's zero-traffic evaluation on the identical `mag_content_key` stayed `TierDecision.UNCHANGED`, and that user's own `FrozenCache` entry was confirmed absent — direct proof the `cag_mag_keys` namespacing design holds under real `HFFrozenCache`, not just against a fake in a unit test.

**Sync Mixer CAG-vs-MAG tiebreak (#5/#28):** a real MAG fact promoted into the real hot tier, then genuinely changed via a real `UpdateMemory.execute` call (existing, unmodified) — the same command a real caller would use for any ordinary fact correction, not a special test-only path. Polling `CagMagSyncCycle.run` (reading MAG's real current value from Postgres on every poll) on a real 50ms interval:

> **Real measured detection-to-eviction latency: 48.8ms** — against OVERVIEW.md's illustrative five-minute figure for the analogous RAG-vs-CAG case (this pairing has no source figure of its own, since the source never names this case at all). The stale `FrozenCache` entry was confirmed evicted. This is the investigation's own claim, proven end-to-end: a real conflict really does arise from this batch's own promotion mechanism, and the MAG-wins rule really does resolve it.

**Combined Capability Validation (#14/#30):** the tiering and sync results above ARE this pairing's real delta — a working promote/demote lifecycle against real infrastructure, genuine per-user isolation, and a sync mechanism that actually catches and corrects the exact divergence the investigation predicted. This replaces OVERVIEW.md's "targets, not results" status for CAG-MAG with an actual measured comparison — the last of the three cross-paradigm pairings to get one. A review finding (below) caught that the tiering and sync mechanisms were each demonstrated independently but never chained on the same entry — fixed with a dedicated test where a real `CagMagTieringPolicy` promotion feeds directly into `CagMagSyncCycle` detection, closing the loop this validation is actually supposed to prove.

## What adversarial review caught

A four-dimension review confirmed 4 of 5 raw findings; one finding — a claimed `CagMagTieringPolicy` tenant-isolation test gap — was raised by two different review dimensions with conflicting verdicts (one dimension confirmed it, one refuted it). Resolved in favor of refuting it: `CagMagTieringPolicy` is a pure passthrough of `tenant_id` to `FrozenCache`/`UserScopedAccessFrequencyTracker`, both of which already have dedicated, real tenant-isolation tests of their own, and this exact pattern — the identical gap claimed against `MagTieringPolicy`, the structurally identical Batch E sibling — was already examined and explicitly refuted in Batch E's own review for the same reason. No new test was added for this; the decision itself is the disclosed outcome.

- **The Combined Capability Validation was never actually tested end-to-end** — the tiering and sync mechanisms were each demonstrated against real infrastructure independently, but no test chained a real `CagMagTieringPolicy` promotion into `CagMagSyncCycle`'s detection on the same entry. Added `test_combined_capability_the_tiering_policys_own_promotion_feeds_sync_cycle_detection`, which does exactly that against real Postgres/Qdrant/Neo4j/`HFFrozenCache`.
- **`CagMagSyncCycle.run` returned `SyncConflict.document_id` as an opaque, one-way derived UUID with no way to trace it back to the caller's original `mag_content_key`** — unlike `SyncCycle`/`MagSyncCycle`, where `document_id` IS the caller's own input id. A real, reachable API usability gap for any batch caller with more than one key. Fixed by changing `run`'s return type to `list[tuple[str, SyncConflict]]`, pairing each conflict with its original key — fixed at this class's own boundary, without touching `sync_mixer.reconcile` or `SyncConflict` itself, both of which stay exactly as shared and unmodified as the design spec committed to.
- **`sync_mixer.reconcile`'s own docstring was stale**, still describing itself as "shared by both" the RAG-vs-CAG and RAG-vs-MAG cases with "RAG... always wins" — no longer accurate once this batch added a third caller where MAG wins. Fixed to describe the function as genuinely paradigm-agnostic, with each caller deciding what "authoritative" means for its own pairing.

## What this batch does not do

- **Does not add a new domain port.** `CagMagTieringPolicy`/`CagMagSyncCycle` compose Batch D's `FrozenCache` and Batch E's `UserScopedAccessFrequencyTracker` directly, using key namespacing rather than a fourth near-duplicate port.
- **Does not modify `FrozenCache`, `HFFrozenCache`, `UserScopedAccessFrequencyTracker`, `InMemoryUserScopedAccessFrequencyTracker`, or `sync_mixer.reconcile`.** Every one is reused completely unmodified.
- **Does not modify MAG's own domain layer, ports, or schema.** Every MAG read/write goes through existing, unmodified MAG application classes.
- **Does not require GPU or vLLM anywhere.**

## The numbers

Unit tests: 570 (baseline, post-Batch-E) → 588 (all mechanisms, including per-user AND per-tenant isolation tests for `CagMagSyncCycle` written in from the start, and a dedicated cross-user-collision test for the key-namespacing design — applying the lesson from Batch E's own review, which had to add its tenant-isolation test in a fix wave after missing it initially) → 589 (the review fix wave's own traceability test for `CagMagSyncCycle`'s new return shape). Integration tests: four (three initial, all passing on their first real run with no design bugs surfacing during implementation — unlike Batch D and Batch E, each of which caught at least one real bug via a real integration run — consistent with applying three batches' worth of accumulated lessons up front; plus one review fix-wave addition chaining a real `CagMagTieringPolicy` promotion into real `CagMagSyncCycle` detection on the same entry, closing the one real gap adversarial review did find in this batch).
