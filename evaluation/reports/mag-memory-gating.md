# MAG Batch E: Memory Gating — Live Measurement Report

**Scope:** #15 (parent), #53 (Top-K selection), #54 (Token budget allocation), #55 (Hierarchical assembly), #57 (Recency-weighted sampling), #58 (Task-specific filtering), #60 (Dynamic re-ranking). Builds on Batch C's scored retrieval (`ScoredEpisode`/`ScoredFact`) and Batch D's spreading activation (`ActivatedNode`) as its three candidate sources — see `docs/superpowers/specs/2026-08-29-mag-memory-gating-design.md` for the design and `docs/architecture/MAG.md` for the underlying concepts.

## The normalization problem this batch actually had to solve

MAG.md names six gating strategies, but doesn't say what they operate *on* — retrieval returns three structurally different types (`ScoredEpisode`, `ScoredFact`, `ActivatedNode`), each with its own idea of "score," "timestamp," and "content." Gating a mixed pool means every strategy would otherwise need six branches instead of one. This batch's real design decision was `GatingCandidate`, a normalization wrapper modeled directly on Batch D's own precedent for the same problem one level down (`ActivatedNode` unifying three graph node types) — three adapter functions (`from_scored_episode`, `from_scored_fact`, `from_activated_node`) flatten each source into one shape, and every gating strategy is written once, against that shape, never against the three original types.

## Six strategies, one orchestrator, MAG.md's own worked example reproduced

`GateMemories` composes five of the six strategies into MAG.md's own described pipeline order (dynamic re-ranking → task filtering → recency decay → token budget → hierarchical assembly) — `TopKSelection` remains fully implemented and independently usable but is deliberately not wired into this default composition, since token-budget allocation is the better fit for a pool that mixes wildly different content lengths (an episode's raw JSON versus a short fact string).

**Live-tested against real Postgres retrieval and a real embedding model** (`test_gate_memories_narrows_real_retrieval_output_to_fit_a_tight_budget`): five real episodes and three real facts, real cosine-similarity retrieval, real tiktoken counts — the full pool costs more tokens than a deliberately tight budget allows, and `GateMemories` narrows it down while staying under budget, reproducing MAG.md's "50 memories at 15K tokens → 20 memories at 7K tokens" shape structurally, not by hardcoding those numbers.

**Fact-before-episode ordering, isolated from coincidence** (`test_gate_memories_orders_facts_before_episodes_even_when_the_episode_scores_higher`, added during the second review round — see below): an episode whose content is set to the query text verbatim, which cosine similarity mathematically maximizes, is guaranteed a real retrieval score at least as high as any fact's. The test asserts the fact still lands first anyway, which can only be `HierarchicalAssembly`'s source-type-priority stage at work, not a coincidence of which way real embedding similarity happened to fall for a particular query.

## Deliberately not written by this batch

- **Hard constraints beyond the token budget** — MAG.md's pipeline description also mentions "forbidden topics" and "required inclusions" as filter criteria; this batch implements only the max-tokens constraint each of the six named strategies actually corresponds to. No strategy or orchestrator stage exists for topic denylisting or forced inclusion.
- **A procedural-memory candidate source** — `GatingCandidate` has three adapter functions, one per existing scored/activated retrieval type. No retrieval strategy in this codebase yet returns a scored list of procedural memories, so there is no fourth adapter to write.
- **`TopKSelection` wired into `GateMemories`'s default pipeline** — a deliberate scope boundary, not an oversight; see above.

## What two full review rounds caught

**Round one** (six-dimension adversarial review against the initial implementation) confirmed all 11 raw findings it produced — every one survived independent skeptic verification. The two most consequential:

- **NaN-score sort corruption.** Python's `sorted()` requires strict weak ordering, which NaN violates (`nan < x` and `x < nan` are both `False`). A NaN score reaching any of this batch's five score-sorting strategies didn't just misplace the NaN candidate — it could silently scramble the relative order of *other*, well-defined scores around it, with no exception raised. Reachable via `DynamicReranking`'s cosine similarity against a corrupted embedding. Fixed with a shared `safe_score()` helper (NaN → `-inf`, "assume irrelevant") applied at every sort site.
- **Recency decay inverted for negative scores.** `RecencyWeightedSampling` reused Batch C's `score * decay` formula unchanged, but Batch C's version only ever received scores already min-max-normalized to `[0, 1]` before decay was applied — a step this batch's version never had, since it operates on a pool that can carry `DynamicReranking`'s raw cosine similarity (`[-1, 1]`). Multiplying a negative score by a decay factor in `(0, 1]` shrinks it *toward* zero as it ages, i.e. an old, confidently-bad memory would rank *above* a fresh, equally-bad one. Fixed by scaling negative scores by `(2 - decay)` instead, which pushes them further from zero (worse) with age, mirroring how a positive score is pushed toward zero.

The other nine: a `token_budget == 0` edge case that incorrectly excluded a genuinely zero-cost candidate; two tests in `test_task_specific_filtering.py` that constructed field-identical candidate fixtures and then asserted on their relative order — dataclass `==` can't distinguish real ordering from any permutation when the compared objects are already equal to each other, the same trap this project's own review process has now caught three separate times across two batches; a skippable, coincidence-dependent ordering assertion in the integration test (replaced with the dedicated test described above); three invariant-only assertions strengthened to exact values or pinned clocks; and an inaccurate code comment.

**Round two** (a scoped re-review of round one's own fix wave, matching this project's established two-round cadence) confirmed all 5 raw findings it produced:

- The NaN fix guarded every sort *comparison* but left the raw NaN sitting in the returned candidate's own `.score` field — inert today (nothing in this codebase yet serializes or aggregates a `GateMemories` result), but a live trap for the first future consumer that does. Closed at the source: `DynamicReranking`'s `_cosine_similarity` now sanitizes any non-finite result to `0.0`, the same fallback it already used for the zero-norm case.
- `_scoring.py`, the new module holding `safe_score()`, had no dedicated test file — breaking this package's own established convention (its sibling private helper module, `_candidates.py`, already has one). Added `test_scoring.py`, pinning the helper's exact contract in isolation.
- A coverage gap where the new zero-budget regression test only covered a single zero-cost candidate, not a mixed pool where a zero-cost candidate is ranked behind a skipped higher-scored non-zero-cost one.
- A stale token-count comment (claimed both fixture candidates were "~500+ tokens each"; one is actually ~255).
- `DynamicReranking`'s own NaN test needed updating for the source-level fix — a NaN-embedding candidate now resolves to `0.0`, not NaN, so a new companion test was added proving the sort itself still defends a candidate that arrives already NaN-scored some other way.

## The numbers

Unit tests: 333 (start of batch, before any gating code existed) → 340 (after round-one fixes) → 349 (after round-two fixes) — 16 net new tests across both review rounds' fix waves, on top of whatever the initial six-strategies-plus-orchestrator implementation itself added. Integration tests: 160 → 161 (the one new dedicated ordering test; the original tight-budget test's narrowing/budget assertions stayed, its coincidence-dependent ordering assertion was replaced rather than deleted-and-not-replaced). Both suites green, no previously-passing test regressed, all integration tests run against real Postgres — no mocks.
