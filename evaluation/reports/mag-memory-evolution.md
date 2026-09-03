# MAG Batch F: Memory Evolution — Live Measurement Report

**Scope:** #16 (parent), #62 (Update), #63 (Invalidate), #64 (Archive), #66 (Refine). All four operate on `SemanticMemory` (Batch A/C's fact storage) — see `docs/superpowers/specs/2026-08-29-mag-memory-evolution-design.md` for the design and `docs/architecture/MAG.md` for the underlying concepts.

## What the four child issues ask for, versus what the parent narrates

The parent issue narrates memory evolution as one pipeline — detect a similar existing memory, compare it against new information with an LLM, branch three ways, propagate the change through the graph. Taken literally, that three-way branch doesn't line up 1:1 with the four child issues' four operations. Matching this project's own established pattern (the parent narrates the source concept, the children define exactly what to build — see every prior batch's own design spec), this batch implements the four precisely-scoped operations the children describe, plus a classification+dispatch orchestrator that answers the parent's own "detection, comparison, decision" language directly.

| Issue | Operation | Trigger | Action |
|---|---|---|---|
| #62 | Update | New info directly contradicts old info | Overwrite the fact; keep the old value, not delete it |
| #63 | Invalidate | Old info is no longer true at all | Mark the fact stale; exclude it from retrieval |
| #64 | Archive | Fact is rarely accessed | Move to cold storage; keep it available for explicit reference |
| #66 | Refine | New info adds nuance, doesn't contradict | Merge old + new into one richer fact |

## The orchestrator: `EvolveMemory`

`ClassifyFactEvolution` — an LLM judgment mirroring `capture_episode.py`'s established salience-scoring pattern (retry loop, type-guarded response, fail-safe default) — decides how a piece of new information relates to an existing fact: update, invalidate, refine, or no_conflict. `EvolveMemory` composes it with three of the four operations, the same way Batch E's `GateMemories` composed five of its own six siblings. Archive is deliberately not part of the dispatch — its trigger (access frequency) has nothing to do with comparing new information against old, so it stays independently invocable, matching how `TopKSelection` stayed fully built but outside `GateMemories`'s own default composition.

**Live-verified against a real Ollama model (qwen3.5), not just fakes:** `EvolveMemory` correctly classified "moved to Berlin last week" against an existing "lives in New York" fact as `update` and dispatched to the right operation, reproducing MAG.md's own worked example through the full classify-then-dispatch path. A second live test reproduced MAG.md's Refine example — "prefers Python" merged with "especially for data analysis, though open to Go for CLI tools" — and the real model's output matched the source's own phrasing verbatim in that run (the test's own assertions require only the real semantic content, not that exact string, since a live model's phrasing isn't guaranteed identical run to run).

## A real, pre-existing gap this batch had to close to make Invalidate/Archive mean anything

`PostgresSemanticMemoryRepository.search_by_similarity` and `QdrantSemanticMemoryIndex.search` both ignored `valid_until` entirely before this batch — a fact past its expiry was still returned by similarity search. Invalidate's own contract ("exclude it from retrieval") was meaningless against a filter that didn't exist. Both search paths now exclude a fact whose `valid_until` has passed or whose new `archived_at` column is set. `find_by_key` stays deliberately unfiltered — a direct, keyed lookup a command needs to fetch the *current* row regardless of status.

## What two full review rounds caught

**Round one** (six-dimension adversarial review against the initial implementation) confirmed all 16 raw findings it produced. The most consequential, found independently by three different review dimensions and empirically reproduced twice:

- **Correcting an archived fact's value silently un-archived it.** `UpdateMemory`/`RefineMemory` composed `RecordSemanticFact` for their overwrite, but `RecordSemanticFact` defaulted `archived_at`/`valid_until` to `None` with no way for the callers to pass the existing fact's real status through. A fact deliberately Archived or Invalidated would reappear in default search the moment anyone corrected or enriched its value — with no error, no log, and (at the time) no test covering the interaction. Fixed by giving `RecordSemanticFact` an `archived_at` parameter (mirroring the pre-existing `valid_until` one) and having Update/Refine explicitly forward both fields from the fact they're overwriting.
- **A concurrent Invalidate + Archive on the same fact could clobber one field with a stale snapshot of the other**, in both the Qdrant payload and (caught again in round two) the Neo4j graph node — see below.
- **The live-Ollama Refine integration test's own assertions were satisfiable by `RefineMemory`'s exhausted-retries fallback** (a plain string concatenation), so a green run didn't actually prove a real LLM merge occurred for that specific fixture.
- A clock-skew window between the application's `datetime.now(UTC)` and the database's own `now()` comparison in `search_by_similarity`; a `no_conflict` fail-safe indistinguishable from a genuine judgment; `EvolveMemory`'s invalidate dispatch discarding the triggering information with no durable trace; missing RLS regression tests for the new history table; and several smaller test-quality and documentation gaps.

**Round two** (a scoped re-review of round one's own fix wave) confirmed all 5 raw findings it produced — a genuinely useful catch, since two of them were bugs *in the first fix wave's own fix*:

- **The Qdrant race fix wasn't mirrored for Neo4j.** Splitting Qdrant's status write into `set_valid_until`/`set_archived_at` (each touching only its own payload key) closed the race for Qdrant, but `InvalidateMemory`/`ArchiveMemory` still funneled both fields into a single `upsert_fact_node` call built from a `dataclasses.replace()` of the *same* stale snapshot read at the top of `execute()` — the identical race, reproduced with a concrete interleaving, survived unmodified for the graph store. Closed by giving `MemoryGraphRepository` the equivalent `set_fact_valid_until`/`set_fact_archived_at` split.
- **Two "unswapped argument" test assertions added in round one's own fix wave were themselves vacuous.** The fixture text chosen ("prefers Python"/"data analysis", "lives in New York"/"Berlin") turned out to be copied verbatim from the system prompts' own worked examples, so `str.find()` located both substrings inside the constant system-prompt portion of the sent text regardless of whether the real arguments were correctly ordered or swapped — confirmed by both reviewers independently swapping the real arguments and watching the "unswapped" test stay green. Fixed by choosing fixture text that appears nowhere in either prompt's own examples, then re-verifying with the same swap-and-revert reproduction, which now correctly fails.
- Only 1 of the 4 (Update/Refine × valid_until/archived_at) status-preservation combinations had any test coverage at all — the sole regression test exercised just Update+archived_at. Closed with the missing three.
- `ConsolidateEpisodes` — a third caller of `RecordSemanticFact`, untouched by round one's fix — has the identical status-reset exposure: a consolidation-derived `fact_key` shares a user's per-key namespace with the memory-evolution operations, so a collision could silently un-archive a fact the same way. Not a regression introduced by this batch's diff (disclosed as informational by both reviewers), but the same bug class and a cheap, direct fix: `ConsolidateEpisodes` now looks up any existing fact by key first and forwards its status through.

## Deliberately not written by this batch

- **Automatic similarity-based "detection"** of which existing fact a new piece of information is about — a caller supplies a specific `fact_key`, matching the explicit-parameters convention every prior MAG batch has followed.
- **Multi-hop propagation to *other* linked memories in the graph.** The parent issue's own language implies walking edges to update related facts too, but this schema has no edge type representing "these two facts are related" at all (Batch D scoped `RELATED_TO` out of its own batch, and nothing since has added it). This batch syncs the one Fact node that actually changed.
- **A durable audit trail for *why* a fact was invalidated.** Invalidate has no history table to snapshot into (unlike Update/Refine, which durably encode the trigger in the new value or the old one). `EvolveMemory`'s invalidate dispatch logs the triggering information and the classifier's reasoning, but nothing persists it to a queryable store.
- **Exhaustive live-model testing of every classification boundary.** Only the `update` path is exercised against a real Ollama model end-to-end; `invalidate`/`no_conflict` are covered only via fakes. This matches every other LLM-classification feature in this codebase (Batch C's salience and causal-relevance live tests do the same) — exhaustively probing an unbounded classification boundary against a live model in a deterministic suite isn't feasible, and this isn't a defect specific to this batch.

## The numbers

Unit tests: 349 (start of batch) → 389 (initial implementation) → 394 (round-one fixes) → 398 (round-two fixes). Integration tests: 166 → 175, all against real Postgres, Qdrant, Neo4j, and (for the classification and refine paths) a live Ollama model — no mocks anywhere in this batch's integration suite.
