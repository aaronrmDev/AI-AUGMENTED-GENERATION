# MAG Batch B: Procedural Memory + Consolidation — Live Measurement Report

**Scope:** Procedural Memory (#17/#51), Consolidation (#11/#50). Builds on Batch A's storage foundation — see `docs/superpowers/specs/2026-08-26-mag-procedural-consolidation-design.md` for the design and `docs/architecture/MAG.md` for the underlying concepts.

## Procedural Memory: correctness, not a quality comparison

Same shape as Batch A's semantic/episodic memory sections — there's no baseline/treatment distinction for "can the system capture and retrieve a reusable procedure correctly."

- **Upsert correctness**: `RecordProcedure` derives a deterministic id (`uuid5` of `user_id`+`task_pattern`), so re-recording the same task pattern updates the existing row rather than accumulating duplicates — verified against real Postgres (`test_saving_the_same_task_pattern_twice_updates_instead_of_duplicating`), applying the exact lesson Batch A's semantic memory needed two review rounds to establish, from this batch's first version.
- **Tenant and user isolation, at two layers**: both the application-level filter and the database-level RLS backstop are tested against real Postgres, including a query with no application-level `WHERE tenant_id` at all (`test_procedural_memory_rls_returns_zero_cross_tenant_rows_without_an_app_level_filter`) — proving RLS itself blocks the cross-tenant row, not just that the repository's own SQL happens to filter correctly.
- **No embedding column, by design**: `docs/database/DATABASE.md`'s `ProceduralMemory` table has no `embedding` column (unlike `EpisodicMemory`/`SemanticMemory`) — retrieval is by exact `task_pattern` match. This is a real, intentional schema difference this batch preserves rather than "fixes," now documented explicitly in `DATABASE.md`.

## Consolidation: a real Ollama reflection, live-measured

**Claim tested:** given a batch of raw episodes, an LLM reflection pass can extract a durable, generalized fact — MAG.md's own worked example (three Python-related episodes consolidating into a primary-language fact).

**Methodology** (`tests/integration/test_consolidate_episodes_command.py`): three real episodes (a pandas question, a pytest question, an f-string question — all Python-specific, no explicit "I use Python" statement in any of them) seeded into real Postgres, reflected on by a real `qwen3.5` model via Ollama (not a fake, not a scripted response) through `ConsolidateEpisodes`, with the extracted fact(s) verified against both real Postgres and real Qdrant afterward.

**Result** (one live run, `2026-08-26`):

```
Consolidation extracted 1 fact(s) from a real Ollama reflection:
  'primary_programming_language' = 'Python' (confidence=0.9)
```

The model correctly inferred the generalized fact from three specific, Python-flavored interactions without ever being told "the user's language is Python" directly — this is a genuine reflection, not a pattern match against explicit text, matching MAG.md's own distinction between raw episodic content and distilled semantic knowledge. The extracted fact reached both real stores (Postgres via `find_by_key`, Qdrant via a real similarity search) with the same id, and all three source episodes were marked consolidated in the real database.

**Caveats:**
- Single live run, not repeated across multiple samples — LLM output for an open-ended reflection task varies more than the near-deterministic classification tasks (CRAG's relevance check, Self-RAG's gate) this project has already live-measured; the fact_key/fact_value wording is expected to vary run to run even when the underlying inference is correct. This report doesn't claim a stable extraction rate across repeated runs, only that one real, unscripted reflection produced a correct, well-formed result.
- The retry-on-malformed-JSON path (mirroring `OllamaJudge`'s corrected #149 behavior) is unit-tested against scripted malformed responses, not exercised live in this run — `qwen3.5` produced parseable JSON on the first attempt this time. Whether real malformed output occurs at a similar rate to what `OllamaJudge` observed (rare, but real — see #149) for this specific prompt shape is not yet known from a single run.
- No qualitative judge comparison: per the design spec, judging whether consolidated-memory answers are *better* than raw-episode answers needs Retrieval Strategies (Batch C) and Gating (Batch E) to make that a fair comparison — deferred, not attempted here.

## Out of scope for this batch

The six advanced retrieval strategies (Batch C) — Consolidation's episode selection here is oldest-unconsolidated-first (a backlog drain, not a recency window — see the design spec's correction note under Consolidation's Port changes), not the full recency+relevance+salience+abstraction weighting MAG.md describes. Memory Evolution's Archive operation (Batch F) — "mark as consolidated" is a timestamp flag here, not a move to cold storage.
