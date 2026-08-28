# MAG Foundation (Batch A) — Live Measurement Report

**Scope:** Memory Hierarchy (#4/#46), Episodic Memory (#7/#47), Semantic Memory (#9/#49). First MAG code in the repository — see `docs/superpowers/specs/2026-08-26-mag-foundation-design.md` for the design and `docs/architecture/MAG.md` for the underlying concepts.

**Why this report doesn't use the RAG evaluation harness's format:** every prior report in `evaluation/reports/` compares an LLM's answer quality across a baseline/treatment pair, judged qualitatively by another LLM. Memory Hierarchy's actual claim (`docs/architecture/MAG.md`: "hot data belongs in context, warm data in RAM, cold data in a persistent store") is a systems/latency claim with no LLM output to judge — forcing it through that harness would produce a quantitative table with meaningless token counts and an empty qualitative section. This report measures what was actually tested instead.

## Memory Hierarchy: working-memory tier vs. cold-store round trip

**Claim tested:** reading a session's recent turns from the fast tier (Redis, `RetrieveWorkingMemory`) is meaningfully faster than reading the same logical data from the durable tier (Postgres, a direct query against `episodic_memory`) — the reason the fast tier exists at all.

**Methodology** (`tests/integration/test_working_memory_latency.py`):
- Real `testcontainers`-launched Postgres and Redis, not mocks or fixtures.
- The Postgres side is seeded with 12,000 *other* sessions' episodes (400 sessions × 30 rows) before measuring, plus `ANALYZE episodic_memory` — an empty, freshly-created table produced a statistical tie against Redis in an earlier version of this test (both stores answer a trivially small lookup at roughly the same loopback-dominated speed), which is not an honest represention of a "durable store holding a real conversation history" baseline.
- An `EXPLAIN` check, run before timing starts, asserts the query plan is *not* a sequential scan — this is a real, executed check, not a claim in a comment; the assertion passed, confirming `ix_episodic_memory_session_id` is actually what the measured numbers reflect.
- 5 untimed warm-up reads on both paths first (asyncpg plan-caching, redis-py's lazy connection), then 50 timed reads each, reporting the median (p50).
- The comparison is structurally biased *against* the hypothesis: the Redis path deserializes 20 JSON payloads and parses 20 ISO datetimes per read; the Postgres path selects one column and discards it.

**Result** (one live run, `2026-08-26`):

| Path | Read | p50 over 50 reads |
|---|---|---|
| Baseline | Postgres (`episodic_memory`, 12,000 background rows, indexed) | 0.800 ms |
| Treatment | Redis (`RetrieveWorkingMemory`) | 0.575 ms |

**Speedup: 1.39×.**

**Caveats:**
- Single run, not repeated across multiple independent samples the way this project's RAG comparisons use `repeat_count=3–5` — a systems latency measurement doesn't have the same sampling-noise profile as an LLM's answer generation, but the exact ratio should be read as directionally real, not a precise constant. Two prior manual runs during this batch's development (before the ANALYZE/EXPLAIN fix) measured 1.32× and 1.41× on the same seeded scale, consistent with this run's 1.39×.
- Both numbers are sub-millisecond — this is `testcontainers` on Docker Desktop for Windows, loopback networking, on this specific machine. The *relative* difference (Redis reads a bounded, per-session key; Postgres has to use an index across a table that grows with every session in the system) is the architectural property being demonstrated, not a production latency SLA.
- Redis's TTL-bearing write path (`push_turn`) was not benchmarked here, only reads — writes are cheap RPUSH+EXPIRE either way and weren't the contested claim.

## Episodic Memory and Semantic Memory: correctness, not a quality comparison

Unlike Memory Hierarchy, there is no baseline/treatment distinction for "can the system capture and retrieve a memory correctly" — that comparison belongs to a later batch, once retrieval strategies (Batch C) and gating (Batch E) exist to make a full pipeline worth judging end-to-end against not having memory at all. This batch validates the storage layer those later batches build on:

- **Dual-write correctness**: `CaptureEpisode` and `RecordSemanticFact` write to both Postgres (the durable, transactionally-consistent record) and Qdrant (the embedding-bearing nearest-neighbor search path) — verified via real testcontainers Postgres and Qdrant, not fakes, for both write and read paths.
- **Nearest-neighbor ordering is real**: `test_search_by_similarity_orders_by_nearest_neighbor` (episodic) and `test_search_by_similarity_returns_real_nearest_neighbor_ordering` (semantic) embed genuinely different content via the real `SentenceTransformersEmbedder` and confirm a semantically-close query actually ranks the close fact/episode first — not asserted against a fixture vector.
- **Tenant and user isolation is real, at two layers**: both the application-level `WHERE tenant_id = ...` / `WHERE user_id = ...` filters *and* the database-level RLS backstop are independently tested (`tests/integration/test_mag_rls_tenant_isolation.py` runs a query with **no** application-level filter at all, proving RLS alone — not app code discipline — is what actually blocks a cross-tenant read).
- **Upsert correctness**: `SemanticMemory` facts upsert by `(user_id, fact_key)` (a `UNIQUE` constraint plus `ON CONFLICT DO UPDATE`, both found necessary by adversarial review — see below) rather than accumulating unbounded duplicates resolved nondeterministically at read time.

## Process note: what the adversarial review caught

The first-pass implementation (three parallel subagents, one per vertical) passed all 45 of its own tests and both ruff/mypy — and still had two real defects an adversarial review found before merge: `RecordSemanticFact` could never actually update a fact (no unique constraint, no upsert — two saves with the same key produced two rows resolved by a random UUID tie-break at read time), and `semantic_memory` had no `tenant_id` column or RLS at all, with the design spec's own justification for that gap being factually wrong about the schema it cited (`Sessions` was claimed to scope through a join alone; it doesn't — it has always had its own `tenant_id` + RLS). Both are fixed in this merged version, along with five smaller issues (a `limit=0` Redis bug that returned an entire session instead of nothing, an architectural layering violation where application code imported concrete Qdrant classes instead of ports, a missing `by_similarity` query path, an untested RLS backstop, and — caught by this report's own stricter test — Qdrant's COSINE-distance collections normalizing every stored vector to unit length, so a search-based embedding read is not bit-identical to what was written). Full detail in the design spec's correction note and the branch's commit history.

## Out of scope for this batch

Procedural Memory, Consolidation (LLM-based episode→fact extraction), the six advanced retrieval strategies beyond plain similarity, Memory Graphs/Neo4j, Memory Gating, Memory Evolution — all later batches (B through F), per the design spec's sequencing.
