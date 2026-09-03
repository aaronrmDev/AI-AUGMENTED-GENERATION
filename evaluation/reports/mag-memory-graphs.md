# MAG Batch D: Memory Graphs — Live Measurement Report

**Scope:** #13 (parent), #76 (spreading activation). Builds on Batch A's storage foundation, Batch B's consolidation pipeline, and Batch C's salience/entity scoring — see `docs/superpowers/specs/2026-08-28-mag-memory-graphs-design.md` for the design and `docs/architecture/MAG.md` for the underlying concepts.

## New ground: Neo4j didn't exist in this project before this batch

Checked before writing the design spec, not assumed: no `docker-compose` service, no Python driver dependency, no test fixture. This batch is the first time any MAG batch had to stand up genuinely new infrastructure rather than build on an existing store. All three additions match the established conventions of the three stores that came before — a session-scoped testcontainer pinned to the same major version the docker-compose service runs, `127.0.0.1` explicit in the URL fixture for the same Windows/IPv6 issue documented on `redis_url`/`qdrant_url`, and a `neo4j:5-community` image matching `OVERVIEW.md`'s own `neo4j >=5.20.0` driver-version floor.

## The schema: implemented, not designed

`docs/database/DATABASE.md` already specified the full node/edge/index schema before this batch existed — `User`, `Session`, `Entity`, `Concept`, `Episode`, `Fact` nodes; `PARTICIPATED_IN`, `MENTIONS`, `RELATED_TO`, `ABSTRACTS_TO`, `TEMPORALLY_FOLLOWS` edges; indexes on `Entity.name`/`Entity.embedding`. This batch's job was building the write and read paths that schema implies, via a new `MemoryGraphRepository` port and `Neo4jMemoryGraphRepository` implementation, all raw Cypher via bound parameters (the one exception — `max_hops` interpolated as a validated literal, since Cypher's variable-length-path syntax requires a parse-time literal there — is disclosed directly in the code, not hidden).

**Real correctness, live-tested against real Neo4j (22 integration tests):** node upsert idempotency, every one of the five edge types, tenant isolation (including the session-adjacent case a review caught was originally missing coverage for on four of the six write methods), and spreading activation's core algorithm — geometric decay per hop, max-not-sum activation when a node is reachable by more than one path, a hard threshold cutoff, and a validated `[1, 10]` hop range with `(0.0, 1.0)` decay-factor range after a review caught both parameters accepting anything, including values that would silently invert the ranking or crash Cypher parsing.

## Spreading activation: the worked example, reproduced structurally

**Claim tested:** MAG.md's own restaurant-recommendation example — a query that a plain vector search would miss because the literal word never appears in the source conversation, answered instead by walking a chain of typed edges.

**Methodology** (`test_spread_activation_reaches_a_fact_two_hops_from_a_mentioned_entity`): a real Neo4j graph seeded with an episode mentioning "Paris" (`MENTIONS` edge) that abstracts into a `restaurant_recommendation` fact (`ABSTRACTS_TO` edge) — two hops from the entity a query would actually start from, with the literal string "restaurant" never appearing on the traversal path at all.

**Result:** spreading activation from `["Paris"]` correctly reaches the fact at hop 2 with activation `0.25` (`0.5 ** 2`), the intermediate episode at hop 1 with activation `0.5`, and the start entity itself at hop 0 with activation `1.0` — exactly the geometric decay the design specifies, over a real Cypher variable-length-path traversal, not hand-rolled Python BFS.

## Wiring the write path into existing commands

`CaptureEpisode`, `RecordSemanticFact`, and `ConsolidateEpisodes` (all pre-existing, already-reviewed code from Batches A–C) now mirror their writes into the graph as a fourth store, alongside Postgres and Qdrant — matching `DATABASE.md`'s own "three separate writes to three separate systems, not one atomic transaction" model, now a fourth. `CaptureEpisode` gained a `user_id` parameter it never needed before (Postgres/Qdrant writes are session-scoped only; the `PARTICIPATED_IN` edge needs the user too).

**Live-verified, real Postgres + Qdrant + Neo4j + Ollama together** (`test_execute_scores_a_failure_episode_more_salient_than_a_routine_one`): a captured episode's `Episode` node, its `PARTICIPATED_IN` edge, and its `MENTIONS` edge (from the same `content["entities"]` field Batch C's `EntityRetrieval` already reads) all land in real Neo4j in the same call that writes to Postgres and Qdrant, verified by a real spreading-activation traversal reaching the episode from the entity it mentioned.

**A real bug this project's own review process caught before it shipped:** `link_mentions`'s original Cypher `MERGE`d the `Entity` node *before* matching the `Episode` it links from — so a transient failure in the preceding `upsert_episode_node` call (which `best_effort_graph_write` catches and logs rather than propagating) still left a persisted, edge-less `Entity` node behind, silently, with no error to surface it. Fixed by reordering to `MATCH` the episode first, matching the "both endpoints must already exist" pattern the other two edge-writing methods already followed.

## Deliberately not written by this batch

- **`RELATED_TO` (entity-to-entity)** — no existing command produces entity relatedness as a natural side effect the way capture/record/consolidate produce their respective node types.
- **`Concept` nodes** — part of the documented schema, but nothing in this batch's scope (episode capture, fact recording, consolidation) distinguishes "an abstract concept" from "an entity" as a separate write target. Disclosed explicitly in the design spec after a review found the original version's scope list silent on it.
- **Graph-write transactional consistency** — a Neo4j write failure is logged and swallowed (`best_effort_graph_write`), not retried or reconciled with the Postgres/Qdrant writes that already succeeded. This project has no outbox/saga mechanism; building one is out of scope for two issues about building and querying a graph. The swallow path itself now has direct unit-test coverage (it had none before a review caught the gap) proving a write failure genuinely doesn't propagate.
- **Automatic natural-language parsing into `start_entity_names`** — same explicit-parameters boundary Batch C established for its own five retrieval strategies; a caller supplies which entity to start from, this batch doesn't parse a free-text query to find it.

## What the review process caught, end to end

Two full review rounds (an initial six-dimension adversarial review, then a scoped re-review of that round's own fix wave) confirmed 19 total findings before merge — 14 in the first round (a dangling-node Cypher ordering bug, missing numeric-parameter validation on `spread_activation`, an unspecified-tie-order bug in a Batch C repository method this batch's `CaptureEpisode` change started relying on, four missing cross-tenant tests, and several documentation/infrastructure gaps), and 5 more in the second round — all of them weaknesses in the *first* fix wave's own new test coverage (two vacuous cross-tenant assertions that would have passed even if tenant scoping were removed entirely, an exception-swallow test that never proved the exception actually ran, a missing sixth cross-tenant test, and an untested validation boundary). 159 integration tests pass after both fix waves, up from 149 before this batch — no previously-passing test regressed.
