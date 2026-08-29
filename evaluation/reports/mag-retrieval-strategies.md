# MAG Batch C: Retrieval Strategies — Live Measurement Report

**Scope:** #12 (parent) and its six children — #67 semantic similarity, #69 temporal, #70 causal, #72 entity-based, #74 salience scoring, #75 recency-decay fusion. Builds on Batch A's storage foundation and Batch B's consolidation pipeline — see `docs/superpowers/specs/2026-08-28-mag-retrieval-strategies-design.md` for the design and `docs/architecture/MAG.md` for the underlying concepts.

## The score-carrying fix this batch resolves

Batch A's first review flagged that every `search_by_similarity`/`search` implementation across MAG discarded the similarity score entirely, deferring the fix to "whichever later batch needs multi-strategy score fusion." This batch is that batch: `ScoredEpisode`/`ScoredFact` now carry a real score everywhere, and Postgres computes cosine *similarity* (`1 - pgvector's <=> distance`) specifically so it lands on the same scale Qdrant already returns natively — a deliberate choice, not an accident, since fusion's per-strategy min-max normalization only produces a meaningful ranking if the same nominal strategy's score means the same thing regardless of which backend supplied it.

## The six strategies: correctness, not a quality comparison

Same shape as Batch A/B's storage sections — there's no baseline/treatment split for "does this strategy retrieve what it claims to retrieve." Each is verified against real Postgres (and Qdrant, for semantic similarity) via testcontainers, with tenant isolation and — after a review caught the gap (see below) — same-tenant session isolation covered for every new repository method.

- **Semantic similarity (#67):** formalizes the existing Qdrant-backed search as a named strategy now that it carries a real score. Session-scoped as of the fix wave below (see "What the review caught").
- **Temporal (#69):** an explicit `within` window scores every match uniformly (binary relevance); no window falls back to recency with linear rank-decay scoring, so fusion has a graded signal even without an explicit window.
- **Entity-based (#72):** matches a structured `content["entities"]` hit or a serialized-content substring fallback, scored uniformly — this system has no per-mention confidence to grade matches by, so it doesn't invent one.
- **Salience (#74):** ranks by the episode's own `salience_score` directly.
- **Causal (#70):** one batched LLM call scores every candidate episode's causal relevance to a query in a single round trip, with the same bounded-retry/validate-inside-the-loop/fail-safe pattern established in Batch B's Consolidation.

## Salience scoring: the data problem found while designing, not after

`salience_score` has existed on `EpisodicMemory` since Batch A, but nothing had ever written a non-default value to it — `CaptureEpisode` always saved `0.0`. A `SalienceRetrieval` strategy built on top of that would have been real, tested code that always returned an arbitrarily-tied ordering in practice: correct but hollow. This batch gives `CaptureEpisode` a real salience-scoring step (an LLM call via the existing `ChatModel` port, same retry/validate/fallback shape as everywhere else).

**Live measurement** (`test_execute_scores_a_failure_episode_more_salient_than_a_routine_one`, real Postgres + Qdrant + `qwen3.5` via Ollama):

```
Salience scoring (real Ollama, qwen3.5): failure episode=0.98, routine episode=0.1
```

A constructed episode containing a traceback and a failed deploy scored 0.98; a routine weather question scored 0.1 — matching #74's own framing ("weights critical decisions or failures more heavily than routine turns") with a real model's judgment, not a scripted response.

## Causal retrieval: a real classification, live-measured

**Live measurement** (`test_execute_ranks_real_postgres_episodes_using_a_real_ollama_model`, real Postgres + `qwen3.5`):

```
Causal relevance scores from a real Ollama model for query 'why did it fail':
  causal episode (traceback + root cause + fix): 1.0
  unrelated episode (goldfish names): 0.0
```

## Recency-decay fusion: orchestration, not a sixth strategy

Per #75's own text ("combining the outputs of the other strategies rather than acting as an independent one"), `RecencyDecayFusionRetrieval` runs whichever of the other five strategies the caller supplied enough parameters for (temporal and salience always run; semantic/causal/entity run only when given a query embedding, a causal query, or an entity respectively — no automatic query decomposition, which is deliberately left to the orchestration layer `OVERVIEW.md` describes, not MAG's own retrieval strategies), min-max normalizes each strategy's scores independently, applies uniform half-life recency decay, and sums weighted contributions per episode — rewarding cross-strategy agreement and deduping overlapping hits in the same step.

**Live measurement, full pipeline** (`test_a_real_fusion_pass_ranks_a_causally_relevant_recent_episode_above_a_routine_one`, real Postgres + Qdrant + `qwen3.5`, all five legs running):

```
Fusion (real Postgres+Qdrant+Ollama, qwen3.5): failure=1.0000, success=0.0000
```

**A gap this project's own review process caught in this test before it shipped:** the scenario above has every other strategy (salience, recency, semantic similarity, entity match) independently favoring the same episode by a wide margin, so it couldn't actually prove the causal leg was doing anything — a completely broken causal call falling back to its flat 0.0 tie would have passed this exact assertion too. A second live test isolates causal via `weights={"temporal": 0.0, "salience": 0.0, "causal": 1.0}` with no `query_embedding`/`entity` given at all, so only the real Ollama causal judgment can differentiate the two episodes:

```
Fusion causal-isolated (real Postgres+Ollama, qwen3.5): failure=1.0000, None=0.0000
```

(The `None` label is a cosmetic artifact of the print statement's `content.get('outcome')` — the second fixture episode's content never sets an `outcome` key — not a bug in the measurement itself; the assertion and the underlying scores are correct.)

## What the review caught

A full adversarial review (six dimensions, each finding independently verified by two skeptic passes) confirmed 11 real findings before merge, most seriously:

- **A real cross-session data leak.** `SemanticSimilarityRetrieval` (and the `EpisodicMemoryIndex.search`/Qdrant path it wraps) filtered only by `tenant_id`, not `session_id` — directly contradicting this batch's own design spec, which explicitly claimed all five non-fusion strategies were session-scoped. Composed into the session-scoped `RecencyDecayFusionRetrieval`, this meant one user's session could surface another session's episodic memories within the same tenant. Fixed at the port level (Qdrant's `query_filter` now requires both fields), not by filtering results after the fact.
- **An unescaped ILIKE wildcard.** `get_by_session_matching_entity`'s substring fallback built its pattern with plain f-string interpolation of the caller-supplied entity string — no SQL injection (every value was correctly bound), but `%`/`_` in an entity string were interpreted as real LIKE wildcards, silently over-matching unrelated episodes.
- **A missing validation gap Batch B's own pattern should have caught here too.** `CaptureEpisode`'s new salience validator lacked the `isinstance(x, bool)` guard `CausalRetrieval`'s validator already has — `float(True) == 1.0` in Python, so a model returning `{"salience_score": true}` would have silently passed as a maximal score with no retry.
- **A test-coverage gap in the batch's own new repository methods.** Every integration test for the four new query methods populated exactly one session per tenant, so the `session_id` filter — a plain application-level `WHERE` clause with no RLS backstop, unlike `tenant_id` — was never actually exercised against a sibling session under the same tenant. Closed with four new tests proving each method doesn't leak across sessions within one tenant.

A scoped re-review of the fix wave itself found one more candidate issue (whether leaving Postgres's `search_by_similarity` tenant-wide-only was an inconsistent half-fix) — adversarial verification correctly refuted it as a deliberate, pre-existing, differently-scoped code path from an earlier batch, not a bug, but the reasoning had never been written down; documented on the port directly rather than left implicit.

137 integration tests pass against real Postgres, Qdrant, Redis, and a real Ollama model after the full fix wave — up from 130 before this batch, all previously-passing tests still green.

## What this batch does not do

- **No automatic query decomposition.** Turning a free-text query like "why did the deployment fail last Tuesday" into a temporal window, an entity string, and a causal query automatically is a query-understanding capability the design spec places at the orchestration layer (`OVERVIEW.md`'s meta-layer that decides which paradigm and data source answers a query at all), not inside MAG's own retrieval strategies. Every strategy here takes explicit, structured parameters instead.
- **No cross-session retrieval.** All five non-fusion strategies — including semantic similarity, after the fix above — scope their candidate set to one `session_id`, matching the worked example's single ongoing-conversation framing. A user-wide "what do I know about User X across all sessions" variant is a plausible future extension, not what this batch's issues asked for.
- **No learned fusion weights.** `weights` in `RecencyDecayFusionRetrieval` is caller-supplied heuristic weighting only, matching #12's "learned *or* heuristic weights" — a mechanism that trains weights from feedback has no home in the current architecture yet.
