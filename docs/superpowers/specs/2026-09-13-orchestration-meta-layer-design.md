# Orchestration Meta-Layer Batch A: Paradigm Router, Context Budget Allocator, Latency-Adaptive Fallback Cascade — Design Spec

**Scope:** the three per-query components of the orchestration meta-layer — `docs/architecture/OVERVIEW.md`'s "The orchestration meta-layer's five components", "Slicing the context window", and "The latency-adaptive fallback cascade in detail", sourced from `docs/inputs/concepts/unified_rag_cag_mag_architecture.md` Concepts 1, 3, and 5 and composed the way that document's §3.4 Pattern 1 ("The Smart Router") composes them. Issues: Epic #150; Stories #151 (Paradigm Router), #152 (Context Budget Allocator), #153 (Latency-Adaptive Fallback Cascade), #154 (combined capability validation), and #155 (Freshness-Aware Data Router, Batch B); Tasks #156–#167, one per plan task in order. The Freshness-Aware Data Router (Concept 9) is the next batch, not this one: it decides once per data source at ingestion time, which is a different moment, a different caller, and a different failure mode from the three per-query decisions below.

## Why these three are one batch

The three cross-paradigm batches already built every pairwise mechanism the meta-layer coordinates — Cache-Warmed RAG, State-Aware RAG, three tiering boundaries, and the Sync Mixer's three tiebreak rules — but nothing in `src/` yet decides, for one incoming query, which of those paradigms to ask, in what order, for how long, and how much of the context window each answer may occupy. Those three decisions cannot be built or measured separately in any meaningful way. The cascade's order is only safe if something has already ruled out paradigms that would answer wrongly; the allocator's reallocation rules are keyed on outcomes only the cascade produces ("if CAG cache misses → RAG slice temporarily expands"); and the router's value is only visible in what the cascade does differently because of it.

## The claims this batch exists to test

The concept source makes three claims that are cheap to repeat and easy to get wrong, and the batch is built so that each one produces a number rather than an assertion.

**The router prevents answers the cascade alone would get wrong.** Concept 1's own table says "What changed in the policy today?" routes to RAG because it "needs latest external data", and an unconditional CAG → MAG → RAG cascade does exactly the wrong thing with that query whenever an older version of the policy is sitting warmed in CAG: the CAG tier reports a confident hit and the cascade returns immediately, which is what Concept 5 says a hit should do. The measurement is an ablation over the same cascade with the router switched off and on, against a corpus where CAG holds a superseded document and RAG holds its replacement. If routing does not change which answer comes back, the router is not earning its place in the pipeline.

**The tier budgets are achievable on this stack.** OVERVIEW.md is explicit that its 10ms / 50ms / 2s timeouts are "the targets the architecture commits to, not results already achieved". Each tier's latency is measured against its own budget, and the cost of embedding the query — work that happens before any tier runs and that every tier depends on — is measured and reported separately instead of being hidden inside whichever tier happens to run first.

**Classification must not cost more than it saves.** A router that takes several hundred milliseconds to decide makes a 10ms cache tier irrelevant. Three classifiers are built and measured on the same held-out query set — a lexical cue matcher, an embedding nearest-prototype classifier, and an LLM classifier through the project's local Ollama model — reporting accuracy and latency together, because the source never says how the classification is performed and the trade-off between those two numbers is the actual design decision.

## Domain (`src/orchestration/domain/`)

All of the logic below is pure: no I/O, no framework imports, and no wall-clock reads.

**New entities** (added to `entities.py`):

- `Paradigm` — an enum with members `CAG`, `MAG`, and `RAG`.
- `RoutingMode` — `CASCADE` when the router is confident, `PARALLEL` when it is not.
- `RoutingDecision` — `paradigms: frozenset[Paradigm]`, `mode: RoutingMode`, and `scores: dict[Paradigm, float]`. The scores are kept so reports and callers can see why a decision was made.
- `TierOutcome` — `HIT`, `PARTIAL`, `MISS`, `TIMEOUT`, or `ERROR`.
- `ContextItem` — `paradigm`, `content: str`, `score: float`, `source_id: uuid.UUID | None`. This is the one shape every tier's contribution is normalized into before assembly.
- `TierResult` — `outcome` (`HIT`, `PARTIAL`, or `MISS` only) and `items: list[ContextItem]`, which is what a tier itself returns.
- `TierAttempt` — `paradigm`, `outcome`, `elapsed_ms`, which is what the cascade records for each tier it tried, including timeouts and errors a tier never got to report itself.
- `CascadeResult` — `items`, `attempts`, `satisfied: frozenset[Paradigm]` (the paradigms whose tier reported a `HIT`), `degraded: bool`, and a derived `contributing` property (the paradigms that returned at least one item).
- `BudgetShares` — the five fractions, defaulting to OVERVIEW.md's 0.40 / 0.25 / 0.20 / 0.10 / 0.05, and validated in `__post_init__` to be non-negative and to sum to 1.0.
- `BudgetAllocation` — integer token counts for `cag`, `mag`, `rag`, `query`, and `reserve`, plus `total`.

**`paradigm_router.py`** — `decide(scores, select_threshold=0.5, uncertainty_margin=0.15) -> RoutingDecision`. Scores are independent per-paradigm values in [0, 1], because Concept 1's own table has multi-paradigm routes ("Compare today's sales with last month" → RAG + MAG) that a single-label classifier cannot express. Every paradigm scoring at or above the threshold is selected; if none is, the highest-scoring paradigm is selected anyway, so a decision is never empty. The decision is confident only when no paradigm's score falls within the margin of the threshold on either side. When it is not confident, the decision widens to include every paradigm inside that band and switches to `PARALLEL` — Concept 1's "if classifier is uncertain, run parallel and merge", applied only to the paradigms that are genuinely in doubt rather than to all three. A tie on the highest score is broken by the fixed CAG → MAG → RAG order, the same cheapest-first order the cascade uses, so the result is deterministic.

**`budget_allocator.py`** — `allocate(total_tokens, contributing, shares=DEFAULT_SHARES) -> BudgetAllocation`. Base slices are floored from the shares, and the flooring remainder goes to `reserve`. Every paradigm not in `contributing` donates its whole base slice, once, to recipients that are contributing, following each of the source's reallocation rules literally:

- **An idle RAG slice** goes to MAG if MAG is contributing, otherwise to CAG. This is Concept 3's "if query is simple (no RAG needed) → MAG slice expands".
- **An idle MAG slice** is split between whichever of CAG and RAG are contributing, in proportion to their base shares. This is "if session is new (no MAG state) → CAG or RAG slice expands"; the source says "CAG or RAG" without choosing, and splitting in proportion is the reading that doesn't invent a preference the source never states.
- **An idle CAG slice** goes to RAG if RAG is contributing, otherwise to MAG. This is "if CAG cache misses → RAG slice temporarily expands"; "temporarily" holds by construction, because an allocation is computed per turn and nothing carries over to the next one.
- **Any slice with no eligible recipient** goes to `reserve`, so the five slices always sum exactly to `total_tokens`.

Donations are computed from base slices only, never from slices that were already increased by another donation, which makes the result independent of the order the rules are applied in. The source's fourth rule — "if MAG state grows too large → trigger compression or eviction" — deliberately never expands a slice. It lives in context assembly below, where an over-budget paradigm's lowest-scoring items are dropped rather than being given room that belongs to another paradigm.

**`similarity.py`** — `cosine_similarity`, moved out of `CacheWarmedRetrieve` so the CAG tier's matching and the prototype classifier share one implementation instead of two copies.

**`errors.py`** — `SessionNotFound`, raised when a budget record matches no session row, and `QueryExceedsBudget`, raised when a question alone is larger than the Query slice.

**New ports** (added to `ports.py`):

- `QueryClassifier.score(query, query_embedding) -> dict[Paradigm, float]` is async, because the LLM implementation does network I/O, and it receives the embedding the use case already computed so the prototype classifier never embeds the same query twice. This follows the codebase's own rule that the split between sync and async ports tracks whether a real implementation performs I/O.
- `CascadeTier` has a `paradigm` property and an async `attempt(request: TierRequest) -> TierResult`. `TierRequest` carries `tenant_id`, `user_id`, `session_id`, `query`, and the `query_embedding`, which is computed once upstream.
- `SessionBudgetRecorder.record(tenant_id, session_id, allocation, contributing)` is async because it writes to Postgres.

## Application (`src/orchestration/application/`)

**`latency_cascade.py` — `LatencyCascade.run(request, decision) -> CascadeResult`.**

- *Eligibility.* The tiers eligible to run are the routed paradigms plus RAG as a universal last resort, ordered CAG → MAG → RAG. A paradigm the router left out is never attempted — that exclusion is exactly what keeps a freshness query away from the frozen cache. RAG is always eligible because Concept 5's cascade always ends in "return comprehensive answer"; a routed CAG or MAG miss has to fall through to something more thorough rather than end with nothing.
- *Stopping, in `CASCADE` mode.* Tiers are attempted in order. A `HIT` satisfies its own paradigm. The cascade stops as soon as every routed paradigm has been satisfied, so a CAG-only route that hits returns immediately (Concept 5's "Hit? → Return answer immediately"), while a RAG + MAG route runs both. A `PARTIAL` result contributes its items without satisfying its paradigm, so the cascade continues to RAG ("if CAG has a partial match → use it + supplement with RAG"). RAG runs whenever any routed paradigm is still unsatisfied by the time the cascade reaches it.
- *`PARALLEL` mode.* Every eligible tier runs concurrently, each under its own timeout, and every item from a tier that did not time out or fail is merged into the result.
- *Unrouted mode.* `run(request, decision=None)` is Concept 5's cascade exactly as the source draws it, with no router in front: CAG, then MAG, then RAG, stopping at the first `HIT`. It exists because it is the ablation baseline the router has to beat, and because a caller with no classifier configured still gets the source's own cascade rather than an error.
- *Timeouts.* The default budgets are `TierTimeouts(cag=0.010, mag=0.050, rag=2.0)` seconds, enforced with `asyncio.wait_for`. A timeout records `TIMEOUT` and moves on. A tier that raises records `ERROR` and moves on — one broken paradigm degrades the answer rather than failing the request — but `asyncio.CancelledError` is always re-raised, never swallowed.
- *Degraded results.* `degraded` is true whenever any tier the cascade attempted ended in `TIMEOUT` or `ERROR`. When RAG times out after earlier tiers already contributed items, those items are returned as a best-effort answer, which is Concept 5's "if RAG is slow → return CAG/MAG best-effort".
- *Background completion.* Concept 5's "+ async update" is implemented honestly at this layer's level: when RAG times out, its task is shielded rather than cancelled, and it keeps running in the background. When it completes, its findings go to an optional `on_rag_findings` callback, which is how a caller writes them into MAG "for next time" (§3.4 Pattern 1). The cascade keeps a reference to every background task so none is garbage-collected mid-flight, and exposes `drain()` so tests and shutdown can wait for them. This is an in-process task, not a durable job: `src/workers/` does not exist yet, so a process exit loses any update still in flight. That limitation is disclosed here rather than papered over.
- *Access recording.* On a RAG hit, an optional `AccessFrequencyTracker` records an access for each document RAG returned. This is Pattern 1's "if result was frequent → flag for CAG pre-loading", and it feeds the Batch D `WarmCache` and `TieringPolicy` that already exist, instead of building a second frequency-counting mechanism.

**`cascade_tiers.py`** adapts existing use cases to the `CascadeTier` port. It adds no new retrieval logic of its own.

- *`CagTier`* uses `CacheWarmedRetrieve`'s warmed-document match. That match is currently private and embeds the query itself; this batch extracts it as a public `best_warmed_match(tenant_id, query_embedding) -> tuple[SearchResult, float] | None`, which keeps the existing FrozenCache content-hash confirmation intact. `CacheWarmedRetrieve.execute` is rebuilt on top of the same method, so its behavior and its existing tests are unchanged. A score at or above `hit_threshold` is a `HIT`; a score at or above `partial_threshold` is a `PARTIAL`; anything lower is a `MISS`. The matching is CPU work, so it runs under `asyncio.to_thread`, which makes the 10ms timeout able to fire instead of being blocked by a stalled event loop.
- *`MagTier`* runs `FindSemanticFacts.by_similarity` over the precomputed embedding. Invalidated and archived facts never reach it, because `PostgresSemanticMemoryRepository.search_by_similarity` already filters on `valid_until` and `archived_at`, which is Concept 5's "if MAG has stale state → invalidate". Facts scoring below `partial_threshold` are discarded. If the best remaining fact scores at or above `hit_threshold`, the result is a `HIT`; otherwise the remaining facts make a `PARTIAL`, and nothing remaining is a `MISS`.
- *`RagTier`* wraps any RAG `Retriever`. A non-empty result is a `HIT` and an empty one is a `MISS`, since RAG has no notion of a partial hit — the fallback tier either finds something or it does not.

Every threshold is a required constructor parameter with no default. The values the integration tests and the evaluation runners pass are taken from score distributions measured on this batch's own corpus and recorded in the report, so no threshold in `src/` is a guess presented as a default.

**`assemble_context.py`** — `assemble_context(items, allocation) -> AssembledContext`. It groups items by paradigm and packs each group into its own slice in descending score order, counting tokens with `src/shared/tokenization.count_tokens`. It uses the skip-and-continue walk that `TokenBudgetAllocation` and `CompressingRetriever` already established, so an oversized item never stops a smaller one from fitting. Anything dropped is reported per paradigm, which is how the "state grows too large" rule is both enforced and made visible. The rendered context labels each section by paradigm, in CAG, MAG, RAG order, so a reader of any single turn can see which paradigm contributed what.

**`unified_answer_question.py`** — `UnifiedAnswerQuestion.execute(tenant_id, user_id, session_id, question) -> UnifiedAnswer`. It runs these steps in order:

1. Reject a question whose own token count exceeds the Query slice with `QueryExceedsBudget`, then embed it once, timed.
2. Classify the query and decide the route, timed.
3. Run the cascade.
4. Allocate the budget from the cascade's `contributing` set.
5. Assemble the context.
6. Generate the answer with the RAG `ChatModel`.
7. If a `SessionBudgetRecorder` was supplied, record the allocation.

`UnifiedAnswer` carries the answer, its sources, the routing decision, every tier attempt, the allocation, the per-paradigm dropped counts, `degraded`, and the per-stage timings. That is everything the evaluation report needs, without the report having to reach into internals.

## Infrastructure (`src/orchestration/infrastructure/`)

- **`LexicalQueryClassifier`** matches cue phrases for each paradigm — freshness cues such as "today", "latest", and "changed" for RAG; session cues such as "we discussed", "remember", and "continue" for MAG; static-reference cues such as "policy", "guide", and "how do I" for CAG — and converts match counts to scores. It is deterministic, takes microseconds, and serves as the floor the other two classifiers have to beat.
- **`PrototypeQueryClassifier`** embeds the query and scores each paradigm with a similarity-weighted vote over its k nearest labeled exemplars. An exemplar's label is a set of paradigms, so multi-paradigm exemplars vote for every paradigm they carry. It uses the existing `EmbeddingModel` port, so no new model dependency is introduced.
- **`LlmQueryClassifier`** asks the `ChatModel` for a JSON object of per-paradigm confidences. If the response doesn't parse — the same failure #149 found in the project's own judge — every paradigm scores 0.5, which falls inside the uncertainty band and therefore routes `PARALLEL` across all tiers. An unparseable classification becomes the safest possible route rather than a crash or a silent guess.
- **`PostgresSessionBudgetRecorder`** writes `{total, slices, contributing, recorded_at}` into `sessions.context_budget`. That JSONB column has existed since `alembic/versions/0001_users_sessions.py` and `docs/database/DATABASE.md` already describes it as "the per-session record of how the 128K context window gets sliced", so no migration is needed — this is the first code to use the column. The recorder calls `set_tenant_context` itself before its `UPDATE`, so the table's `tenant_isolation` RLS policy enforces the tenant boundary on every write. An update that matches no row, whether the session is missing or belongs to another tenant, raises `SessionNotFound` instead of passing silently.

## Tenant and user isolation

Every `TierRequest` carries both `tenant_id` and `user_id`, and every existing port a tier calls already enforces its own scope: `FrozenCache` and `CacheWarmedRetrieve` are tenant-scoped (a Batch D review fix), MAG semantic search is scoped by user and tenant, and the RAG retriever is tenant-scoped. No tier caches anything across requests, so the meta-layer introduces no new shared state that could leak between tenants. The unit tests cover this explicitly: two tenants, and two users within one tenant, issue the same query and each gets back only their own items.

## Testing plan

Unit tests use fakes, are deterministic, and never assert on wall-clock latency, because latency assertions are flaky in CI. Timeout behavior is tested with fake tiers that sleep far beyond their budget, such as 300ms against a 10ms timeout, so the expected outcome class is unambiguous.

- `paradigm_router.decide` is tested against the score shapes of all six Concept 1 example queries, plus threshold boundaries, the never-empty fallback, tie-breaking, and the widening of `PARALLEL` to exactly the paradigms inside the uncertainty band.
- `budget_allocator.allocate` is tested for its sum invariant across many totals and every one of the eight possible contributing sets, for each reallocation rule on its own, for order independence, for the reserve fallback, and for share validation.
- `LatencyCascade` has tests for:
  - a CAG-only hit short-circuiting before MAG or RAG runs
  - the unrouted mode stopping at the first `HIT` in CAG → MAG → RAG order
  - a CAG miss falling through to RAG
  - a RAG-only route never touching CAG or MAG
  - a partial CAG match supplemented by RAG
  - a RAG + MAG route running both tiers
  - a timeout recording `TIMEOUT` and continuing
  - an exception recording `ERROR` and continuing
  - `CancelledError` propagating
  - `PARALLEL` merging every eligible tier
  - best-effort degraded results
  - the background RAG task completing into `on_rag_findings` after `drain()`
  - access recording on a RAG hit
- The three tiers are tested for their threshold classification and scope propagation, and `CacheWarmedRetrieve`'s existing tests pass unchanged after the extraction.
- `assemble_context` is tested for packing order, skip-and-continue, per-paradigm dropped counts, and section labeling.
- The three classifiers are tested with fake embedders and fake chat models, including the LLM parse-failure fallback.
- `UnifiedAnswerQuestion` is tested for stage wiring, for the allocation reaching the recorder, and for the timings being populated.

Integration tests use real dependencies:

- **`PrototypeQueryClassifier` with the real MiniLM model.** It checks that real embeddings produce well-formed scores and that a lightly reworded exemplar routes to its own label. Accuracy on held-out queries is deliberately not a test gate, since gating on it would invite tuning the exemplars until the evaluation queries pass; it is measured in the report instead.
- **`PostgresSessionBudgetRecorder` against real Postgres through TestContainers.** It checks the JSONB round-trip and that RLS refuses a write under the wrong tenant, and it needs only Docker, so it runs in every local integration pass.
- **The full cascade** with real Postgres, Qdrant, MiniLM, and `HFFrozenCache` (`distilgpt2` on CPU). It covers the routed-away stale-CAG case, a real CAG hit short-circuit, and a real MAG hit. It needs no LLM, so it runs in every local integration pass. (This repository's only GitHub Actions workflow is `repo-hygiene.yml`, which checks links, placeholders, and secrets; the test suite runs locally.)
- **`LlmQueryClassifier` and end-to-end generation against the local Ollama model.** These skip with a stated reason when Ollama is unreachable, following the precedent set by the vLLM tests.

## Measurement plan

Runners live under `evaluation/scenarios/orchestration-meta-layer/`, and the narrative report goes to `evaluation/reports/orchestration-meta-layer.md`.

1. **Router comparison.** The held-out query set is generated by the local `qwen3.5` model from each route's description, and only after the three classifiers, their cue lists, and their exemplars are committed, so its phrasings were not written by the author of the cues. Every generated label is then reviewed by hand, and any correction is recorded in the file rather than made silently. All six Concept 1 table queries are appended verbatim, and none of them appears among the exemplars. Each classifier is reported on exact route-match accuracy, per-paradigm precision and recall, confident rate, and classification latency at p50 and p95. The report still states plainly that the route descriptions, the label review, and the classifiers share one author, so these numbers are an in-distribution estimate, not a measurement of real traffic.
2. **Router ablation.** The same cascade runs with and without the router on a support corpus in which CAG holds a superseded policy document and RAG holds its replacement. The report gives the stale-answer rate under each condition.
3. **Tier latency against budget.** For each tier, p50 and p95 latency are compared with its 10 / 50 / 2000ms budget, alongside the counts of each outcome. Query-embedding cost is reported as its own line.
4. **Self-versus-self comparison** through the existing `RunComparison`. The baseline is RAG-only `AnswerQuestion`; the treatment is `UnifiedAnswerQuestion`, on the same model and the same questions. The report covers latency, token counts, task success, and the four-dimension judge scores.
5. **Allocator effect.** Per-slice token use and dropped-item counts are compared between dynamic reallocation and static base slices on the same turns. The report claims a quality effect only if the judge scores in item 4 actually show one.

## What this batch does not do

- **It does not build the Freshness-Aware Data Router.** That component is the next batch, because it decides at ingestion time rather than per query.
- **It does not change any HTTP endpoint.** `POST /chat` stays RAG-only. Exposing the unified path means accepting a `session_id` from a client, which is an object-level authorization surface (OWASP API1) that deserves its own batch and its own security review, rather than being added as a side effect of this one.
- **It does not put real vLLM behind the CAG tier.** The tier checks `HFFrozenCache`, the same CPU proxy Batch D used. The generation-side speedup on a real cache hit is already measured on real vLLM in `evaluation/reports/cag-prefix-caching.md`; what this tier contributes is the lookup decision and its latency.
- **It does not make the background RAG update durable.** Durability arrives with `src/workers/`.
- **It does not tune routing thresholds against real traffic.** No real traffic exists yet, so the thresholds are measured defaults, disclosed as such.
