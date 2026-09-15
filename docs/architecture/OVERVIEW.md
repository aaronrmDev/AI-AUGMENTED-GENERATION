# Architecture Overview

This document is the detailed architecture and technology-stack reference behind CLAUDE.md's rules — the layered system design, the orchestration meta-layer that ties RAG, CAG, and MAG together, the context-budget math, the fallback cascade, three synthesis techniques this project commits to by name, the full technology stack, what that stack costs to run, where it's fine to substitute a different tool, and the concrete module layout this codebase has been building toward since implementation began. Everything here traces to CLAUDE.md's own rules or to the two source documents it was synthesized from — `docs/inputs/concepts/unified_rag_cag_mag_architecture.md` (the orchestration-concepts document) and `docs/inputs/concepts/fullstack_unified_ai_system.md` (the stack-and-repository-layout document) — and where a claim isn't something either of those states outright, that's said plainly rather than left to look like settled fact.

## The layered architecture: three parallel paradigms under one coordinator, all resting on the LLM core

```text
LAYER 4: ORCHESTRATION (Meta-Layer)
  Paradigm Router · Context Budget Allocator · Latency-Adaptive Fallback Cascade · Sync Mixer · Freshness-Aware Data Router
────────────────────────────────────────────────────────────────
LAYER 3: MAG (State Layer)          — session memory, highly stateful, writes continuously
LAYER 2: CAG (Cache Layer)          — GPU-resident cache, pre-baked, invalidates in batches
LAYER 1: RAG (Retrieval Layer)      — external index, stateless, updates instantly
────────────────────────────────────────────────────────────────
FOUNDATION: LLM Core (inference engine + context window, shared by every layer above)
```

Both source documents draw this same four-layer stack, numbered identically (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md` §4.2, "Layer Architecture"; the original project brief's §2.1). The foundation earns its position for a literal reason the source states directly: the LLM's context window and inference engine are "shared by all layers above" it — CAG's frozen KV cache lives inside that engine's GPU memory, RAG's retrieved chunks get injected into that same context window, and MAG's session state gets read back into it every turn. Nothing above the foundation can operate without it, so it sits underneath everything else by necessity, not by convention.

The three paradigm layers above the foundation are drawn as a vertical stack — RAG at Layer 1, CAG at Layer 2, MAG at Layer 3 — but they are not dependent on each other the way layers usually are in a stack diagram: RAG doesn't need CAG to function, and CAG doesn't need MAG. `fullstack_unified_ai_system.md`'s own system diagram (§1, "High-Level Architecture") actually draws them side by side as three parallel "PARADIGM LAYERS" boxes sitting under a single Orchestration layer, rather than stacked on top of one another — which is the more literal picture of how they relate. Read against the numbering, though, the vertical order does track something real: it follows the same progression the paradigm comparison table uses for "Operational State" — RAG is "Completely Stateless," CAG is "Pre-baked / Frozen," and MAG is "Highly Stateful & Mutating" (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md` §1). Read bottom-to-top, the stack goes from the paradigm with the least memory of its own (RAG, which looks nothing up until asked) to the one frozen in place between updates (CAG) to the one that rewrites itself on every turn (MAG). That reading is this document's own inference from the source's comparison table, not something either source states as the explicit rationale for the diagram's order — but it's the only account of the ordering that's consistent with both diagrams at once.

Orchestration sits above all three for a reason the source is explicit about: it is the "meta-layer" that "decides which paradigm handles which piece of knowledge at which moment" (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md` §1). That decision is structurally impossible to make from inside any one paradigm layer — RAG's index has no way of knowing whether MAG's session state already answers the query, and CAG's frozen cache has no way of knowing whether RAG's index just got fresher. Only a layer positioned above all three, with visibility into all three, can route a query, divide the context budget, decide which paradigm to try first, and keep their answers from contradicting each other. The next section walks through the five components that do that job.

## The orchestration meta-layer's five components

Each of the five pieces below owns one specific decision. What they have in common is that the decision belongs at this layer specifically because it requires visibility across all three paradigms — something no single paradigm layer has on its own.

### Paradigm Router — deciding which paradigm answers a query

The router classifies an incoming query along several axes before it goes anywhere: how fresh the answer needs to be (real-time favors RAG, static favors CAG, session-only favors MAG), whether it references prior conversation (a MAG signal), how tight the latency budget is, and how complex the question is. A query like "What's our refund policy?" is static and asked often, so it routes to CAG; "What changed in the policy today?" needs data CAG's frozen cache can't have, so it routes to RAG; "Continue where we left off yesterday" is pure state retrieval with no external-knowledge component at all, so it routes to MAG alone; a query like "Compare today's sales with last month" needs both live external data and the session's own context, so it routes to RAG and MAG together (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 1). When the classifier isn't confident which paradigm applies, the source's answer is to run candidates in parallel and merge the results rather than guess. This decision has to sit above the three paradigms because none of them can see the others' contents to know whether they're even the right one to ask — CAG doesn't know if its cached answer is now stale, and RAG doesn't know a cheaper answer already exists in MAG.

This project builds the router as `decide` in `src/orchestration/domain/paradigm_router.py`, applied to scores from any `QueryClassifier` (`src/orchestration/domain/ports.py`). Scores are independent per paradigm, so the RAG + MAG route in the source's own table can be expressed. Every paradigm at or above the threshold is selected, and a score within the uncertainty margin of the threshold widens the route to `PARALLEL` — the source's "run parallel and merge". A classification in which nothing clears the threshold, or a classifier that times out or fails, routes every paradigm in parallel through `fallback_decision()` instead of guessing one: an early version that picked the single highest score sent queries with no signal confidently to CAG alone, the one route where a weak frozen-cache match answers with nothing to contradict it. Three classifiers implement the port — lexical cues, a nearest-prototype vote over MiniLM embeddings, and the local `qwen3.5` model — so the choice between them can rest on measured accuracy and latency rather than on the source, which never says how the classification is done. Measured on a held-out set of 46 routing queries (`evaluation/reports/orchestration-meta-layer-router.md`), the choice turned out to be a real trade-off rather than a clear winner. The local `qwen3.5` classifier routed 74% of queries exactly and covered the paradigms a query needed 85% of the time, but it took 18.9 seconds at the median and 60 seconds at p95 to decide — about 1,900 times, at the median, the 10ms budget of the cache tier it would sit in front of. Under `UnifiedAnswerQuestion`'s 2-second classifier timeout it therefore falls back to running every tier on at least half of all queries, on purpose. The lexical classifier decides in 0.11ms but routes only 59% of queries exactly. The MiniLM prototype classifier decides in 2.4ms and covers 83% of needed paradigms, but it is confident on only 37% of queries, so most of its routes run every tier in parallel. Embedding the query costs 9 to 11ms at the median across this batch's runs, before any classifier or tier runs, which by itself uses nearly all of CAG's budget. Whether routing earns its place was tested directly, against a frozen cache holding a superseded return policy (`evaluation/reports/orchestration-meta-layer-cascade.md`). For 4 of 6 freshness questions, the cascade as the source draws it gave the model only the superseded text. With routing on, that never happened. But the superseded text still reached the model alongside the current document in 3 of 6 lexically routed runs and 5 of 6 prototype-routed runs, whenever CAG answered alongside RAG. So routing removes the worst outcome without keeping stale text out of the context.

### Context Budget Allocator — deciding how much context window each paradigm gets

Even a large context window runs out if CAG tries to preload an entire corpus, MAG tries to hold fifty turns of history, and RAG tries to inject twenty retrieved chunks, all at once, on every request — the model ends up "lost in the middle" of a context stuffed with everything and organized by nothing (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 3). The allocator is the one component with a global view of the total budget and each paradigm's current need, so it's the only place that can arbitrate the fight over space before it happens. Its actual numbers are covered in their own section below. In this project the allocator is `allocate` in `src/orchestration/domain/budget_allocator.py`, a pure function computed once per turn from the paradigms whose tiers actually returned content. `src/orchestration/application/assemble_context.py` then packs each paradigm's items into its slice by score and reports whatever didn't fit. Each turn's allocation is written to `sessions.context_budget` by `PostgresSessionBudgetRecorder`, whose record shape and tenant and user scoping are described in `docs/database/DATABASE.md`. Measured across a sweep of window sizes (`evaluation/reports/orchestration-meta-layer-cascade.md`), reallocation changes nothing while every slice has room: at 1,000 and at 400 tokens, dynamic and static slices kept the same 616 tokens of this corpus's context. It starts to matter once static slices bind. At 250 tokens, static slices dropped 8 items and kept 461 tokens, while reallocation dropped none and kept all 616; at 150 tokens, static slices dropped 14 items and reallocation 6.

### Latency-Adaptive Fallback Cascade — deciding what order paradigms get tried in, and when to give up and move on

CAG, MAG, and RAG have wildly different latency profiles — CAG answers from a frozen cache in effectively no time, MAG's session lookups typically take 1–10ms, and RAG's external retrieval typically takes 50–500ms, with the cascade giving it up to a full 2s before giving up. Left to their own devices none of the three paradigms would know when to stop waiting on themselves and hand off to something faster or more thorough; a paradigm layer has no view of what the other tiers would cost the caller, so it can't make that trade-off itself. The cascade sits above all three specifically to enforce a shared timeout budget and decide, layer by layer, whether to return what it has or escalate. Its tiers and timeouts are covered in detail further down. In this project it is `LatencyCascade` in `src/orchestration/application/latency_cascade.py`. Its tiers in `cascade_tiers.py` adapt the existing Cache-Warmed RAG match, MAG semantic-fact search, and any RAG retriever, and `UnifiedAnswerQuestion` composes it with the router and the allocator. It runs in three modes: unrouted, exactly as the source draws it; routed, trying only the paradigms the router selected plus RAG as the last resort; and parallel, for an uncertain route. Building it against real Postgres surfaced a constraint the source never mentions. A timeout cancels a tier mid-query, and SQLAlchemy responds by terminating that query's connection, so every tier must own its unit of work rather than share the request's session; the MAG tier's search opens its own session per call for exactly this reason (`src/orchestration/infrastructure/session_scoped_semantic_fact_search.py`).

Clients reach the whole per-query path over HTTP through chat sessions (`src/api/routers/sessions.py`):

- `POST /sessions` creates a session for the caller.
- `GET /sessions` lists the caller's own sessions.
- `POST /sessions/{session_id}/answers` answers through `AnswerInSession`, which checks that the session belongs to the token's user before anything is embedded or retrieved. A session that's missing, another user's, or another tenant's gets the same 404.

The API composes the cascade with MAG and RAG tiers only (`src/api/unified_pipeline.py`). A CAG tier needs a warmed frozen cache in the serving process and a worker to warm it, and neither exists yet, so a query the router sends to CAG alone is answered from RAG. Answering shares a per-user rate limit with the RAG-only `POST /chat`. The batch's live run against uvicorn and Ollama, and its security review, are recorded in `docs/superpowers/plans/2026-09-13-unified-api-sessions.md`.

### Sync Mixer — deciding how three different update rhythms stay reconciled

RAG's index updates the instant a source document changes; CAG's cache invalidates in scheduled or event-driven batches; MAG writes to its memory tables on every turn (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 4). None of the three paradigms has a channel to tell the other two that the underlying fact they're both holding just changed — RAG's re-indexing has no built-in path to CAG's cache or MAG's session state. The mixer exists because that coordination has to happen somewhere, and the only place with visibility into all three stores at once is the orchestration layer. Its tiebreak rule for when the three paradigms actually disagree is covered in its own section below.

### Freshness-Aware Data Router — deciding, at ingestion, which paradigm should own a piece of data in the first place

This is a different decision from the Paradigm Router's, and it happens at a different time: the Paradigm Router decides, per query, which paradigm to consult; the Freshness-Aware Data Router decides, per data source, which paradigm should hold that data at all, based on how fast it changes. Stock prices and live scores change by the second and are "too volatile for cache," so they go to RAG only; company policies and manuals change monthly or quarterly and are "stable enough to pre-load," so they go to CAG; user session state changes every turn and "must be mutable per interaction," so it goes to MAG (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 9). This is a system-wide policy decision made once per data source, not a per-request routing decision, which is why it belongs at the orchestration layer rather than inside any single paradigm — a paradigm layer only sees the data it's already been given; deciding where new data should land in the first place requires seeing across all three. In this project the router's rules live in `src/orchestration/domain/freshness_router.py`, and three use cases apply them:

- **`IngestDataSource`** routes a source's first version by its declared change interval: under one day to RAG only, a day or longer to a CAG pre-load with RAG as backup, and anything user-scoped to MAG. On every later version it replaces RAG's copy and evicts the stale CAG entry at once. The change stays marked pending until all of that has landed, and the refresh and review below write only while nothing is pending, so a failed or concurrent step can't re-cache superseded text.
- **`RefreshCachedSources`** is the batch pre-load. It trusts a cached copy for half the source's expected change interval after the last ingestion that confirmed it. `ExpiringFrozenCache` enforces that expiry at lookup, so an expired entry falls through to RAG without waiting for a sweep.
- **`ReviewSourceFreshness`** re-learns each tenant source's interval from its version history in `data_sources` and `data_source_versions` (`docs/database/DATABASE.md`). It demotes a cached source after three changes inside three days, and promotes a RAG-only source after seven quiet days.

Measured over 30 simulated days against real stores (`evaluation/reports/freshness-router.md`):

- **Staleness.** Freshness-aware placement never assembled a context holding superseded text. Caching every source with only a nightly refresh did so on 87% of an hourly price feed's probes.
- **Stable sources.** They still answered from CAG: 98% of return-policy probes.
- **Personal data.** Placed in MAG, a user's personal fact never reached another user of the same tenant. Every tenant-wide placement exposed it on all 241 probes.

Invalidating cached copies on every change also removed staleness, with no routing at all, but only because the simulation delivers each change to the router in the hour it happens. With real ingestion lag, a cached copy serves superseded text until the change arrives. Against that baseline, routing's measured benefit was fewer wasted pre-loads, and isolation of personal data.

## Slicing the context window: the budget numbers and why they're shaped this way

The default allocation, against a 128K-token context window, is:

| Slice | Size | Content | Paradigm |
|-------|------|---------|----------|
| **CAG Slice** | 40% (51K) | Frozen pre-loaded docs, system prompt, static knowledge | CAG |
| **MAG Slice** | 25% (32K) | Session state, conversation history, user preferences | MAG |
| **RAG Slice** | 20% (26K) | Dynamically retrieved chunks per query | RAG |
| **Query Slice** | 10% (13K) | Current user query + instructions | — |
| **Reserve** | 5% (6K) | Buffer for generation output | — |

(`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 3; the original project brief's §2.2)

The shares trace directly back to each paradigm's own nature. CAG gets the largest slice because CAG's content is frozen and reused turn after turn at effectively zero marginal cost once it's cached — front-loading the budget with static content pays for itself across an entire session, unlike RAG's chunks, which have to be fetched and re-injected fresh on every call that needs them. MAG's 25% has to cover a session's worth of continuity — preferences, history, scratch state — without being allowed to grow unbounded, which is exactly why MAG's own architecture (covered in `docs/architecture/MAG.md`) includes consolidation and eviction rather than just letting this slice expand indefinitely. RAG gets the smallest of the three paradigm slices because it's invoked selectively through the router and the cascade rather than on every turn, so it doesn't need a standing allocation as large as CAG's or MAG's. The Query slice is close to non-negotiable — without room for the actual question and its instructions there's nothing to answer — and the Reserve exists so the model always has somewhere to put its output even if the four slices ahead of it filled the rest of the window.

The allocation adjusts turn by turn rather than staying fixed: the source's dynamic-reallocation rules give unused budget to whichever paradigm is likely to benefit most on that particular turn, rather than letting it sit idle in a slice nothing needs: if a query is simple enough that RAG isn't required, the freed RAG budget expands MAG's slice instead, since more room for conversational context costs nothing and improves continuity when retrieval wasn't going to be used anyway. If a session is brand new, MAG has no state to contribute, so the CAG or RAG slice expands to absorb what would otherwise be dead space. If CAG comes back with a cache miss, the RAG slice temporarily grows to compensate for the knowledge CAG couldn't supply. And if MAG's own state grows too large on its own, the source's answer isn't to keep expanding its slice indefinitely but to trigger compression or eviction instead (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 3). The original project brief's own condensed version of this rule stated the two most common cases directly: "if RAG is not needed, MAG expands; if MAG is empty, CAG expands" — both are specific instances of the same general principle, that a paradigm's slice grows when it has something to offer and another paradigm's slice has nothing to spend its budget on that turn.

## The latency-adaptive fallback cascade in detail

| Tier | Paradigm | Timeout | What happens at this tier |
|------|----------|---------|----------------------------|
| 1 | CAG | 10ms | Cache checked; a hit returns immediately from pre-loaded static knowledge |
| 2 | MAG | 50ms | Session state / conversation history checked; a hit returns a stateful answer |
| 3 | RAG | 2s | External retrieval always runs (no hit/miss — this tier does the full lookup); returns a comprehensive answer |

(`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 5; the original project brief's §2.3)

The cascade tries the cheapest paradigm first and only pays for a more expensive one when the cheaper tiers genuinely can't answer. A query enters at CAG: if the cache holds the answer, the response goes back to the user with no further work, because paying for a MAG lookup or a RAG round-trip on top of a cache hit would be pure waste. A CAG miss falls through to MAG, which checks session state and conversation history; a hit there returns a stateful answer built from what the session already knows. Only when both of the fast tiers miss, or what they found isn't sufficient, does the request fall through to RAG's full retrieval — embedding, database lookup, ranking, network transfer — which is slower by design because it's doing genuinely more work: reaching outside the system for knowledge neither the cache nor the session has.

The timeouts double as guards against a slow tier stalling a request that a faster tier could have handled adequately, not just performance targets: if the CAG lookup itself is slow, the cascade skips it rather than waiting; if MAG's state turns out to be complex enough that resolving it exceeds its 50ms budget, the cascade moves on to RAG rather than blocking there. The source also describes smarter fallback behavior than a strict miss-and-retry: a partial CAG match can be supplemented with a RAG call rather than discarded outright; MAG state that looks stale can be invalidated and refetched via RAG instead of trusted as-is; and if RAG itself is running slow, the cascade can return whatever CAG or MAG already has as a best-effort answer while the RAG result updates asynchronously in the background (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 5). These budgets have now been measured against a running cascade (`evaluation/reports/orchestration-meta-layer-cascade.md`). With the default timeouts and prototype routing, CAG answered within 10ms in all 45 attempts (p50 2.0ms), MAG within 50ms in all 10 (p50 7.5ms), and RAG within 2 seconds in all 40 (p50 5.3ms). That is a six-document corpus, a CPU-sized frozen cache, local Postgres and Qdrant, one request at a time, and `SearchDocuments` as the RAG retriever — a small-scale result, not a production one. Concurrent requests are unmeasured. Every RAG component that does CPU work of its own was measured in a PARALLEL route by #168 and #181: compression, bi-encoder and cross-encoder reranking, HyDE's passage embedding, BM25, and `CacheWarmedRetrieve`. On the event loop, all but BM25 over six chunks took the MAG tier past its budget on every attempt, and HyDE and `CacheWarmedRetrieve` took the CAG tier past its budget too. On worker threads, every tier stayed within budget (`evaluation/reports/orchestration-meta-layer-retrievers.md`). Two findings matter more than the headline numbers. First, embedding the query costs about 9ms before any tier runs, close to CAG's entire budget. Second, the budgets held only once no tier blocked the shared event loop. Before the RAG retriever's query embedding became a cache lookup (`CachingEmbeddingModel`), CAG finished within budget in just 29% of attempts in the first measurement run (recorded in the execution notes of `docs/superpowers/plans/2026-09-13-orchestration-meta-layer.md`), because in a parallel route the other tier's synchronous CPU work kept CAG's already-finished result from being delivered in time. `UnifiedAnswerQuestion` now also embeds the question on a worker thread, so its own embedding can't stall other requests sharing the loop.

## Three named synthesis techniques this system commits to

Beyond the orchestration components above, the source document names three specific ways the three paradigms are meant to actively work together, not just coexist. CLAUDE.md's current architecture rules use these ideas without naming them; naming them here makes them things this project can refer to directly.

### Tiered Knowledge Hot-Cold Architecture

Knowledge has a temperature, in the sense that how often something gets looked up should determine where it's stored. The source lays out three tiers: hot data — system prompts, static docs, code repositories, textbooks, FAQs — gets accessed on the order of a thousand times an hour and belongs in CAG, where lookups cost close to nothing because the content is already sitting in the frozen cache. Warm data — user preferences, session state, conversation history, an agent's scratchpad — gets accessed on the order of ten times an hour and belongs in MAG, in RAM or a fast database, at 1–10ms latency. Cold data — an enterprise knowledge base, live databases, real-time APIs, external web data, documents that are rarely touched — gets accessed roughly once a day or purely on demand and belongs in RAG's vector database or external index, at 50–500ms latency (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 2). A customer support bot illustrates the placement cleanly: its product manual, FAQ, and return policy are hot and sit in CAG; a user's ticket history and current issue state are warm and sit in MAG; the latest forum posts and real-time inventory are cold and stay in RAG.

Placement keeps adjusting after the fact, too. New data gets classified by its expected access pattern at ingestion and placed accordingly — hot into CAG preload, warm into MAG tables, cold into the RAG index — but the source specifies two ongoing corrections on top of that initial placement: promotion, where cold data that turns out to be accessed far more often than expected gets moved up to warm or hot, and demotion, where hot data that turns out to be accessed rarely gets moved back down to save the GPU VRAM it was occupying. Invalidation happens at whichever tier the changed data currently lives in. The underlying principle is to match the storage tier to the actual access pattern rather than to how important the data seems in the abstract — hot data belongs in GPU memory, warm in RAM, cold on disk or over the network, and getting that placement wrong in either direction wastes either latency or VRAM.

This tiering boundary is built and live-measured across all three pairings this project has a boundary for — the only one of the three named synthesis techniques that's true of. RAG+CAG's `TieringPolicy`/`WarmCache`, RAG+MAG's `MagTieringPolicy`, and CAG+MAG's `CagMagTieringPolicy` each implement the promote/demote decision for their own pairing. RAG+CAG's and CAG+MAG's were tested against a real `HFFrozenCache` — a genuine transformer KV cache, not a stub — with real measured time-to-first-token (TTFT) wins of roughly 3.9x for RAG documents and 2.6x for MAG-sourced content, and RAG+MAG's against a real `SemanticMemoryWarmStore` over Postgres and Qdrant (`evaluation/reports/rag-cag-synthesis.md`, `rag-mag-synthesis.md`, `cag-mag-synthesis.md`).

### State-Aware RAG

Standard RAG retrieves based only on the text of the current query, which throws away everything the system already knows about the user through MAG. The source's example makes the gap concrete: a user who's spent ten turns discussing Python data science, and whose MAG state records a preference for matplotlib, a dislike of Plotly, and habitual use of pandas, asks "How do I visualize this?" A retrieval pipeline with no access to that state has nothing to go on but four words and returns a generic visualization article that name-drops five different libraries. State-Aware RAG instead reads MAG's state first — the user's preferences, the conversation history, the current task context — and uses it to rewrite the query before retrieval ever runs: "How do I fix this?" becomes, for a user MAG knows is deploying FastAPI on Docker at an intermediate skill level, something closer to "how to fix FastAPI Docker deployment issues for an intermediate developer." Retrieval then runs against that enriched query, ranking gets a boost for results matching the user's known stack and skill level, and whatever comes back gets written back into MAG so the next turn has it too (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 6). For the visualization example, that pipeline retrieves matplotlib tutorials and pandas plotting documentation specifically, instead of the generic five-library overview standard RAG would have returned. The source's own framing for the relationship is direct: MAG makes RAG personal, and RAG in turn gives MAG something external to be personal about.

Built as `StateAwareRetrieve` (`src/orchestration/application/state_aware_retrieve.py`), live-tested against exactly this visualization scenario with a real, deliberately ambiguous three-document corpus (matplotlib, seaborn, plotly). The first implementation attempt exposed a real bug worth naming here rather than only in the evaluation report: fetching only `top_k` candidates before applying the state-derived ranking boost meant the boost never got a chance to promote a better-but-lower-ranked document at a realistic `top_k=1` — fixed by adopting this project's own established over-fetch-then-rerank shape (fetch more candidates, boost, then truncate), the same shape `RerankingRetriever` already used. `evaluation/reports/rag-mag-synthesis.md` has the full account, including a second real bug the same corpus exposed in the corpus design itself (a cross-reference between two library documents was giving one an undeserved keyword-overlap credit).

### Cache-Warmed RAG

RAG pays the full cost of embedding, database lookup, ranking, and network transfer on every single query, even though real query traffic tends not to spread evenly across a document set — the source's working assumption is that roughly 80% of queries hit the same 20% of documents, a Pareto pattern (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 7). Cache-Warmed RAG turns that skew into a shortcut: an analytics process tracks which documents RAG retrieves most often, the most frequently retrieved documents get pre-loaded into CAG's frozen KV cache, and at query time the system checks CAG first — an instant answer on a hit, a normal RAG fall-through on a miss. The cache gets periodically re-warmed as the analytics shift, and even on a cache miss the source notes that the pre-loaded documents can still improve RAG's own ranking by providing context the retriever wouldn't otherwise have. The worked example is a support bot where 80% of questions cluster around return policy, shipping, and account setup — those three documents get pre-loaded into CAG, and the result is that 80% of traffic gets answered in close to zero time while only the remaining 20% pays RAG's full retrieval cost. That 80/20 split is the source's own stated assumption about query distribution, not a number this project has measured against its own corpus — the technique only pays off to the degree the assumption actually holds, which is exactly what the analytics step in its mechanism exists to check.

Built as `CacheWarmedRetrieve` (`src/orchestration/application/cache_warmed_retrieve.py`), the first of this project's three cross-paradigm batches and the one the other two reused their own tiering/sync machinery from. Real bugs the review process caught before merge, worth naming here since they're the kind of thing this technique's own description above can't anticipate in the abstract: a cross-tenant data leak from the first draft's access tracker and frozen cache omitting `tenant_id` entirely, and `CacheWarmedRetrieve` serving stale content on a false-positive cache hit before a `content_hash` comparison against the cache's real state was added. `evaluation/reports/rag-cag-synthesis.md` has the full account and the real measured hit-rate and latency numbers.

## The sync mixer's tiebreak rule: RAG wins, except where CAG's own copy makes MAG the source of truth

The source names one explicit rule: when the three paradigms' different sync rhythms leave them disagreeing about the same underlying fact — RAG's index already updated, CAG's cache hasn't invalidated yet, MAG's session state still remembers the old value — RAG, as the external source, wins as the source of truth (`docs/inputs/concepts/unified_rag_cag_mag_architecture.md`, Concept 4). The source's worked example is a price drop from $100 to $80: RAG's index updates the moment the new document is ingested; the mixer detects the change and flags the corresponding CAG cache entry for invalidation; CAG actually invalidates on its next batch cycle, which the source's example puts at within five minutes — an illustrative figure from that example, not a fixed SLA this project has committed to; and MAG's session state, if it was holding the old price, gets corrected via notification once CAG's invalidation lands. The end state is that every paradigm converges on the same $80 answer, but only because there was an explicit, stated rule for which paradigm's data to trust while the others catch up — without it, the three would just keep giving three different answers to the same question for as long as their sync rhythms stayed out of step.

That rule covers two of this project's three real pairings and was implemented exactly as stated for both: RAG wins in the RAG+CAG batch's `SyncCycle` and the RAG+MAG batch's `MagSyncCycle`, both real, both measured against real infrastructure (`evaluation/reports/rag-cag-synthesis.md`, `evaluation/reports/rag-mag-synthesis.md`). The source names no rule at all for the third pairing, CAG-vs-MAG with no RAG involved — and building it surfaced a case the source's own worked example doesn't cover, since a CAG-vs-MAG conflict can only arise here because CAG's own hot-tier entry is a promoted copy of a MAG record in the first place, not an independent paradigm holding a competing fact the way RAG's index is. The CAG+MAG batch's own investigation (`docs/superpowers/specs/2026-09-02-cross-paradigm-cag-mag-design.md`) reasoned through this directly and reached the opposite rule for this one case: MAG wins, because MAG is the paradigm that actually owns the data being cached, and CAG's copy is derived, not authoritative, by construction — the same logic that makes RAG authoritative in the other two pairings (the external, non-derived source wins) points the other way once the "external source" in the conflict is MAG instead of RAG. `CagMagSyncCycle` implements exactly this, reusing the identical `sync_mixer.reconcile()` function the other two pairings use — the function itself doesn't know or enforce which side wins, only comparing a content hash, which is what lets one function serve all three pairings' opposite tiebreak rules without any of them needing their own copy of it. Real measurement against real infrastructure confirmed a genuine CAG-vs-MAG-only conflict does arise from this project's own tiering mechanism, and found it detected and resolved in a real, measured 48.8ms (`evaluation/reports/cag-mag-synthesis.md`).

## The full technology stack

The stack below is what the original project brief's §3.1–§3.8 already committed to, carried over here with one correction: React is **18+** throughout. The original project brief's header for its frontend section named React major version 19, while its own stack table in that very same section already listed "React 18+" — and neither source document specified version 19 anywhere else. That was an internal contradiction in the original file, not a deliberate version choice, and it's corrected here to React 18+ consistently, since this is where that table now lives. Five of the choices below already have their reasoning recorded separately as ADRs — vLLM over SGLang, Qdrant over Milvus/Weaviate, standard attention over alternative attention, Hexagonal+CQRS for MAG, and RAG orchestration, where ADR-0006's hand-rolled Python classes supersede ADR-0005's LangChain/LangGraph choice — in `docs/decisions/adr/`; this section is the inventory, those files are the "why."

`fullstack_unified_ai_system.md` frames PyTorch's role in one useful metaphor worth carrying forward: PyTorch is the engine — tensor operations, model weights, custom CUDA kernels — a serving framework like vLLM is the car built on top of that engine, and this project's own code is the driver. PyTorch alone doesn't give production-grade batching, PagedAttention, prefix caching at serving scale, or tensor parallelism across GPUs; vLLM does, because vLLM is built on PyTorch specifically to add those capabilities. The stack below reflects that split: PyTorch is used directly only for custom KV-cache research and fine-tuning, and vLLM is the actual serving layer everything else talks to.

### Core backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.11+ | Main backend |
| DL Framework | PyTorch | 2.3+ | Tensor ops, custom CUDA |
| LLM Serving | vLLM | 0.5+ | Production inference |
| API Framework | FastAPI | 0.111+ | REST + WebSocket |
| Validation | Pydantic v2 | — | Data models |
| Async Runtime | uvloop | — | Fast async event loop |
| Type Checking | mypy | latest | Static types |
| Lint/Format | ruff | latest | Linting + formatting |
| Package Manager | uv / poetry | latest | Dependencies |

### Served models and hardware target

The stack table above and the CAG ecosystem table below both assume NVIDIA/CUDA, because that's what `fullstack_unified_ai_system.md` assumes throughout — Triton, `flash-attn`, and the CUDA toolkit version are all NVIDIA-specific. This project's actual initial deployment target is different: **vLLM on ROCm**, running on an AMD Radeon 7900 XTX (24GB VRAM), because that's the hardware actually available for this project's own experimentation. This is stated here as an explicit reconciliation, the same way the React 18+ correction earlier in this section and `docs/governance/GIT_WORKFLOW.md`'s merge-commit ruling are — vLLM does have official ROCm support, so the serving-engine choice (ADR-0001) doesn't change, but the CUDA-specific kernel entries in the CAG ecosystem table below (`triton`, `flash-attn`) need ROCm equivalents that haven't been selected or benchmarked yet, which is flagged here as design intent, not yet verified, rather than left as a silent assumption that CUDA is available.

Two models are meant to be served locally against this hardware, and three more are meant to be called through their APIs as reference points rather than self-hosted — the full reasoning for why each one is in which category, and the citations behind every figure below, live in `docs/evaluation/COMPARISON_METHODOLOGY.md`; this table is the inventory, that document is the "why." The vLLM-on-ROCm serving stack itself is confirmed set up and working (ROCm 7.2.0, vLLM 0.28.0, verified with a real generation and zero GEMM errors). Five CAG measurements have run against it, all serving `Qwen/Qwen2.5-0.5B-Instruct` rather than either self-hosted model below: Prefix Caching, PagedAttention, Cache-Aware Batching, Multi-Turn Caching, and the Combinations batch's High-Throughput Serving archetype (`tests/integration/test_cag_*_against_real_vllm.py`, each reported under `evaluation/reports/cag-*.md`). Every other measurement under `evaluation/reports/` used CPU-feasible substitutes instead: Ollama, real small Hugging Face models where a technique's own correctness genuinely depends on a real forward pass (`distilgpt2` throughout CAG's KV-cache-compression and speculative-decoding batches), and `sentence-transformers` for embeddings. `docs/architecture/CAG.md` says, technique by technique, which kind of measurement each CAG technique got.

| Model | Role | Context | Notes |
|---|---|---|---|
| Gemma 4 (12B / 26B-A4B / 31B) | Self-hosted, vLLM on ROCm | 256K native | Apache 2.0; chosen first, ahead of the context comparison |
| Qwen3.8-27B | Self-hosted, vLLM on ROCm | 262,144 native, extensible to 1M (YaRN) | Apache 2.0; single-GPU by design; stronger of the two on coding benchmarks and native context |
| DeepSeek V4 (V4-Pro / V4-Flash) | API reference | 1M, 384K max output | MIT-licensed weights, but not realistically self-hostable on one 24GB card |
| Claude (Sonnet 5 / Opus 5) | API reference, primary qualitative judge | 1M | Not one of the models being ablated — see `docs/evaluation/qualitative-rubric.md` |
| Gemini (3.1 Pro / 3.6 Flash) | API reference, secondary qualitative judge | ~1.05M | Independent model family from both the self-hosted models and the primary judge |

### Data storage

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vector DB | Qdrant | Semantic search, HNSW indexing |
| Relational DB | PostgreSQL 16 + pgvector | Structured data + vectors |
| Cache / Session | Redis 7+ | Hot cache, pub/sub, sessions |
| Graph DB | Neo4j | Memory relationships, MAG graphs |
| Object Storage | MinIO | Document storage |
| Message Queue | RabbitMQ | Event-driven sync |
| Task Queue | Celery | Background jobs |
| Document Store | MongoDB (optional) | Unstructured logs |

### RAG ecosystem

| Component | Library | Purpose |
|-----------|---------|---------|
| Orchestration | LangChain >=0.2.0 / LlamaIndex >=0.10.0 | RAG pipelines |
| Agent Graphs | LangGraph >=0.1.0 | CRAG, Self-RAG workflows |
| Document Parsing | unstructured >=0.14.0 | PDF, MD, TXT parsing |
| Embeddings | sentence-transformers >=3.0 | Vector generation |
| Keyword Search | rank-bm25 >=0.2.2 | BM25 keyword search |
| Reranking | BAAI/bge-reranker-v2-m3 | Cross-encoder reranking |

### CAG ecosystem

| Component | Library | Purpose |
|-----------|---------|---------|
| Serving Engine | vLLM >=0.5.0 | PagedAttention, prefix caching |
| Attention Kernels | flash-attn >=2.5.0 | Optimized attention |
| Quantization | bitsandbytes, auto-gptq, optimum | Model compression |
| Custom CUDA | triton >=2.3.0 | Custom kernels |

### MAG ecosystem

| Component | Library | Purpose |
|-----------|---------|---------|
| Memory Framework | mem0ai >=1.0.0 | Memory layer for LLMs |
| PostgreSQL Client | psycopg >=3.1.0 | Structured storage |
| Vector Extension | pgvector >=0.2.0 | Postgres vectors |
| Graph DB | neo4j >=5.20.0 | Relationship memory |
| Async Redis | redis-py >=5.0.0 / aioredis >=2.0.0 | Hot cache |

### Observability

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM Tracing | Langfuse | Trace LLM calls |
| Metrics | Prometheus | System metrics |
| Dashboards | Grafana | Visualization |
| Logging | structlog | Structured logging |
| APM | OpenTelemetry | Distributed tracing |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containers | Docker | Containerization |
| Orchestration | Kubernetes | Container orchestration |
| Packaging | Helm | K8s package management |
| Ingress | Traefik | Load balancing |
| TLS | cert-manager | TLS automation |
| Secrets | HashiCorp Vault | Secret management |
| IaC | Terraform | Infrastructure as code |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React 18+ | Web UI |
| Styling | Tailwind CSS | Utility-first CSS |
| State | Zustand | State management |
| Streaming | SSE / WebSocket | Token streaming |

## Hardware and cost: what this stack actually requires to run

A technology choice is only meaningful once you know what it costs to actually run, which is exactly what this section adds on top of the stack tables above — none of it is in CLAUDE.md today. The figures below are `fullstack_unified_ai_system.md`'s own estimates (§7); nothing here has been benchmarked against a real deployment of this project, since no deployment exists yet.

A single development machine needs, at minimum, an RTX 4090 with 24GB of VRAM, 32GB of system RAM, a 500GB SSD, and 8 CPU cores, with 64GB of RAM and an RTX 4090 or A6000 (48GB) recommended if the budget allows it. At that scale, the model that actually fits is Llama-3-8B-Instruct quantized to AWQ 4-bit, which the source notes fits in as little as 8GB of VRAM — comfortably inside even the minimum GPU spec. Production hardware looks different by node role: an API node needs no GPU at all — 32GB RAM, a 500GB SSD, 8 CPU cores, and 10Gbps networking cover it — while a GPU node running vLLM needs two A100 80GB GPUs or four L40S GPUs, 128GB RAM, 2TB of NVMe storage, 32 CPU cores, and 25Gbps networking with NVLink if multiple GPUs are involved. At that scale the model changes too: Llama-3-70B-Instruct, tensor-parallel across two to four GPUs, replaces the 8B model used for development.

Cloud estimates from the same source (AWS, monthly, approximate): a dev environment on a `g5.xlarge` (one A10G GPU) runs around $500/month; staging on a `g5.12xlarge` (four A10G GPUs) runs around $4,500/month; and a production deployment on a `p4d.24xlarge` (eight A100 GPUs) runs around $30,000/month. The jump between those three numbers is almost entirely GPU count and class, not anything else in the stack — which is consistent with the hardware tables above, where the API-serving side of this architecture stays cheap and commodity while the GPU-serving side is where the real cost lives.

## When you might deviate from this stack

The tables above are the default, not the only option this project's source material considers. `fullstack_unified_ai_system.md` §8 lays out three specific directions a deviation might go, each for a different reason.

**If you'd rather build in Go or Rust:** FastAPI's rough equivalents are Gin or Echo in Go and Axum or Actix in Rust; Celery's nearest Go equivalent is Asynq, with Rust offering no direct match beyond building on `tokio` directly (the source spells this out explicitly, listing Celery's Rust alternative as "None (use tokio)"); Qdrant has an official client in both languages. PyTorch does not follow that same both-languages pattern: its Go column in the source table is empty, and reading that against the table's own convention — an explicit "None" where no equivalent truly exists — an empty cell means no Go binding exists for PyTorch at all, not merely that one wasn't listed. Rust fares only a little better, with a binding (`tch-rs`) the source itself calls limited. The source's own verdict is blunt about when this is worth it: Python is strongly recommended for this project because of its ML ecosystem, and Go or Rust make sense only for specific, narrowly-scoped high-performance microservices carved out of the larger system — not as a wholesale replacement for the Python stack above.

**If you'd rather not self-host:** every self-hosted component in the stack tables has a managed equivalent — Qdrant can become Pinecone or Weaviate Cloud, PostgreSQL can become AWS RDS or Cloud SQL, Redis can become AWS ElastiCache or Redis Cloud, Neo4j can become Neo4j Aura, vLLM's serving role can move to Together AI, Fireworks, or Groq, Kubernetes can become AWS EKS or GCP GKE, and MinIO can become AWS S3 or GCS. This trades operational burden for a recurring cost and less control over the underlying infrastructure — the right call when a team would rather not run its own database and cache fleet, and the wrong call when the cost or data-locality requirements make self-hosting worth the operational overhead.

**If you want something simpler than the full stack:** LangChain can be replaced by LiteLLM plus custom orchestration code, vLLM can be replaced by Ollama for local-only serving, the Qdrant-plus-BM25 hybrid-search pair can collapse into Chroma as an all-in-one store, Neo4j can be replaced by NetworkX running in memory, Celery-plus-RabbitMQ can be replaced by APScheduler plus Redis, and Kubernetes can be replaced by Docker Compose plus systemd. Every substitution in this list trades production capability — scale, durability, operational maturity — for less setup and faster iteration, which makes this the right direction for early experimentation or a proof of concept, not for the production-grade system this stack is otherwise built for.

## The Phase 1 module blueprint

`fullstack_unified_ai_system.md` §4 lays out a concrete `src/` module structure for this system, organized around the same three paradigms plus the orchestration layer that coordinates them. The Auth Foundation sub-project (August 2026) established the authentication infrastructure alongside these paradigm modules: `src/api/` and `src/identity/` exist, along with `tests/unit/`, `tests/integration/`, and the Docker configuration needed to run them locally (`docker/Dockerfile.api` and `docker/docker-compose.yml`). The RAG Pipeline sub-project that followed it has since built the first of the three paradigm modules: `src/rag/` now exists, with its domain, application, and infrastructure layers in place, and `src/api/routers/` has grown a `documents` router and a `chat` router alongside the `auth` router Auth Foundation added — the endpoints a client actually calls to upload a document and get a grounded answer back. The MAG Memory sub-project has since built the second: `src/mag/` now exists, with its own domain, application, and infrastructure layers in place across seven completed batches (Foundation through Combinations) covering episodic, semantic, and procedural memory, consolidation, memory evolution, memory gating, and memory graphs — see `docs/architecture/MAG.md` for the concepts and `evaluation/reports/mag-*.md` for each batch's own live-measured narrative report. `src/cag/` began with three CPU-feasible batches: a domain-layer module (`src/cag/domain/attention_compatibility.py`) encoding ADR-0003's alternative-attention compatibility split; a second batch adding the domain spine (`ports.py`'s two compressor interfaces, `entities.py`'s `CompressedKV`, `compression_metrics.py`) plus five real KV Cache Compression algorithms under `src/cag/infrastructure/` (KIVI, KVQuant, PALU, MiniCache, ShadowKV); and a third batch adding `src/cag/application/` (`SpeculativeDecode`, the propose-verify-accept loop) plus three Speculative Decoding candidate generators (Medusa, Lookahead Decoding, Prompt Lookup Decoding) and the `HFTargetModel` and `domain/ngram_search.py` code they share — see `docs/architecture/CAG.md` for the concepts and `evaluation/reports/cag-*.md` for each batch's own measured report. Six more techniques followed once that serving stack was running on this project's own AMD 7900 XTX — Prefix Caching, PagedAttention, Cache-Aware Batching, and Multi-Turn Caching validated against live vLLM, KV Cache Eviction measured on real `distilgpt2` tensors, and Hybrid Offloading run in simulation — and the Combinations batch closed out CAG's backlog with `src/cag/application/`'s `LongContextPipeline` and `RealTimeChatPipeline` plus a live-vLLM High-Throughput Serving measurement (`evaluation/reports/cag-combinations.md`). `src/orchestration/` has its first real content: a cross-paradigm RAG+CAG synthesis batch built `domain/` (`entities.py`'s `CacheHit`/`SyncConflict`/`TierDecision`, `ports.py`'s `AccessFrequencyTracker`/`FrozenCache`, `sync_mixer.py`'s RAG-wins reconciliation), `application/` (`WarmCache`, `TieringPolicy`, `SyncCycle`, `CacheWarmedRetrieve`), and `infrastructure/` (`InMemoryAccessFrequencyTracker`, `HFFrozenCache` — a real, CPU-sized proxy for CAG's GPU-resident cache using the same real-`distilgpt2` methodology CAG Batch B's own integration test proved out), implementing Cache-Warmed RAG, the CAG↔RAG hot/cold tiering boundary, and the Sync Mixer's RAG-vs-CAG tiebreak — see `evaluation/reports/rag-cag-synthesis.md` for the live-measured report. A second cross-paradigm batch, RAG+MAG synthesis, has since built `domain/`'s `WarmEntry` entity and `UserScopedAccessFrequencyTracker`/`WarmStore` ports (deliberately separate from the RAG+CAG ports above, since MAG's warm tier is inherently personal — every MAG table is `user_id`-keyed — unlike CAG's tenant-wide shared cache), generalized `sync_mixer.reconcile` to take a plain content hash so both pairings' sync mechanisms share it, and added `application/`'s `StateAwareRetrieve`, `MagTieringPolicy`, and `MagSyncCycle` plus `infrastructure/`'s `InMemoryUserScopedAccessFrequencyTracker` and `SemanticMemoryWarmStore` (wrapping this project's existing `PostgresSemanticMemoryRepository`/`QdrantSemanticMemoryIndex`), implementing State-Aware RAG, the RAG↔MAG warm/cold tiering boundary, and the Sync Mixer's RAG-vs-MAG tiebreak — see `evaluation/reports/rag-mag-synthesis.md` for the live-measured report. A third and final cross-paradigm batch, CAG+MAG synthesis, completes the cross-paradigm backlog: `domain/`'s `cag_mag_keys.py` (the `cache_key`/`tracker_key` namespacing functions that let this batch reuse Batch D's tenant-only `FrozenCache` and Batch E's `UserScopedAccessFrequencyTracker` unmodified for inherently-personal MAG content, rather than adding a fourth near-duplicate port) and `application/`'s `CagMagTieringPolicy`/`CagMagSyncCycle` (reusing `sync_mixer.reconcile` a third time, with MAG — not RAG — playing the authoritative role, per this batch's own investigation into whether a genuine CAG-vs-MAG-only conflict can arise), implementing the CAG↔MAG hot/warm tiering boundary and its Sync Mixer tiebreak — see `evaluation/reports/cag-mag-synthesis.md` for the live-measured report. A per-query meta-layer batch then built three of the four remaining coordinators, under this project's hexagonal layering rather than the blueprint's flat `router.py`, `budget_allocator.py`, and `latency_cascade.py`. The Paradigm Router is `domain/paradigm_router.py` plus three `QueryClassifier` implementations under `infrastructure/`. The Context Budget Allocator is `domain/budget_allocator.py`, `application/assemble_context.py`, and `infrastructure/postgres_session_budget_recorder.py`. The Latency-Adaptive Fallback Cascade is `application/latency_cascade.py` with `application/cascade_tiers.py`. All three are composed as `application/unified_answer_question.py` — see `evaluation/reports/orchestration-meta-layer.md` for the live-measured report. A second meta-layer batch then built the fifth coordinator, the Freshness-Aware Data Router. It is `domain/freshness_router.py` plus three use cases (`application/ingest_data_source.py`, `application/refresh_cached_sources.py`, and `application/review_source_freshness.py`), with `ExpiringFrozenCache`, `PostgresDataSourceRepository`, `ChunkedRagIndex`, and `RecordSemanticFactWriter` under `infrastructure/`. Its live-measured report is `evaluation/reports/freshness-router.md`. All five components of the orchestration meta-layer now exist, and every named cross-paradigm synthesis technique and every CPU-feasible tiering/sync boundary between the three paradigms is now implemented and live-measured. `src/workers/`, the Kubernetes configuration under `k8s/`, the additional Docker images (`Dockerfile.worker` and `Dockerfile.vllm`) and production Docker Compose variant (`docker-compose.prod.yml`), and the experiment notebooks under `notebooks/` remain pending as well. The blueprint below documents the full target structure; what follows is both a record of what's been built so far and a guide for what remains to be built.

```text
src/
├── rag/
│   ├── chunking/       # base.py, fixed_size.py, semantic.py, recursive.py, parent_document.py
│   ├── embedding/      # base.py, sentence_transformers.py, openai.py
│   ├── retrieval/      # base.py, vector.py, hybrid.py, multi_query.py, hyde.py
│   ├── reranking/      # base.py, cross_encoder.py, llm_reranker.py
│   ├── advanced/       # self_rag.py, crag.py, context_compression.py, parent_document.py
│   ├── pipeline.py     # Main RAG pipeline orchestrator
│   └── indexer.py      # Document indexing service
│
├── cag/
│   ├── cache/          # base.py, prefix_cache.py, kv_cache.py, block_manager.py
│   ├── eviction/       # base.py, h2o.py, snapkv.py, random.py
│   ├── compression/    # base.py, kivi.py, kvquant.py, low_rank.py
│   ├── serving/        # vllm_client.py, speculative.py, batching.py
│   ├── offloading/     # base.py, cpu_offload.py, disk_offload.py
│   ├── preprocessor.py # Pre-load documents into cache
│   └── warmup.py       # Cache warming from analytics
│
├── mag/
│   ├── memory/         # base.py, episodic.py, semantic.py, procedural.py, working.py
│   ├── storage/        # redis_store.py, postgres_store.py, neo4j_store.py, qdrant_store.py
│   ├── retrieval/      # base.py, semantic.py, temporal.py, graph.py, multi_strategy.py
│   ├── consolidation/  # base.py, llm_reflection.py, pattern_extraction.py
│   ├── evolution/      # base.py, contradiction.py, update_policy.py
│   ├── gating/         # base.py, token_budget.py, relevance.py
│   ├── agent_loop.py       # Main MAG agent loop
│   └── memory_manager.py   # Central memory coordinator
│
├── orchestration/
│   ├── router.py            # Paradigm router (query classifier)
│   ├── budget_allocator.py  # Context budget allocator
│   ├── sync_mixer.py        # Synchronization mixer
│   ├── latency_cascade.py   # Latency-adaptive fallback
│   ├── freshness_router.py  # Freshness-aware data routing
│   └── unified_context.py   # Assembles unified context
│
└── workers/
    ├── celery_app.py             # Celery app factory
    ├── consolidation_worker.py   # MAG consolidation tasks
    ├── indexing_worker.py        # RAG indexing tasks
    ├── cache_warmup_worker.py    # CAG cache warming
    └── sync_worker.py            # Cross-paradigm sync

tests/
├── unit/          # test_rag_chunking.py, test_rag_retrieval.py, test_mag_memory.py, test_cag_cache.py, test_orchestration.py
├── integration/   # test_api_endpoints.py, test_rag_pipeline.py, test_mag_agent.py, test_full_stack.py
├── fixtures/      # sample_documents/
└── conftest.py

docker/
├── Dockerfile.api
├── Dockerfile.worker
├── Dockerfile.vllm
├── docker-compose.yml
└── docker-compose.prod.yml

k8s/
└── helm/
    └── unified-ai/
        ├── Chart.yaml
        ├── values.yaml
        └── templates/  # api-deployment.yaml, vllm-deployment.yaml, redis-statefulset.yaml,
                         # postgres-statefulset.yaml, qdrant-statefulset.yaml, ingress.yaml

notebooks/
├── 01_rag_experiments.ipynb
├── 02_cag_benchmarks.ipynb
├── 03_mag_prototypes.ipynb
└── 04_unified_demo.ipynb
```

The Auth Foundation sub-project added `src/api/` and `src/identity/` during Phase 0 foundation work (`docs/superpowers/specs/2026-08-22-auth-foundation-design.md`), and they sit outside the tree diagram above because authentication isn't one of the three RAG/CAG/MAG paradigms the blueprint is organized around — they needed their own home instead of a place inside the diagram's paradigm-centric structure. They provide the authentication and API infrastructure that all three paradigms depend on, rather than implementing one paradigm themselves.

Every module directory in the paradigm and worker layers maps directly onto something already named earlier in this document: `orchestration/` holds one file per meta-layer component described above, `rag/`, `cag/`, and `mag/` each hold a subdirectory per pipeline stage described in the paradigm's own architecture doc, and `workers/` holds the background jobs — consolidation, indexing, cache warmup, cross-paradigm sync — that the orchestration and memory sections above assume are running asynchronously. This structure is the target for Phase 1 specifically: the source's own implementation roadmap (`fullstack_unified_ai_system.md` §5) scopes Phase 1 to "Working API with basic RAG" over its first two weeks — a FastAPI scaffold running under Docker, PostgreSQL, Qdrant, and Redis wired up via `docker-compose`, a document upload endpoint, fixed-size chunking with MiniLM embeddings, a vector search endpoint, and a basic chat endpoint answering from RAG context. That's a small slice of the full tree above — mostly `rag/`, the earliest pieces of `orchestration/`, and the `docker/` files needed to run it locally — and it's long since been delivered and then substantially exceeded. `src/rag/infrastructure/` carries a fixed-size chunker, a `sentence-transformers` embedder, a Qdrant-backed vector store, and a Claude-backed chat model, and `src/rag/application/` wires those into the upload, search, and answer-question use cases the `documents` and `chat` routers expose — the upload endpoint, fixed-size chunking with MiniLM embeddings, vector search, and basic RAG-grounded chat that the roadmap called for, plus the five further chunking strategies, hybrid search and reranking, parent-document retrieval and context compression, multi-query/HyDE/Self-RAG, and CRAG that the roadmap scoped for later phases. CAG and MAG are both well past "haven't started": as this section's own opening paragraphs describe in full, MAG has seven completed batches covering its entire memory hierarchy, CAG has all nine of its techniques built, eight measured (Alternative Attention carries tested code rather than a measurement), plus its three Combinations pipelines built and measured too, and `src/orchestration/` has real domain, application, and infrastructure code across all three cross-paradigm synthesis batches. The unified per-query path is now served over HTTP through the `sessions` router. What genuinely remains unbuilt, stated plainly rather than left to this paragraph's own out-of-date framing: an HTTP endpoint for freshness-routed ingestion (it accepts client-supplied scope, and deciding who may write tenant-wide sources needs a role model this project doesn't have yet), a scheduler to drive the router's refresh and review, and everything past Phase 1 — the frontend, `src/workers/`, and the Kubernetes/production-Docker layer.
