# Orchestration Meta-Layer Batch A: Paradigm Router, Context Budget Allocator, Latency-Adaptive Fallback Cascade — Live Measurement Report

**Scope:** Epic #150; Stories #151 (Paradigm Router), #152 (Context Budget Allocator), #153 (Latency-Adaptive Fallback Cascade), and #154 (combined capability validation); Tasks #156–#167. The design is `docs/superpowers/specs/2026-09-13-orchestration-meta-layer-design.md`, and the plan, with every place execution changed it, is `docs/superpowers/plans/2026-09-13-orchestration-meta-layer.md`. The full tables behind this report are in four generated files: `orchestration-meta-layer-router.md`, `orchestration-meta-layer-cascade.md`, `orchestration-meta-layer-comparison.md`, and `orchestration-meta-layer-comparison-oracle.md`. The Freshness-Aware Data Router (#155) is the next batch.

## What this batch is

The three cross-paradigm batches built every pairwise mechanism the meta-layer coordinates. This batch builds the three per-query decisions above them, and composes them the way the concept source's Pattern 1, "The Smart Router", does:

- **The Paradigm Router** decides which paradigms to ask. `decide` in `src/orchestration/domain/paradigm_router.py` turns independent per-paradigm scores from a `QueryClassifier` into a route. A score near the threshold widens the route to a parallel run. A classification with no signal, or a classifier that fails or times out, runs every paradigm in parallel rather than guessing one.
- **The Latency-Adaptive Fallback Cascade** decides in what order, and for how long. `LatencyCascade` in `src/orchestration/application/latency_cascade.py` runs unrouted (exactly as Concept 5 draws it), routed, or in parallel, under 10ms / 50ms / 2s tier timeouts, with best-effort answers and bounded background RAG completion.
- **The Context Budget Allocator** decides how much of the window each answer may occupy. `allocate` in `src/orchestration/domain/budget_allocator.py` splits the window 40/25/20/10/5 and donates idle slices to the paradigms that returned content, and each turn's allocation is recorded in `sessions.context_budget`.

`UnifiedAnswerQuestion` composes the three. Existing paradigm code is reached through adapters rather than changed: MAG's semantic search sits behind `SessionScopedSemanticFactSearch`, and any RAG retriever behind `RagTier`. Two things outside the new meta-layer files did change. `CacheWarmedRetrieve` gained a public, tenant-indexed match, so the CAG tier can ask whether a warmed document answers without triggering retrieval. And `CachingEmbeddingModel` is new in `src/rag/infrastructure/`, for the reason given under the second claim.

## Claim 1: the router prevents answers the cascade alone would get wrong

The ablation ran six freshness phrasings through one cascade, with a frozen cache holding the superseded thirty-day return policy and Qdrant holding the current forty-five-day one (`orchestration-meta-layer-cascade.md`):

| Arm | Stale text only | Current text only | Both |
|---|---|---|---|
| Router off (Concept 5 as drawn) | 4 | 1 | 1 |
| Router on, lexical | 0 | 3 | 3 |
| Router on, MiniLM prototype | 0 | 1 | 5 |

Without the router, four of six contexts held only the superseded policy. That is the failure the spec predicted: CAG reports a confident hit, and the cascade stops there, as Concept 5 says a hit should. Routing removed every stale-only context. It did not keep the stale text out, though. Whenever a route still included CAG, the superseded policy arrived next to the current one — in 3 of 6 lexically routed runs and 5 of 6 prototype-routed runs.

Whether a mixed context then produces a stale answer is a generation question, and in the comparisons below it didn't. Both return-policy questions ran prototype-routed with CAG and RAG both hitting, so both versions reached the model, and it answered forty-five days in all ten runs. That is two questions, one model, and a current document that states its own change date, so it shows mixed contexts can work here, not that they are safe in general. What routing cannot do at all is protect a static question whose cache has gone stale — question 1 under the oracle arm, below, is exactly that case.

## Claim 2: the tier budgets are achievable on this stack

Measured with prototype routing and the default timeouts, 10 questions × 5 repeats after one untimed warm-up:

| Tier | Attempts | Outcomes | p50 | p95 | Budget | Within budget |
|---|---|---|---|---|---|---|
| CAG | 45 | 30 hit, 15 miss | 2.00ms | 2.58ms | 10ms | 100% |
| MAG | 10 | 10 miss | 7.11ms | 8.27ms | 50ms | 100% |
| RAG | 40 | 40 hit | 5.24ms | 6.72ms | 2,000ms | 100% |

Every tier met its budget on every attempt, with three qualifications. First, query embedding (MiniLM on CPU) costs p50 9.12ms, max 12.40ms, before any tier runs. Every tier depends on that embedding, so the medians put the whole CAG path at about 11ms — just over the 10ms the source budgets for checking the cache — even though the lookup itself takes 2ms. Second, every MAG attempt in this run was a miss. A miss still runs the full similarity query, but a MAG hit's latency was never measured under the budget. Third, this is a six-document corpus with loopback Postgres and Qdrant on one machine.

The number that matters most is from the first run of this measurement, not the final one. That run put CAG within budget in only 29% of attempts (p50 10.76ms, 32 of 45 timed out), yet a CAG attempt measured alone takes p50 0.40ms. Two hypotheses were probed against the real cascade and MiniLM:

- *Rejected:* contention left over from embedding the query just before the cascade. A CAG attempt right after one still took p50 0.68ms.
- *Confirmed:* in a parallel route, `SearchDocuments` re-embeds the question synchronously on the event loop. While that CPU work holds the loop, the CAG tier's finished worker thread can't deliver its result before `wait_for` times it out. With the embedding on the loop, 15 of 60 CAG attempts timed out (p50 9.77ms); with it on a worker thread, none did (p50 0.86ms).

Concept 5's timeouts are per tier, but asyncio tiers share one thread, so a tier's budget holds only as long as no sibling blocks the loop. The fix shares one `CachingEmbeddingModel` between the use case and the retriever, so the retriever's embed becomes a lookup, and `LatencyCascade` documents that a tier must never block the loop. An integration test pins that the retriever's embed is a cache hit with the real model: a normalization step on either side would bring the starvation back without failing anything else, because a cache miss still returns a correct embedding.

## Claim 3: classification must not cost more than it saves

Three classifiers were measured on 46 held-out routing queries: 40 generated by qwen3.5 after the classifiers were committed (4 labels corrected by hand review, each correction recorded in the file), plus Concept 1's six example queries verbatim (`orchestration-meta-layer-router.md`):

| Classifier | Exact route | Covers needed paradigms | Confident | MAG recall | p50 | p95 |
|---|---|---|---|---|---|---|
| Lexical cues | 59% | 72% | 89% | 58% | 0.11ms | 0.18ms |
| MiniLM prototype (k=5) | 52% | 83% | 37% | 63% | 2.37ms | 2.76ms |
| qwen3.5 | 74% | 85% | 96% | 95% | 18,924ms | 59,965ms |

The trade-off is real, and on this machine it settles itself. The LLM classifier is the most accurate by a wide margin, but its median decision takes 18.9 seconds, about 1,900 times CAG's entire budget. `UnifiedAnswerQuestion` classifies under a 2-second timeout, so here the LLM classifier always falls back to running every tier in parallel. That behavior is deliberate and documented at the default.

The two millisecond-scale classifiers do cost less than they save: both are small next to the query embedding every tier needs anyway (10.97ms p50 in that run). Between them the choice is which error costs more. The lexical classifier is usually confident but misses needed paradigms more often, with MAG the worst at 58% recall. The prototype classifier covers more, but is confident on only 37% of queries, so most of its routes run every tier. The comparison below measures what the prototype classifier's misses cost end to end, and the answer is that MAG recall is the number that decides it.

The route descriptions, the label review, and all three classifiers share one author, so these numbers are in-distribution estimates, not a measurement of real traffic.

## The allocator's effect

The same ten prototype-routed questions were assembled at four window sizes, once with dynamic reallocation and once with static base slices:

| Window | Items dropped (dynamic) | Items dropped (static) | Tokens kept (dynamic) | Tokens kept (static) |
|---|---|---|---|---|
| 1,000 | 0 | 0 | 616 | 616 |
| 400 | 0 | 0 | 616 | 616 |
| 250 | 0 | 8 | 616 | 461 |
| 150 | 6 | 14 | 500 | 306 |

Reallocation matters only once static slices bind. At 250 tokens it kept all 616 tokens of context where static slices dropped 8 items, and at 150 tokens it kept 500 against 306. The first version of this sweep used only a 1,000-token window, where neither variant dropped anything — a null result that measured nothing — so the smaller windows were added.

The spec allows a quality claim for the allocator only if the judge scores show one. The comparisons below ran at the 128K window OVERVIEW.md budgets for, where this corpus never comes close to filling any slice, so the allocator cannot have affected those scores, and no quality effect is claimed.

## The unified pipeline against RAG-only, and what routing error costs

Both comparisons run `RunComparison` self-versus-self on qwen3.5: ten questions, five repeats each, the same judge, and the same deterministic success checks. The baseline is RAG-only `AnswerQuestion` (top 3). The treatment is `UnifiedAnswerQuestion` under generous tier timeouts, routed either by the MiniLM prototype classifier or by each question's labeled route in `queries.yaml` — an oracle that shows what the pipeline adds once routing error is taken out.

| Treatment routed by | Task success (RAG-only → unified) | Input tokens | p50 latency |
|---|---|---|---|
| MiniLM prototype classifier (`orchestration-meta-layer-comparison.md`) | 78% → 76% | 1,235 → 1,338 | 8,366ms → 8,515ms |
| Labeled routes, an oracle (`orchestration-meta-layer-comparison-oracle.md`) | 80% → 90% | 1,235 → 1,054 | 7,720ms → 6,692ms |

The two baselines run the identical RAG-only pipeline and still differ by two points, which is the run-to-run noise these task-success numbers carry.

**Routed by the prototype classifier, the unified pipeline was no better than RAG-only.** The gap, 76% against 78%, is inside that noise. The route log shows why. The classifier never sent the three personal questions — "What shipping speed did I say I prefer?", "What shoe size do I wear?", and "Are the running shoes in my size in stock right now?" — to MAG. It routed them to CAG and RAG, neither of which holds the user's facts, so the model said the context didn't contain the answer, exactly as RAG-only did. RAG-only already answered every other question, which left routing nothing to add. This is the router comparison's 63% MAG recall, seen end to end.

**Routed correctly, the pipeline added exactly what MAG knows, and lost one question to a stale cache.** With labeled routes, MAG hit on all three personal questions, and all three were answered in all five repeats, against 0%, 20%, and 80% for RAG-only. The treatment also sent 15% fewer input tokens and ran 13% faster at the median, because a precise route assembles less context. The one loss is question 1, "What is the return window for unopened items?". It carries no freshness cue, so Concept 1's rule routes it to CAG alone — and CAG holds the superseded policy. The treatment answered "thirty days" in all five repeats, and the judge scored its groundedness 1. That is the router ablation's lesson in reverse: a correct route to a stale cache still gives a stale answer. Keeping frozen content current is the Sync Mixer's job (`SyncCycle`, from the RAG+CAG batch), which this comparison deliberately doesn't run, and routing can't substitute for it.

Output token counts are the noisiest numbers in these reports. The same RAG-only baseline produced 8,948 and 9,140 output tokens in these two runs and 10,693 in an earlier one, so neither the prototype treatment's +32.6% nor the oracle's −6.1% is read as an effect.

Taken together, on this corpus the meta-layer's measurable value is MAG's answers, delivered only when routing selects MAG, plus smaller and faster contexts when routes are precise. A classifier that misses MAG erases that value entirely. That makes MAG recall, rather than exact-route accuracy, the classifier number worth improving next.

## What building against real infrastructure changed

- **A cancelled query takes its connection with it.** When a cascade timeout cancels a query mid-flight, SQLAlchemy terminates the asyncpg connection. After that, `rollback()` and `close()` both raise, and only `invalidate()` recovers. Every tier and the budget recorder therefore own their own unit of work. Tests against real Postgres pin that the pool's checked-out count returns to its baseline after both the awaited and the unawaited cancellation paths.
- **`localhost` stalls on this machine.** Raw asyncpg connections to the TestContainers Postgres through `localhost` timed out after 20 seconds, while `127.0.0.1` connected in about 40ms. Pooled connections had hidden this until invalidation forced fresh ones. The fixtures and the runner now pin `127.0.0.1`, and the three affected test files went from 535s to 23s.
- **Partial thresholds sit midway.** The plan's "must-miss score minus 0.05" would have turned an unrelated query into a partial CAG match carrying the superseded policy. The measured thresholds are CAG 0.43 hit / 0.35 partial and MAG 0.42 / 0.37.
- **The live model quoted the old value out of the current document.** An early version of the current policy text said the window "changed from thirty days today", and qwen3.5 answered a freshness question by quoting exactly that. The fixture now never names the superseded value, and the live test asserts the answer states 45 and never 30.
- **The event loop is part of every tier's budget.** See the second claim above.

## What review caught

The first review of `develop..HEAD` found no critical issues and seven important ones. Each was checked against the code, fixed test-first, and revert-checked where a revert could show the test catching it:

1. **The MAG tier shared the request's database session.** A cancelled MAG query would have broken the Postgres-backed hybrid RAG fallback behind it, and a parallel route put two tiers on one `AsyncSession`. The fix is a `SemanticFactSearch` port, implemented with a session per search.
2. **A classifier that raised or hung failed the whole request.** Classification now runs under a timeout, falls back to every tier in parallel, and records why in `routing_fallback`. The LLM classifier raises `ClassificationFailed` instead of returning 0.5 for every paradigm, a value that lands in the uncertainty band only at the default threshold.
3. **Scores with no signal routed confidently to CAG alone.** That is the one route where a weak frozen-cache match answers with nothing to contradict it. Those scores now fall back to every tier in parallel.
4. **Background RAG completion was unbounded.** Each completion now has a deadline, and concurrent completions are capped.
5. **The warmed-cache match iterated a live dict on a worker thread.** The fix scans a tenant-indexed snapshot. With the snapshot removed, a test warming a document mid-scan fails with "dictionary changed size during iteration".
6. **The budget write was scoped by tenant only.** It is now scoped by user too, and a real-Postgres test shows another user in the same tenant is refused.
7. **The budget was recorded after generation.** A missing session now fails before an answer is paid for.

Its minor findings were fixed in the same pass:

- A failing access tracker or findings sink no longer fails a request.
- A cancellation in one parallel tier now cancels its siblings.
- `BudgetShares` validation no longer lets a 5e-10 overshoot through under `isclose`'s default relative tolerance.
- An unmeasured stale rate is reported as n/a rather than 0%.
- The measured thresholds moved out of test code.

A follow-up review confirmed all seven fixes and raised a further round, also fixed:

- `drain()` now waits for cancelled tiers to finish cleaning up.
- A session close that fails after a successful search no longer discards the facts it found.
- `ClassificationFailed` carries only the reply's length, because a model reply can echo the user's question into a log.
- Budget recording has its own stage timing, so the stages add up to the whole request.

## What these measurements cannot show

- **CAG is a CPU proxy.** The tier checks a distilgpt2 `HFFrozenCache`, so what it measures is a lookup decision and its latency, not a real vLLM prefix-cache hit. That hit was measured separately, on real vLLM, in `cag-prefix-caching.md`.
- **The scale is small.** Six documents, three facts, ten comparison questions, six ablation phrasings, one machine, loopback stores.
- **One author.** The same person wrote the routing descriptions, reviewed the generated labels, labeled the comparison questions' routes, and built the classifiers.
- **Self-judging.** qwen3.5 both generates and judges, as in every earlier batch. The task-success checks are deterministic string checks, which is why they, not the judge scores, carry this report's conclusions.
- **Generous timeouts in the comparisons.** Both comparisons run the tiers under generous timeouts (5s / 5s / 10s), so answer quality isn't confounded with tier timeouts on a CPU machine. Latency against the real budgets is the separate measurement above.
- **In-process background work.** Background RAG completion doesn't survive a restart until `src/workers/` exists, and its limits (a 30-second deadline, 16 concurrent completions) are provisional.
- **Thresholds from this corpus.** Routing and match thresholds are measured defaults, not tuned against real traffic, which doesn't exist yet.

## What this batch does not do

- **It does not build the Freshness-Aware Data Router** (#155). That router decides once per data source, at ingestion time, rather than per query.
- **It does not change any HTTP endpoint.** `POST /chat` stays RAG-only. Exposing the unified path means accepting a `session_id` from a client, an object-level authorization surface (OWASP API1) that deserves its own batch and security review.
- **It does not put real vLLM behind the CAG tier.**
- **It does not make background RAG updates durable.**

## The numbers

- **Unit tests:** 777 on `develop` before this batch, 1,014 after, across 18 new unit test files (138 in all).
- **Integration tests:** 219 before, 239 after, across 5 new integration test files (58 in all). The full integration run passed 231 and skipped 8 in 9 minutes 48 seconds.
- **Skips:** all 8 are the pre-existing vLLM tests, which need this project's own GPU machine.
- **Ollama-backed tests:** the two in `test_orchestration_against_ollama.py` ran live against the local qwen3.5 model and passed. Wherever Ollama isn't reachable, they skip and state why.
- **Static checks:** `mypy` in strict mode reports no issues across the 215 source files in `src/`. `ruff` is clean on `src/` and on every file this batch created or changed.
