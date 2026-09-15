# Orchestration Meta-Layer — Cascade Measurements

Corpus: 6 documents in Qdrant (4 also frozen into a distilgpt2 HFFrozenCache on CPU, the return policy in its superseded thirty-day version), 3 MAG semantic facts in Postgres. Thresholds carried over unchanged from the integration corpus: CAG hit 0.43, partial 0.35; MAG hit 0.42, partial 0.37. Tier latency: 10 questions x 5 repeats, prototype routing, default TierTimeouts, after one untimed warm-up. Query embedding (MiniLM, CPU) is paid before any tier runs: p50 9.20ms, max 11.46ms. The RAG retriever shares a CachingEmbeddingModel with the pipeline, so its embedding of the already-embedded question is a lookup rather than CPU work on the event loop, which starved the other tiers in PARALLEL routes before it was wired. A timed-out CAG match keeps its worker thread running to completion (threads can't be cancelled), so a tier measured right after a CAG timeout can include that overlap. Allocator sweep: the same 10 prototype-routed questions, assembled at each window with dynamic reallocation and with static base slices.

## Tier latency against Concept 5's budgets

| Tier | Attempts | Outcomes | p50 ms | p95 ms | Budget ms | Within budget |
|---|---|---|---|---|---|---|
| CAG | 45 | hit 30, miss 15 | 1.99 | 2.54 | 10 | 100% |
| MAG | 10 | miss 10 | 7.53 | 9.77 | 50 | 100% |
| RAG | 40 | hit 40 | 5.25 | 6.55 | 2000 | 100% |

## Superseded frozen-cache text reaching the model, router off vs. on

| Arm | Runs | Stale only | Current only | Both | Stale rate |
|---|---|---|---|---|---|
| router off (Concept 5 cascade as drawn) | 6 | 4 | 1 | 1 | 67% |
| router on (lexical) | 6 | 0 | 3 | 3 | 0% |
| router on (prototype, MiniLM) | 6 | 0 | 1 | 5 | 0% |

## Dynamic reallocation vs. static base slices

| Window tokens | Turns | Dropped (dynamic) | Dropped (static) | Tokens used (dynamic) | Tokens used (static) |
|---|---|---|---|---|---|
| 1000 | 10 | 0 | 0 | 616 | 616 |
| 400 | 10 | 0 | 0 | 616 | 616 |
| 250 | 10 | 0 | 8 | 616 | 461 |
| 150 | 10 | 6 | 14 | 500 | 306 |
