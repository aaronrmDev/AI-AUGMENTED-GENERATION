# Orchestration Meta-Layer Batch B: Freshness-Aware Data Router — Live Measurement Report

**Scope:** Story #155 under Epic #150; Tasks #169–#180. The design is `docs/superpowers/specs/2026-09-13-freshness-aware-data-router-design.md`, and the plan, with every place execution changed it, is `docs/superpowers/plans/2026-09-13-freshness-aware-data-router.md`. Every number below comes from the generated tables in `freshness-router-measurements.md`.

## What this batch is

Batch A built the three meta-layer components that decide something per query. This batch builds the fifth and last: Concept 9's Freshness-Aware Data Router, which decides once per data source, at ingestion, which paradigm should hold that data at all.

- **The rules** are pure functions in `src/orchestration/domain/freshness_router.py`. A declared change interval under one day routes RAG only; a day or longer routes to a CAG pre-load with RAG as backup; a user-scoped source routes to MAG. A cached copy's TTL is half the interval. Three changes inside three days demote a cached source, and seven quiet days promote a RAG-only one.
- **`IngestDataSource`** applies the routing to each delivered version. A change replaces RAG's copy, and evicts a cached copy at once.
- **`RefreshCachedSources`** is the batch pre-load. It pre-loads only content an ingestion confirmed within its TTL.
- **`ReviewSourceFreshness`** re-learns each tenant source's interval from its version history, and migrates the source when it drifts.

Existing code changed in three places:

- `CacheWarmedRetrieve.forget`, so an evicted document stops shadowing valid ones;
- concrete `delete_document` methods on `QdrantVectorStore` and `PostgresDocumentRepository`, so a replaced source leaves no superseded chunk in either store;
- migration 0006, for `data_sources` and `data_source_versions`, with RLS from creation.

The shared RAG ports are unchanged.

## How it was measured

The runner (`evaluation/scenarios/freshness-router/run_freshness_measurements.py`) plays out 30 simulated days against real Postgres, Qdrant, Neo4j, MiniLM, and a distilgpt2 frozen cache:

- **Sources.** Six: an hourly price feed; a return policy that changes once, on day 12 at 09:00; a catalog that changes daily and, from day 10, hourly; a static shipping guide; a flash sale that changes hourly until day 5 and then goes quiet; and one user's size preference, which changes on day 15.
- **Cadence.** Every source is ingested hourly. Refresh runs at midnight, and review too in the freshness-aware arm.
- **Probes.** Every source is probed every 3 hours through Batch A's unrouted cascade, which trusts a CAG hit exactly as Concept 5 draws it. The size question is asked by its owner and by another user of the same tenant.
- **Scoring.** Staleness is read from the assembled context, using version-specific markers. No LLM is involved.

Four arms place the same sources:

- **Cache everything, batch refresh only:** every source goes to CAG and RAG, and the cache is re-warmed nightly with no invalidation in between.
- **Cache everything, invalidate on change:** `IngestDataSource` with every source forced onto the cached route.
- **RAG only:** every source forced onto RAG.
- **Freshness-aware:** the defaults, with review.

## Claim 1: volatile data in a frozen cache guarantees stale answers

| Source | Batch refresh only | Invalidate on change | RAG only | Freshness-aware |
|---|---|---|---|---|
| Hourly price feed | 210 stale-only of 241 (87%) | 0 | 0 | 0 |
| Catalog (hourly from day 10) | 147 (61%) | 0 | 0 | 0 |
| Flash sale (hourly until day 5) | 28 (12%) | 0 | 0 | 0 |
| Return policy (one change, 09:00) | 5 (2%) | 0 | 0 | 0 |

The claim holds for a cache that is only refreshed in batches. The price feed served its previous hour's price on every probe except the one that fell right after the nightly re-warm. It doesn't hold as stated for a cache that is invalidated when a source changes: that arm never served superseded text either, with no routing at all.

The honest reading, then, is that invalidation removes staleness, and routing decides whether caching a source is worth doing. That cost appears under pre-load churn, below. No arm ever produced a mixed context, because a source's RAG copy is replaced in place, so old and new text never sit side by side.

## Claim 2: stable data in RAG wastes latency

Share of probes answered from CAG:

| Source | RAG only | Freshness-aware |
|---|---|---|
| Shipping guide (never changes) | 0% | 100% |
| Return policy | 0% | 98% |
| Flash sale (promoted on day 12) | 0% | 63% |
| Catalog (demoted on day 11) | 0% | 30% |
| Hourly price feed | 0% | 0% |

Freshness-aware placement kept its stable sources answering from the cache, and kept the volatile feed out of it. How much latency that saves is cited rather than re-measured here, since this run's clock is simulated. On this same stack, Batch A measured CAG at a median of about 2ms and RAG at about 5ms (`evaluation/reports/orchestration-meta-layer-cascade.md`). The measured difference per probe is therefore a few milliseconds on a six-document corpus. The claim's direction holds; its size on real traffic isn't shown.

## Claim 3: session data belongs in MAG

Probes by one user whose context contained another user's size preference:

| Batch refresh only | Invalidate on change | RAG only | Freshness-aware |
|---|---|---|---|
| 241 of 241 | 241 of 241 | 241 of 241 | 0 of 241 |

This is the batch's largest measured effect, and Concept 9 doesn't mention it. It argues for MAG from mutability. Any tenant-wide placement of a user's personal fact, in the RAG index or the frozen cache, put that fact in front of every other user of the tenant, on every probe. Placed in MAG, it reached its owner on all 241 probes, current throughout, including after its day-15 change, and it never reached the other user.

## Claim 4: placements must migrate when change patterns shift

| Source | Migration | Pattern shift | Migrated | Lag |
|---|---|---|---|---|
| Catalog | cached → RAG only | day 10, 00:00 | day 11, 00:00 | 24 h |
| Flash sale | RAG only → cached | day 5, 00:00 | day 12, 00:00 | 168 h |

Both lags are what the rules and the review cadence allow, not detection delays:

- **Catalog.** The demotion rule was satisfied three hours after the catalog turned hourly. The nightly review acted at the next midnight.
- **Flash sale.** Promotion waits seven quiet days by design, which is the hysteresis that keeps a source from flapping.

Before its demotion, the catalog was pre-loaded 10 times and served 30% of probes from CAG. The invalidate-on-change arm, which never demotes, pre-loaded it 31 times for a 39% share. The 21 extra pre-loads after day 10 each served about one probe before the next hourly change evicted them.

## Claim 5: a TTL bounds staleness

A cached warranty source, declared at 7 days, was confirmed daily through day 7. On day 8 its text was changed directly in the RAG index, behind the router's back, while its ingestion feed stayed stalled.

| Policy | Probes serving superseded text from CAG |
|---|---|
| TTL factor 0.5 | 20 (60 simulated hours) |
| No TTL | 185 (555 simulated hours, the rest of the month) |

The TTL bounded the exposure at 3.5 days after the last confirmation, so the entry expired on day 10 at 12:00. After that, the batch refresh wouldn't re-pre-load unconfirmed content, and every probe was answered from RAG's current text. Without a TTL, the stale copy was served until the run ended.

## Pre-load churn

The planned "never-served pre-loads" column reads 0 for every arm and source. A probe runs at midnight, right after the refresh, so every pre-load serves at least that one probe before any eviction. The metric is correct but can't show churn on this probe schedule. Probes served per pre-load, taken from the same table, can:

| Arm and source | Pre-loads | Probes served from CAG | Per pre-load |
|---|---|---|---|
| Invalidate on change, hourly price feed | 31 | about 31 (13% of 241) | about 1 |
| Invalidate on change, flash sale | 31 | about 31 (13%) | about 1 |
| Freshness-aware, return policy | 2 | 236 (98%) | 118 |
| Freshness-aware, shipping guide | 1 | 241 (100%) | 241 |
| Freshness-aware, flash sale | 1 | about 152 (63%) | about 152 |

Invalidating on change kept the volatile sources fresh by paying for a pre-load every night that one hourly change threw away. On the real GPU cache this CPU proxy stands in for, a pre-load is a full prefill. Routing the volatile feed away from CAG avoided every one of those.

## Concept 9's spectrum, routed

Seven of Concept 9's eight data types route where the source puts them. The exception is code repositories: the source's "cache main branch, RAG for PRs" split is really two sources with different change rates, and one declared interval can't express it.

## What building it changed

- **Recalibrated wording.** The runner's calibration rejected the planned corpus before any arm ran:
  - A shoe catalog scored 0.49 against the size-preference question.
  - A "product" warranty question scored 0.36 against the return policy.

  The catalog, the flash sale, and the warranty moved to distinct topics (kitchen blenders, garden hoses, a bicycle frame). Every question then scored at least 0.52 against its own source and at most 0.32 against any other. Batch A's thresholds were not changed.
- **Interval parameters.** A schema test first bound intervals as strings through a cast, and asyncpg rejected them before Postgres checked any constraint. The test binds `timedelta` values now.
- **End-to-end tests held first time.** All six passed on their first run, with the planned questions and Batch A's thresholds.

## What these measurements cannot show

- **Synthetic schedules and one author.** The change schedules, the corpus, the markers, and the rules share one author.
- **A simulated clock.** Nothing here measures wall-clock latency, concurrency, or scheduling. Tier latency is cited from Batch A.
- **Contexts, not answers.** Staleness and exposure are read from the assembled context. Whether a model would repeat the stale or foreign text wasn't measured, although Batch A's comparison showed qwen3.5 reproducing a stale cached value when that was all it was given.
- **A CPU cache proxy.** The frozen cache is distilgpt2 on CPU, so the cost of a pre-load is argued from what a prefill is, not measured on a GPU.
- **One probe cadence.** Probes every 3 hours, aligned with a midnight refresh, is why the never-served column reads 0.
- **One unexplained retrieval miss.** One of 241 RAG-only flash-sale probes held neither the current nor any superseded text. Every other RAG probe found its source. A plausible cause, which this run doesn't verify, is approximate vector search recall under this run's churn of tens of thousands of deletes and re-inserts in one collection.
- **Default thresholds.** The one-day boundary, the 3-change and 7-quiet-day hysteresis, and the 0.5 TTL factor are disclosed defaults, not tuned against real change data.

## What this batch does not do

- **It adds no HTTP endpoint and doesn't change `UploadDocument`.** Freshness-routed ingestion is a use case callers compose; uploaded documents aren't tracked as data sources.
- **It schedules nothing.** Refresh and review run when a caller drives them. Durable scheduling belongs to `src/workers/`.
- **It moves no data into or out of MAG by frequency,** because scope decides MAG.
- **It doesn't model the code-repository split.**
- **It puts no real vLLM behind the cache.**

## The numbers

- **Unit tests:** 1,037 on `develop` before this batch, 1,109 after, across 9 new unit test files (148 in all).
- **Integration tests:** 239 before, 260 after, across 4 new integration test files and 2 new tests in `test_migration.py` (62 files in all). The full integration run passed 252 and skipped 8.
- **Skips:** all 8 are the pre-existing vLLM tests, which need this project's own GPU machine.
- **Static checks:** `mypy` in strict mode reports no issues across the 223 source files in `src/`. `ruff` is clean on `src/` and on every file this batch created or changed.
