# Orchestration Meta-Layer Batch B: Freshness-Aware Data Router — Live Measurement Report

**Scope:** Story #155 under Epic #150; Tasks #169–#180. The design is `docs/superpowers/specs/2026-09-13-freshness-aware-data-router-design.md`, and the plan, with every place execution changed it, is `docs/superpowers/plans/2026-09-13-freshness-aware-data-router.md`. Every number below comes from the generated tables in `freshness-router-measurements.md`.

## What this batch is

Batch A built the three meta-layer components that decide something per query. This batch builds the fifth and last: Concept 9's Freshness-Aware Data Router, which decides once per data source, at ingestion, which paradigm should hold that data at all.

- **The rules** are pure functions in `src/orchestration/domain/freshness_router.py`. A declared change interval under one day routes RAG only; a day or longer routes to a CAG pre-load with RAG as backup; a user-scoped source routes to MAG. A cached copy's TTL is half the interval. Three changes inside three days demote a cached source, and seven quiet days promote a RAG-only one.
- **`IngestDataSource`** applies the routing to each delivered version. A change replaces RAG's copy, and evicts a cached copy at once. The change stays marked pending until every effect and the save have landed, so a failure part-way is re-applied by the next ingestion.
- **`RefreshCachedSources`** is the batch pre-load. It pre-loads only content an ingestion confirmed within its TTL.
- **`ReviewSourceFreshness`** re-learns each tenant source's interval from its version history, and migrates the source when it drifts.

The refresh and the review write only while the source still has the hash they read and no change is pending, so neither can cache or migrate a source that changed under it.

Existing code changed in three places:

- `CacheWarmedRetrieve`: `forget` drops a warmed memo, and `best_warmed_match` falls through to the best candidate the cache still confirms, so an evicted or expired document never shadows valid ones;
- concrete `delete_document` methods on `QdrantVectorStore` and `PostgresDocumentRepository`, so a replaced source leaves no superseded chunk in either store. Qdrant's can keep named chunks, so a replacement is upserted before the old version is deleted;
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

That result depends on a condition this simulation makes perfect: the router hears of every change in the hour it happens, because every source is ingested hourly and every change lands on the hour. With real ingestion lag, an invalidate-on-change cache keeps serving superseded text until the change arrives, as a batch-refreshed cache does until its refresh. The honest reading, then, is that invalidation removes the staleness a cache adds beyond ingestion lag, and routing decides whether caching a source is worth doing. That cost appears under pre-load churn, below.

No arm ever produced a mixed context. Two things guarantee that here. A source's RAG copy is replaced in place, so old and new text never sit side by side in RAG. And the cascade stops at the first tier that hits, so a context built from a CAG hit never also carries RAG's copy.

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

This is the batch's largest effect, and Concept 9 doesn't mention it. It argues for MAG from mutability. Any tenant-wide placement of a user's personal fact, in the RAG index or the frozen cache, put that fact into every other user's assembled context, on every probe. Placed in MAG, the fact reached its owner on all 241 probes, current throughout, including after its day-15 change, and it never reached the other user.

The 241 of 241 is structural rather than a retrieval finding. Calibration guarantees that the size question retrieves the size source from whichever store holds it, and a tenant-wide store holds it for every user. What the measurement adds is that nothing between ingestion and the assembled context stopped it.

## Claim 4: placements must migrate when change patterns shift

| Source | Migration | Pattern shift | Migrated | Lag | Pre-loads before | Pre-loads after |
|---|---|---|---|---|---|---|
| Catalog | cached → RAG only | day 10, 00:00 | day 11, 00:00 | 24 h | 10 | 0 |
| Flash sale | RAG only → cached | day 5, 00:00 | day 12, 00:00 | 168 h | 0 | 1 |

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
| Invalidate on change, hourly price feed | 31 | 31 (13% of 241) | 1 |
| Invalidate on change, flash sale | 31 | 31 (13%) | 1 |
| Freshness-aware, return policy | 2 | 236 (98%) | 118 |
| Freshness-aware, shipping guide | 1 | 241 (100%) | 241 |
| Freshness-aware, flash sale | 1 | 153 (63%), every probe from day 12 on | 153 |

Every invalidate-on-change pre-load of those two sources served exactly one probe, the midnight probe right after the refresh, and then expired. Invalidation isn't what ended them. Both sources declare a one-hour interval, so their TTL is 30 minutes. Each entry expired at 00:30, before the 01:00 ingestion could change or confirm it, and a confirmation can't renew an entry that's already gone. The flash sale shows it plainly: it stopped changing on day 5, and still served one probe per pre-load for the rest of the month.

This churn comes from caching a source whose TTL is shorter than its ingestion cadence (see "Known limits of the rules"). On the real GPU cache this CPU proxy stands in for, each of those pre-loads is a full prefill. Freshness-aware routing kept both sources off the cache while they changed hourly. Once the flash sale went quiet, one promotion and one pre-load served its 153 remaining probes.

## Concept 9's spectrum, routed

Seven of Concept 9's eight data types route where the source puts them. The exception is code repositories: the source's "cache main branch, RAG for PRs" split is really two sources with different change rates, and one declared interval can't express it.

## What building it changed

- **Recalibrated wording.** The runner's calibration rejected the planned corpus before any arm ran:
  - A shoe catalog scored 0.49 against the size-preference question.
  - A "product" warranty question scored 0.36 against the return policy.

  The catalog, the flash sale, and the warranty moved to distinct topics (kitchen blenders, garden hoses, a bicycle frame). Every question then scored at least 0.52 against its own source and at most 0.32 against any other. Batch A's thresholds were not changed.
- **Interval parameters.** A schema test first bound intervals as strings through a cast, and asyncpg rejected them before Postgres checked any constraint. The test binds `timedelta` values now.
- **End-to-end tests held first time.** All six passed on their first run, with the planned questions and Batch A's thresholds.
- **The final review.** An independent review of the whole branch found six important issues, each fixed test-first before the merge:
  - Races. The refresh and the review acted on snapshots that a concurrent ingestion could supersede.
  - Partial failure. An ingestion that evicted the cache and then failed to save let the next refresh re-cache superseded text.
  - Shadowing. An expired warmed memo hid live documents scoring below it.
  - Report accuracy. Two conclusions in this report were overstated: the flash sale's pre-load churn was attributed to changes, and invalidation's zero staleness was stated without its zero-lag condition.
  - Personal text. A MAG source's text was copied into the version history.
  - User deletes. The user foreign key had no delete rule.

  The first three are fixed by the pending marker, the conditional writes, and the fall-through described above. The report's two conclusions are corrected here, MAG versions now carry no text, and user deletes cascade. The plan's execution notes record each fix, with the minor findings fixed alongside them.
- **Undisclosed departures.** The same review found two plan departures the report hadn't named. Every source is ingested hourly rather than on its own schedule, so confirmations exist to anchor a TTL. And the migration table first lacked its planned pre-load columns, which it now has.

## What these measurements cannot show

- **Synthetic schedules and one author.** The change schedules, the corpus, the markers, and the rules share one author.
- **A simulated clock.** Nothing here measures wall-clock latency, concurrency, or scheduling. Tier latency is cited from Batch A.
- **Contexts, not answers.** Staleness and exposure are read from the assembled context. Whether a model would repeat the stale or foreign text wasn't measured, although Batch A's comparison showed qwen3.5 reproducing a stale cached value when that was all it was given.
- **A CPU cache proxy.** The frozen cache is distilgpt2 on CPU, so the cost of a pre-load is argued from what a prefill is, not measured on a GPU.
- **CAG share by paradigm, not by document.** A probe counts as served from CAG when its context holds any CAG item. The check doesn't confirm that the item is the probed source's own document. Calibration keeps every question well below the hit threshold against other sources, which makes a cross-source CAG hit unlikely, but the share isn't verified document by document.
- **No concurrency or failure.** Ingestion, refresh, review, and probes run one after another, and no effect fails. The pending marker and the conditional writes that protect against races and partial failures are covered by unit and integration tests, not by this run.
- **One probe cadence.** Probes every 3 hours, aligned with a midnight refresh, is why the never-served column reads 0.
- **One retrieval miss that didn't recur.** In the first full run, one of 241 RAG-only flash-sale probes held neither the current nor any superseded text. In the rerun after the final review, every RAG probe found its source, and every other number reproduced exactly. The rerun also changed how Qdrant replaces a document (upsert first, then delete), so it can't separate chance from that change. A plausible cause, which neither run verifies, is approximate vector search recall under tens of thousands of deletes and re-inserts in one collection.
- **Default thresholds.** The one-day boundary, the 3-change and 7-quiet-day hysteresis, and the 0.5 TTL factor are disclosed defaults, not tuned against real change data.

## Known limits of the rules

- **A confirmation renews only a live entry.** A source whose TTL is shorter than its ingestion cadence drops out of the cache between confirmations, and returns only at the next refresh after one. With hourly ingestion, any declared interval under two hours has that problem.
- **A burst of corrections shortens a stable source's TTL for good.** Three changes inside three days demote a source declared at, say, 60 days. Seven quiet days later it is promoted with a re-learned interval of about seven days, and its TTL drops from 30 days to about 3.5. Nothing lengthens a cached source's interval again.
- **`cached_until` is a record, not a control.** The cache enforces its own expiry; the column exists for operators and the measurement runner.

## What this batch does not do

- **It adds no HTTP endpoint and doesn't change `UploadDocument`.** Freshness-routed ingestion is a use case callers compose; uploaded documents aren't tracked as data sources.
- **It schedules nothing.** Refresh and review run when a caller drives them. Durable scheduling belongs to `src/workers/`.
- **It moves no data into or out of MAG by frequency,** because scope decides MAG.
- **It doesn't model the code-repository split.**
- **It puts no real vLLM behind the cache.**

## The numbers

- **Unit tests:** 1,037 on `develop` before this batch, 1,123 after, across 9 new unit test files (148 in all). The final review's fixes account for 14 of the new tests.
- **Integration tests:** 239 before, 265 after, across 4 new integration test files and 2 new tests in `test_migration.py` (62 files in all). The final review's fixes account for 5 of the new tests. The full integration run after those fixes passed 257 and skipped 8.
- **Skips:** all 8 are the pre-existing vLLM tests, which need this project's own GPU machine. The two Ollama tests ran, because a local Ollama model was available.
- **Static checks:** `mypy` in strict mode reports no issues across the 223 source files in `src/`. `ruff` is clean on `src/` and on every file this batch created or changed.
