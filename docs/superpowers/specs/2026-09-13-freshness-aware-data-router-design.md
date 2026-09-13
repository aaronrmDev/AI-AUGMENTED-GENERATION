# Orchestration Meta-Layer Batch B: Freshness-Aware Data Router — Design Spec

**Scope:** the fifth and last component of the orchestration meta-layer, in `docs/architecture/OVERVIEW.md`'s "Freshness-Aware Data Router" section. It is sourced from `docs/inputs/concepts/unified_rag_cag_mag_architecture.md` Concept 9, "Freshness-Aware Data Routing", and sits at the top of that document's §3.4 Pattern 4 ("Data Sources → Freshness-Aware Routing → …"). Issues: Story #155 under Epic #150. This batch's Tasks are filed with its plan.

## What this component decides, and why it is its own batch

Batch A's three components each decide something per query. This one decides once per data source, at ingestion time: which paradigm should hold the data at all. Its caller, its clock, and its failure mode all differ from Batch A's:

- **Caller:** an ingestion feed, not a user's question.
- **Clock:** days of change history, not a 10ms budget.
- **Failure mode:** a stale answer served confidently for as long as a wrong placement lasts, not a slow request.

Concept 9 describes four behaviours. The batch builds all four:

1. **Classify data.** Tag every data source with its expected change frequency.
2. **Route at ingestion.** Volatile data goes to the RAG index only. Stable data goes to a CAG pre-load with RAG as backup. Session-specific data goes to MAG.
3. **Monitor and migrate.** Track each source's actual change frequency, and migrate it between paradigms when the pattern shifts.
4. **Detect stale cache entries.** Cached data has a TTL, and an expired entry falls back to RAG.

## The claims this batch exists to test

Concept 9 makes five claims. Each one gets a measured number rather than an assertion.

- **Volatile data in a frozen cache guarantees stale answers.** Measured: the stale-context rate for fast-changing sources, with every source cached and refreshed in batches, against freshness-aware routing.
- **Stable data in RAG wastes latency.** Measured: the share of probes a stable source answers from CAG under freshness-aware routing, against RAG-only placement. Tier latency itself was already measured in Batch A (`evaluation/reports/orchestration-meta-layer-cascade.md`: CAG p50 about 2ms, RAG about 5ms on the same stack), so it is cited rather than claimed anew.
- **Session data belongs in MAG.** Concept 9 argues this from mutability. This batch measures a consequence the source doesn't mention. A user's personal data placed in the tenant-wide RAG index or CAG cache is visible to every other user of that tenant; placed in MAG, it isn't.
- **Placements must migrate when change patterns shift.** Measured: for a source whose real rate drifts away from its declared rate, the simulated time from the shift to the migration, and the preload churn avoided afterwards.
- **A TTL bounds staleness.** Measured: for a cached source whose upstream changes while its ingestion feed has stalled, how long the stale copy is served with a TTL and without one.

## Decisions made for this batch

The user delegated design authority for this session ("take full control over the project"). Every decision a brainstorming session would normally put to them is recorded here with its reason, so any of them can be revisited.

1. **A data source is a named stream of text versions, not a file.**
   - A source is identified by a `source_key`, unique within its tenant, and for user-scoped sources within its user.
   - Each ingestion delivers the source's full current text.
   - A *change* is an ingestion whose content hash differs from the previous version's. Re-ingesting identical content is a *confirmation*, not a change.
   - Why: change frequency can only be observed on something that has versions, and the existing `Documents` table models one-shot uploads.
2. **Three routes, not eight.**
   - Concept 9's "How It Works" collapses its eight-row table into three routes: `RAG_ONLY`, `CAG_WITH_RAG_BACKUP`, and `MAG`.
   - The table's "CAG + RAG hybrid" rows are the stable route. Its "CAG" rows are the same route too, because the source's own step 2 says stable data goes to "CAG pre-load + RAG backup". RAG always holds a stable source's current text.
3. **The volatile boundary defaults to one day.**
   - A declared change interval shorter than one day routes `RAG_ONLY`; one day or longer routes `CAG_WITH_RAG_BACKUP`.
   - Why: the table puts "minutes-hours" on RAG and "daily-weekly" on the CAG + RAG hybrid, and one day is the boundary between those rows.
   - It is a constructor parameter, disclosed as a default rather than a tuned value.
4. **Scope decides MAG; frequency never does.**
   - A source declared user-scoped routes to `MAG` whatever its interval.
   - A tenant-scoped source never routes to `MAG`, and migration never moves data into or out of MAG.
   - Why: MAG's tables are keyed by user, and a change in how often data changes can't turn tenant data into personal data.
5. **CAG writes happen in a batch; invalidation happens immediately.**
   - When a stable source changes, its RAG copy is replaced and its CAG entry is evicted within the same ingestion.
   - The new text is pre-loaded into CAG only by the next run of `RefreshCachedSources`.
   - Why: this is Concept 9's own example ("When price changes → RAG index updates instantly; CAG price cache invalidated"), and it matches CAG's "batch invalidation" mutability in CLAUDE.md's paradigm table.
   - Consequence: pre-loading, the expensive step, stays batched.
6. **The TTL is anchored to the last confirmation, not to the pre-load.**
   - A cached entry expires `ttl` after the ingestion that last confirmed its content, where `ttl = expected_change_interval × ttl_factor` and `ttl_factor` defaults to 0.5.
   - The factor samples the source twice per expected change period.
   - A confirmation renews the TTL of an entry that is still cached.
   - `RefreshCachedSources` pre-loads only content confirmed within its TTL. An expired, unconfirmed source therefore stays on RAG until an ingestion confirms it again. Re-pre-loading the router's own stored copy would re-serve exactly the stale text the TTL exists to stop.
   - The TTL is enforced at lookup, so an expired entry is a cache miss and the cascade falls through to RAG without waiting for any sweep.
7. **Migration re-learns the profile, with hysteresis.**
   - `ReviewSourceFreshness` estimates each tenant-scoped source's observed change interval from its version history.
   - A cached source is demoted to `RAG_ONLY` when at least 3 changes fall within the 3 volatile boundaries before the review: change that is both sustained and recent, at least as fast as the boundary. Fast changes that ended before that window don't count, so a source that once changed quickly and has since gone quiet is left alone.
   - A `RAG_ONLY` source is promoted to `CAG_WITH_RAG_BACKUP` once it has gone 7 volatile boundaries without a change.
   - On migration, the observed interval replaces the declared one, so routing and TTL stay consistent with the source's real behaviour.
   - The gap between "3 changes within 3 days" and "7 quiet days" keeps a source from flapping, and the two conditions can never hold at once. Both numbers are defaults.
8. **The declared profile seeds a source; it doesn't overrule what was observed.**
   - The profile passed to the first ingestion creates the source.
   - Later ingestions keep the stored, possibly re-learned, interval.
   - Re-declaring a source with a different scope is refused, because moving data between tenant and user scope is a migration of ownership, which no batch here performs.
9. **The widely implemented RAG ports stay unchanged.**
   - Replacing a changed source's RAG text needs to delete its previous chunks from Qdrant and from Postgres; neither port can.
   - Six evaluation runners implement `DocumentRepository` in memory, so adding abstract methods would break all six for a capability only this batch needs.
   - `delete_document` is added as a concrete method on `QdrantVectorStore` and `PostgresDocumentRepository`. A new orchestration port, `RagIndex`, owns replace and remove, and its infrastructure adapter composes the concrete stores.
   - A stale chunk left in Postgres would still be served by hybrid RAG's `BM25KeywordSearch`, so both stores are cleared on every replace.
10. **No HTTP endpoint and no scheduler.**
    - The three use cases are driven by a caller on whatever cadence it chooses, the same shape as the existing `SyncCycle`.
    - Durable scheduling belongs to `src/workers/`, which doesn't exist yet.
    - An ingestion endpoint accepting source profiles from clients is an authorization surface that deserves its own review.

## Domain (`src/orchestration/domain/`)

**`entities.py` additions:**

- `SourceScope`: `TENANT`, `USER`.
- `IngestionRoute`: `RAG_ONLY`, `CAG_WITH_RAG_BACKUP`, `MAG`.
- `DataSourceProfile(source_key: str, scope: SourceScope, expected_change_interval: timedelta)`. Validation: a non-blank key, and a positive interval.
- `FreshnessPolicy(volatile_below: timedelta = 1 day, demote_after_changes: int = 3, promote_after_quiet_multiple: int = 7, ttl_factor: float | None = 0.5)`. Validation: positive values. `ttl_factor=None` disables expiry, which only the TTL ablation uses.
- `DataSource`: the stored record. It carries `id`, `tenant_id`, `user_id: UUID | None`, `source_key`, `scope`, `expected_change_interval`, `route`, `content_hash`, `last_changed_at`, `last_ingested_at`, and `cached_until: datetime | None`.
- `IngestionResult(source_id, route, changed: bool)`.
- `SourceMigration(source_key, from_route, to_route, observed_interval)`.
- `MigrationDecision(to_route: IngestionRoute, interval: timedelta)`.

**`freshness_router.py`,** the file OVERVIEW.md's module blueprint names:

- `route_for(profile, policy) -> IngestionRoute`: user scope goes to `MAG`; an interval below `policy.volatile_below` goes to `RAG_ONLY`; everything else goes to `CAG_WITH_RAG_BACKUP`.
- `cache_ttl(interval, policy) -> timedelta | None`: `interval × ttl_factor`, or `None` when `ttl_factor` is `None`.
- `observed_change_interval(times: Sequence[datetime]) -> timedelta | None`: the mean gap between consecutive timestamps; `None` with fewer than two.
- `decide_migration(source, version_times, now, policy) -> MigrationDecision | None`: the new route with its re-learned interval, or `None` for no migration. `MAG` sources always return `None`. The demote and promote rules follow decision 7. A demotion's interval is the mean gap between the changes inside the review window (measured from the version before the first of them); a promotion's is the time since the last change.

**`errors.py` additions:**

- `ScopeMismatch`: user scope without a `user_id`, or tenant scope with one.
- `ProfileConflict`: re-declaring an existing source with a different scope.

**`ports.py` additions:**

- `DataSourceRepository`: `get`, `save`, `append_version`, `version_times`, `current_content`, and `list_sources`, every method tenant-scoped. `get` also takes the optional `user_id`.
- `RagIndex`: `replace(tenant_id, document_id, title, text)` and `remove(tenant_id, document_id)`.
- `ExpiringCache(FrozenCache)`: adds `preload_until(tenant_id, document_id, content, expires_at: datetime | None)` and `renew(tenant_id, document_id, expires_at: datetime | None) -> bool`. `lookup` and `contains` report an expired entry as absent.
- `SessionFactWriter`: `record(tenant_id, user_id, fact_key, fact_value)`.

## Application (`src/orchestration/application/`)

**`IngestDataSource(repository, rag_index, cache, warmed, fact_writer, *, policy, route_policy=route_for)`.** `execute(tenant_id, profile, content, now, user_id=None) -> IngestionResult` runs these steps:

1. Validate scope against `user_id`.
2. Load the existing source, or create one routed by `route_policy(profile, policy)`.
3. Refuse a scope conflict.
4. Hash the content and decide whether it changed.
5. On a change, append a version.
6. Act on the route:
   - `RAG_ONLY`: on a change, `rag_index.replace`.
   - `CAG_WITH_RAG_BACKUP`: on a change, `rag_index.replace`, then `cache.evict` and `warmed.forget`. On a confirmation, `cache.renew` to the new TTL expiry.
   - `MAG`: on a change, `fact_writer.record`, with `fact_key` set to the source key.
7. Save the source with its hash, `last_changed_at`, `last_ingested_at`, and `cached_until`.

`route_policy` is injectable so the measurement's baselines can place every source on one route through the same code path. This is Batch A's ablation shape: a classifier of `None` gave the unrouted baseline.

**`RefreshCachedSources(repository, cache, warmed, *, policy)`.** `run(tenant_id, now) -> list[str]` handles every `CAG_WITH_RAG_BACKUP` source not currently cached whose `last_ingested_at + ttl` is still in the future. For each one it:

- pre-loads the current content with `expires_at = last_ingested_at + ttl`,
- notes it warmed,
- records `cached_until`.

It returns the keys it pre-loaded.

**`ReviewSourceFreshness(repository, cache, warmed, *, policy)`.** `run(tenant_id, now) -> list[SourceMigration]` applies `decide_migration` to every tenant-scoped source:

- A demotion evicts and forgets the cache entry and clears `cached_until`.
- A promotion changes only the route and interval; the next refresh pre-loads it.
- Both record the new route and the observed interval.

**`CacheWarmedRetrieve.forget(tenant_id, document_id)`.** It drops the warmed memo. An evicted document then stops shadowing other warmed documents in `best_warmed_match`, which confirms only its single best candidate against the cache.

## Infrastructure

- **`alembic/versions/0006_data_sources.py`.**
  - `data_sources` holds every column above, with `expected_change_seconds` as a positive `bigint`, and `route` and `scope` as `CHECK`-constrained text.
  - A `CHECK` requires `user_id IS NOT NULL` exactly when `scope = 'user'`.
  - Uniqueness is on `(tenant_id, user_id, source_key)` `NULLS NOT DISTINCT` (PostgreSQL 15+; the stack runs 16).
  - `data_source_versions` holds `id`, `data_source_id` (FK), `tenant_id`, `content_hash`, `content`, and `ingested_at`, indexed on `(data_source_id, ingested_at)`.
  - Both tables get `tenant_isolation` RLS and `app_user` grants from creation.
- **`PostgresDataSourceRepository(sessionmaker)`.** Each call sets the tenant context and commits its own short transaction, following `PostgresSessionBudgetRecorder`.
- **`ExpiringFrozenCache(inner: FrozenCache, clock)`.** It wraps any `FrozenCache`. An entry past its expiry is evicted from the inner cache on first sight and reported absent.
- **`ChunkedRagIndex(sessionmaker, vector_store: QdrantVectorStore, chunker, embedder)`.**
  - `replace` deletes the document's Qdrant points by `tenant_id` and `document_id` payload filter, and its Postgres chunks and row.
  - It then saves a fresh `documents` row (filename set to the source key, storage path `data-source:<key>`, status `completed`) and its chunks.
  - The Postgres side runs in one transaction under RLS.
- **`RecordSemanticFactWriter(record_semantic_fact)`.** It adapts MAG's existing `RecordSemanticFact` command, unmodified, to `SessionFactWriter`.
- **`QdrantVectorStore.delete_document` and `PostgresDocumentRepository.delete_document`.** New concrete methods, each with its own integration test.

## Tenant and user isolation

- Every repository call takes `tenant_id` and runs under RLS.
- User-scoped sources carry `user_id`, and uniqueness includes it, so two users' identically named sources stay separate.
- The frozen cache, the warmed memo, the RAG index, and MAG facts are keyed by tenant already, and MAG facts by user as well.
- The integration tests check the following:
  - another tenant can't read, replace, or review a source;
  - another user's same-named source is a different record;
  - under the default `route_for`, a user-scoped source never reaches the RAG index or the frozen cache.

## Testing plan

Unit tests use fakes and an injected clock, and never assert on wall-clock time.

- **`route_for`** is tested against each of Concept 9's eight rows as a declared profile, at the exact volatile boundary, and with user scope.
- **`decide_migration`** is tested for:
  - demotion at exactly 3 changes inside the window and not at 2;
  - a change exactly on the window's edge;
  - no demotion for fast changes that ended before the window;
  - promotion at exactly 7 boundaries of quiet and not before;
  - `MAG` never migrating;
  - the promoted interval.
- **`observed_change_interval`** is tested with too few versions and with unevenly spaced versions.
- **`IngestDataSource`** is tested for:
  - each route's first ingestion;
  - a confirmation versus a change on each route;
  - eviction and forgetting on a change to a cached source;
  - TTL renewal on a confirmation;
  - scope validation and profile conflicts;
  - an injected `route_policy`.
- **`RefreshCachedSources`** is tested for:
  - pre-loading only uncached, confirmed-within-TTL sources;
  - expiry anchored to the last confirmation;
  - skipping `RAG_ONLY` and `MAG` sources;
  - `ttl_factor=None`.
- **`ReviewSourceFreshness`** is tested for applying demotion (eviction included) and promotion (no pre-load), and for leaving user-scoped sources alone.
- **`ExpiringFrozenCache`** is tested at expiry, one tick before it, on renewal, on `None` expiry, and with a plain `preload`.
- **`CacheWarmedRetrieve.forget`** is tested to show a forgotten document no longer shadows a valid one.

Integration tests use real dependencies:

- the migration's constraints and RLS;
- `PostgresDataSourceRepository` round trips and tenant refusal;
- both `delete_document` methods;
- `ChunkedRagIndex.replace` leaving no superseded chunk in either store;
- the MAG writer against real Postgres, Qdrant, and Neo4j;
- an end-to-end test: ingest a volatile source, a stable source, and a user-scoped source, refresh, and query through Batch A's `LatencyCascade`. It asserts:
  - the volatile answer comes from RAG;
  - the stable answer comes from CAG;
  - a changed stable source is served from RAG until the next refresh;
  - an expired entry falls back to RAG;
  - the personal fact is visible to its user and not to another.

## Measurement plan

The runner lives under `evaluation/scenarios/freshness-router/`, and the narrative report goes to `evaluation/reports/freshness-router.md`. Every part runs against real Postgres, Qdrant, Neo4j, MiniLM, and a distilgpt2 `HFFrozenCache`, on a simulated clock. No LLM is used: staleness is judged on the assembled context, with version-specific markers in every source's text. That is deterministic, and it measures what the router controls.

1. **Route table.** Each of Concept 9's eight data types is declared as a profile and routed. The report sets each route beside the source's own paradigm column and names any row the single-interval rule can't express. Code repositories are an example: their "cache main branch, RAG for PRs" split is two sources, not one.
2. **A 30-day placement ablation.**
   - **The corpus** of six sources, each with one fixed question:
     - a price feed declared at 1 hour, changing hourly;
     - a return policy declared at 90 days, changing once, on day 12;
     - a catalog declared at 7 days, changing daily until day 10 and hourly after that;
     - a shipping guide declared at 365 days, never changing;
     - a flash-sale feed declared at 1 hour, changing hourly until day 5 and never after;
     - a user's size preference, user-scoped and declared at 30 days, changing once, on day 15.
   - **The schedule:** ingestions follow each source's real schedule. Refresh and review run daily at midnight. Probes run every 3 simulated hours through Batch A's unrouted `LatencyCascade` (Concept 5 as drawn, which trusts a CAG hit); the size question is asked as its owner and as another user.
   - **Four arms:**
     - *cache everything, batch refresh only*: a runner-local baseline, which re-pre-loads every source nightly and never invalidates on change;
     - *cache everything, invalidate on change*: `IngestDataSource` with `route_policy` always `CAG_WITH_RAG_BACKUP`, and no review;
     - *RAG only*: `route_policy` always `RAG_ONLY`, and no review;
     - *freshness-aware*: the defaults, with review.
   - **Reported per arm and source:**
     - probes;
     - stale-only, mixed, and fresh-only contexts;
     - the share of probes answered from CAG, meaning the assembled context holds a CAG item;
     - pre-loads performed;
     - pre-loads evicted before any probe hit them;
     - probes by the non-owning user whose context contained the owner's preference.
3. **Migration lag.** In the freshness-aware arm, the catalog source starts changing hourly on day 10 and the flash-sale feed goes quiet on day 5. For each migration the report gives the simulated time of the shift, the time of the migration, and the pre-loads before and after.
4. **TTL bound.** Run on its own over the same simulated month: a stable source, declared at 7 days and ingested daily, is changed directly in the RAG index on day 8, while its ingestion feed has stalled since day 7. The freshness-aware policy is run with `ttl_factor=0.5` and with `ttl_factor=None`. The report gives the hours during which the superseded text was served from CAG.

The report states plainly what these numbers can't show:

- The change schedules are synthetic.
- One author wrote the corpus, the schedules, and the rules.
- The CAG cache is a CPU proxy.
- Latency differences are cited from Batch A rather than re-measured under this simulated clock.

## What this batch does not do

- **It does not add an HTTP endpoint or change `UploadDocument`.** Ingestion through the router is a use case callers compose. The existing upload path stays as it is, and its documents are not tracked as data sources.
- **It does not schedule anything.** Refresh and review run when a caller drives them; durable cadence arrives with `src/workers/`.
- **It does not migrate data into or out of MAG,** for the reason in decision 4.
- **It does not model the code-repository split** between a main branch and pull requests; that is two sources.
- **It does not put real vLLM behind the cache.** The frozen cache is the same CPU proxy Batch A used.
- **It does not tune its thresholds against real change data,** which doesn't exist yet. The one-day boundary, the 3-change and 7-quiet hysteresis, and the 0.5 TTL factor are disclosed defaults.
