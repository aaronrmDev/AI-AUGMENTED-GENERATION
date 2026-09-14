# Orchestration Meta-Layer Batch B: Freshness-Aware Data Router — Design Spec

**Scope:** the fifth and last component of the orchestration meta-layer, in `docs/architecture/OVERVIEW.md`'s "Freshness-Aware Data Router" section. It is sourced from `docs/inputs/concepts/unified_rag_cag_mag_architecture.md` Concept 9, "Freshness-Aware Data Routing", and sits at the top of that document's §3.4 Pattern 4 ("Data Sources → Freshness-Aware Routing → …"). Issues: Story #155 under Epic #150; Tasks #169–#180, one per plan task in order.

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
   - A confirmation renews the TTL of an entry that is still cached. It can't revive an expired one, so a source ingested less often than its TTL drops out of the cache between confirmations and returns only at the next refresh after one.
   - `RefreshCachedSources` pre-loads only content confirmed within its TTL. An expired, unconfirmed source therefore stays on RAG until an ingestion confirms it again. Re-pre-loading the router's own stored copy would re-serve exactly the stale text the TTL exists to stop.
   - The TTL is enforced at lookup, so an expired entry is a cache miss and the cascade falls through to RAG without waiting for any sweep.
7. **Migration re-learns the profile, with hysteresis.**
   - `ReviewSourceFreshness` estimates each tenant-scoped source's observed change interval from its version history.
   - A cached source is demoted to `RAG_ONLY` when its last 3 changes, and the version before them, all fall strictly inside the 3 volatile boundaries before the review. Their mean gap is then under the boundary, so the re-learned interval routes `RAG_ONLY` too. Fast changes that ended before that window don't count, so a source that once changed quickly and has since gone quiet is left alone.
   - A `RAG_ONLY` source is promoted to `CAG_WITH_RAG_BACKUP` once it has gone 7 volatile boundaries without a change.
   - On migration, the observed interval replaces the declared one, so routing and TTL stay consistent with the source's real behaviour. A demotion's interval is at least one second, so a burst of changes under one timestamp can't re-learn a zero interval the schema refuses.
   - Consequence: a stable source that a burst of corrections demotes comes back, seven quiet boundaries later, with an interval of about seven boundaries, whatever it declared. Its TTL stays that short, because nothing lengthens a cached source's interval again.
   - The gap between "3 changes within 3 days" and "7 quiet days" keeps a source from flapping, and the two conditions can never hold at once. Both numbers are defaults.
8. **The declared profile seeds a source; it doesn't overrule what was observed.**
   - The profile passed to the first ingestion creates the source.
   - Later ingestions keep the stored, possibly re-learned, interval.
   - Scope is part of a source's identity: the same key declared tenant-wide, or by a different user, is a different source. Moving data between tenant and user scope would be a migration of ownership, which no batch here performs.
9. **The widely implemented RAG ports stay unchanged.**
   - Replacing a changed source's RAG text needs to delete its previous chunks from Qdrant and from Postgres; neither port can.
   - Six evaluation runners implement `DocumentRepository` in memory, so adding abstract methods would break all six for a capability only this batch needs.
   - `delete_document` is added as a concrete method on `QdrantVectorStore` and `PostgresDocumentRepository`. A new orchestration port, `RagIndex`, owns replacement, and its infrastructure adapter composes the concrete stores.
   - A stale chunk left in Postgres would still be served by hybrid RAG's `BM25KeywordSearch`, so both stores are cleared on every replace.
10. **No HTTP endpoint and no scheduler.**
    - The three use cases are driven by a caller on whatever cadence it chooses, the same shape as the existing `SyncCycle`.
    - Durable scheduling belongs to `src/workers/`, which doesn't exist yet.
    - An ingestion endpoint accepting source profiles from clients is an authorization surface that deserves its own review.
11. **A change stays pending until every effect of it has landed.** The final review added this and decision 12, each from a concrete interleaving that the unit tests now reproduce.
    - `IngestDataSource` records a changed hash as `pending_content_hash` before it replaces RAG's copy, evicts the cache, or writes MAG. The save that records the new version clears it.
    - `RefreshCachedSources` and `ReviewSourceFreshness` read a snapshot, do slow work, and then write. Their writes are conditional on the content hash they read and on no change being pending. A change that lands in between makes the write match no row, and the refresh then evicts what it just pre-loaded.
    - A later save updates only the content columns, so it can't overwrite a route or interval that a migration wrote in between.
    - If an effect fails part-way, the marker stays set. The next ingestion re-applies the change, even if the feed has meanwhile reverted to the stored hash.
    - Why: without the marker, an ingestion that evicted the cache and then failed to save left the previous version looking current and confirmed, so the next refresh re-cached text RAG had already superseded. A refresh or review acting on a snapshot could likewise cache or migrate a source that had just changed.
12. **User-scoped data reaches only MAG, unless a caller explicitly opts out.**
    - `IngestDataSource` refuses a new user-scoped source that its `route_policy` places outside `MAG`, unless it was constructed with `permit_user_scope_outside_mag=True`. Only the measurement's baselines set it, because placing a personal fact tenant-wide is the exposure Claim 3 measures.
    - A MAG-routed version is recorded with its hash and time but no text, so a user's personal fact isn't copied outside MAG. Deleting a user deletes their sources and versions.
    - `RecordSemanticFactWriter` stores each source under the fact key `source:<source_key>`, so a routed source can't overwrite a fact MAG learned under the same name.

## Domain (`src/orchestration/domain/`)

**`entities.py` additions:**

- `SourceScope`: `TENANT`, `USER`.
- `IngestionRoute`: `RAG_ONLY`, `CAG_WITH_RAG_BACKUP`, `MAG`.
- `DataSourceProfile(source_key: str, scope: SourceScope, expected_change_interval: timedelta)`. Validation: a non-blank key, and a positive interval.
- `FreshnessPolicy(volatile_below: timedelta = 1 day, demote_after_changes: int = 3, promote_after_quiet_multiple: int = 7, ttl_factor: float | None = 0.5)`. Validation: positive values. `ttl_factor=None` disables expiry, which only the TTL ablation uses.
- `DataSource`: the stored record. It carries `id`, `tenant_id`, `user_id: UUID | None`, `source_key`, `scope`, `expected_change_interval`, `route`, `content_hash`, `last_changed_at`, `last_ingested_at`, `cached_until: datetime | None`, and `pending_hash: str | None` (decision 11). `cached_until` is informational: the cache enforces its own expiry.
- `SourceVersion(content: str | None)`: a changed version saved with its source; `content` is `None` for a MAG-routed source.
- `IngestionResult(source_id, route, changed: bool)`.
- `SourceMigration(source_key, from_route, to_route, observed_interval)`.
- `MigrationDecision(to_route: IngestionRoute, interval: timedelta)`.

**`freshness_router.py`,** the file OVERVIEW.md's module blueprint names:

- `route_for(profile, policy) -> IngestionRoute`: user scope goes to `MAG`; an interval below `policy.volatile_below` goes to `RAG_ONLY`; everything else goes to `CAG_WITH_RAG_BACKUP`.
- `cache_ttl(interval, policy) -> timedelta | None`: `interval × ttl_factor`, or `None` when `ttl_factor` is `None`.
- `observed_change_interval(times: Sequence[datetime]) -> timedelta | None`: the mean gap between consecutive timestamps; `None` with fewer than two.
- `source_id_for(tenant_id, source_key, user_id) -> UUID`: a deterministic id, so a retried ingestion replaces the same RAG document and cache entry.
- `decide_migration(source, version_times, now, policy) -> MigrationDecision | None`: the new route with its re-learned interval, or `None` for no migration. `MAG` sources always return `None`. The demote and promote rules follow decision 7. A demotion's interval is the mean gap between the changes inside the review window (measured from the version before the first of them), and at least one second; a promotion's is the time since the last change.

**`errors.py` additions:**

- `ScopeMismatch`: user scope without a `user_id`, or tenant scope with one.

**`ports.py` additions:**

- `DataSourceRepository`, every method tenant-scoped:
  - `get`, which also takes the optional `user_id`;
  - `save(source, version: SourceVersion | None = None)`, which inserts a new source, or updates only an existing one's content columns (hash, change and ingestion times, `cached_until`) and clears its pending marker, recording `version` in the same transaction;
  - `mark_pending(tenant_id, source_id, content_hash)`;
  - `record_cached_until(tenant_id, source_id, expected_hash, cached_until) -> bool` and `migrate(tenant_id, source_id, expected_hash, route, interval, cached_until) -> bool`, which apply only while the stored hash is `expected_hash` and no change is pending, and report whether they did;
  - `version_times`, `current_content` (the latest version with text), and `list_sources`.
- `RagIndex`: `replace(tenant_id, document_id, title, text)`.
- `ExpiringCache(FrozenCache)`: adds `preload_until(tenant_id, document_id, content, expires_at: datetime | None)` and `renew(tenant_id, document_id, expires_at: datetime | None) -> bool`. `lookup` and `contains` report an expired entry as absent.
- `SessionFactWriter`: `record(tenant_id, user_id, fact_key, fact_value)`.

## Application (`src/orchestration/application/`)

**`IngestDataSource(repository, rag_index, cache, warmed, fact_writer, *, policy, route_policy=route_for, permit_user_scope_outside_mag=False)`.** `execute(tenant_id, profile, content, now, user_id=None) -> IngestionResult` runs these steps:

1. Validate scope against `user_id`.
2. Load the existing source, or create one routed by `route_policy(profile, policy)`. A new tenant-scoped source routed to `MAG` is refused, and so is a new user-scoped source routed elsewhere unless `permit_user_scope_outside_mag` is set (decision 12).
3. Hash the content. It is a change when the hash differs from the stored one. The route's effects are applied again as well when an earlier change is still pending (decision 11).
4. When effects are applied, `repository.mark_pending` records the hash first.
5. Act on the route:
   - `RAG_ONLY`: `rag_index.replace`.
   - `CAG_WITH_RAG_BACKUP`: `rag_index.replace`, then `cache.evict` and `warmed.forget`. On a confirmation with nothing pending, `cache.renew` to the new TTL expiry instead.
   - `MAG`: `fact_writer.record`.
6. Save the source with its hash, `last_changed_at`, `last_ingested_at`, and `cached_until`, which clears the marker. A change also records its version, without text for `MAG`, in the same transaction. A failure part-way leaves the marker set and the stored hash on the previous version, so the next ingestion applies the effects again, and deterministic ids make it replace the same RAG document.

`route_policy` is injectable so the measurement's baselines can place every source on one route through the same code path. This is Batch A's ablation shape: a classifier of `None` gave the unrouted baseline.

**`RefreshCachedSources(repository, cache, warmed, *, policy)`.** `run(tenant_id, now) -> list[str]` handles every `CAG_WITH_RAG_BACKUP` source with no pending change, not currently cached, whose `last_ingested_at + ttl` is still in the future. For each one it:

- reads the current content, and skips the source if that content no longer hashes to the snapshot's `content_hash`;
- pre-loads it with `expires_at = last_ingested_at + ttl`, and notes it warmed;
- records `cached_until` through `record_cached_until`, conditional on the snapshot's hash. If that write applies to no row, a change landed during the pre-load, so it evicts and forgets the entry again.

It returns the keys it pre-loaded.

**`ReviewSourceFreshness(repository, cache, warmed, *, policy)`.** `run(tenant_id, now) -> list[SourceMigration]` applies `decide_migration` to every tenant-scoped source with no pending change:

- Each migration is written through `migrate`, conditional on the snapshot's hash, before the cache is touched. A migration whose write applies to no row is dropped.
- A demotion then evicts and forgets the cache entry, and the write clears `cached_until`.
- A promotion changes only the route and interval; the next refresh pre-loads it.

**`CacheWarmedRetrieve.forget(tenant_id, document_id)`.** It drops the warmed memo. `best_warmed_match` confirms candidates against the cache in score order and returns the first the cache still holds, so an evicted or expired document never shadows a valid one.

## Infrastructure

- **`alembic/versions/0006_data_sources.py`.**
  - `data_sources` holds every column above, with `expected_change_interval` as a positive PostgreSQL `interval`, and `route` and `scope` as `CHECK`-constrained text.
  - A `CHECK` requires `user_id IS NOT NULL` exactly when `scope = 'user'`.
  - Uniqueness is on `(tenant_id, user_id, source_key)` `NULLS NOT DISTINCT` (PostgreSQL 15+; the stack runs 16).
  - `data_sources` also holds `pending_content_hash`, and its `user_id` foreign key is `ON DELETE CASCADE`.
  - `data_source_versions` holds `id`, a `seq` identity column, `data_source_id` (FK, `ON DELETE CASCADE`), `tenant_id`, `content_hash`, a nullable `content`, and `ingested_at`, indexed on `(data_source_id, seq)`.
  - Both tables get `tenant_isolation` RLS and `app_user` grants from creation.
- **`PostgresDataSourceRepository(sessionmaker)`.** Each call sets the tenant context and commits its own short transaction, following `PostgresSessionBudgetRecorder`. `record_cached_until` and `migrate` are single conditional `UPDATE`s that report whether a row matched. Versions are ordered by `ingested_at`, then `seq`.
- **`ExpiringFrozenCache(inner: FrozenCache, clock)`.** It wraps any `FrozenCache`. An entry past its expiry is evicted from the inner cache on first sight and reported absent. The expiry check and eviction happen under one lock, and `preload_until` records the expiry before the slow inner pre-load starts.
- **`ChunkedRagIndex(sessionmaker, vector_store: QdrantVectorStore, chunker, embedder)`.**
  - In one Postgres transaction under RLS, `replace` deletes the document's chunks and row, then saves a fresh `documents` row (filename set to the source key, storage path `data-source:<key>`, status `completed`) and its chunks.
  - In Qdrant, it upserts the new points first, then deletes every other point with the document's `tenant_id` and `document_id` payload, so vector search always finds some version of the document.
- **`RecordSemanticFactWriter(record_semantic_fact)`.** It adapts MAG's existing `RecordSemanticFact` command, unmodified, to `SessionFactWriter`, under the fact key `source:<source_key>`.
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
  - the promoted interval;
  - a burst of changes under one timestamp demoting with a one-second interval.
- **`observed_change_interval`** is tested with too few versions and with unevenly spaced versions.
- **`IngestDataSource`** is tested for:
  - each route's first ingestion;
  - a confirmation versus a change on each route;
  - eviction and forgetting on a change to a cached source;
  - TTL renewal on a confirmation;
  - scope validation, and refusing user scope outside `MAG` unless permitted;
  - an injected `route_policy`;
  - a failed save leaving the change pending, and the retry applying it;
  - a pending change re-applied when the feed reverts to the stored hash;
  - a MAG version recorded without text.
- **`RefreshCachedSources`** is tested for:
  - pre-loading only uncached, confirmed-within-TTL sources;
  - expiry anchored to the last confirmation;
  - skipping `RAG_ONLY`, `MAG`, and pending sources;
  - `ttl_factor=None`;
  - a change landing while the content is read, and while it is pre-loaded.
- **`ReviewSourceFreshness`** is tested for applying demotion (eviction included) and promotion (no pre-load), for leaving user-scoped and pending sources alone, and for dropping a migration when the source changes before the write.
- **`ExpiringFrozenCache`** is tested at expiry, one tick before it, on renewal, on `None` expiry, with a plain `preload`, and with an expiry check interleaved into a re-preload, which must not evict the new entry.
- **`CacheWarmedRetrieve.forget`** is tested to show a forgotten document no longer shadows a valid one, and `best_warmed_match` to show an evicted best candidate doesn't shadow a confirmed runner-up.
- **The fakes** are tested to apply conditional writes only to an unchanged source, so the unit tests above exercise the same contract as the Postgres repository.

Integration tests use real dependencies:

- the migration's constraints and RLS;
- `PostgresDataSourceRepository` round trips and tenant refusal, a later save keeping route and interval, conditional writes blocked by a hash mismatch or a pending change, versions without text, insertion order under one timestamp, and a user delete cascading;
- both `delete_document` methods, including Qdrant keeping the named chunks;
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
