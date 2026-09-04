# Database Schema

This system splits its data across four different stores — PostgreSQL, Qdrant, Redis, and Neo4j — instead of picking one general-purpose database and making everything fit inside it. That split isn't arbitrary: it falls directly out of the three-paradigm design at the heart of this project. RAG needs a store built for similarity search over a large, growing corpus of external documents. CAG's actual cache lives in GPU memory, not on disk, so nothing here backs it directly — but the system still needs somewhere to track which cache entries exist. MAG needs a store for state that's read and written on almost every turn, a store for durable structured facts, and a store for the relationships between those facts, and no single database is good at all three of those jobs at once. What follows walks through each of the four stores, what it holds, and why that particular data ended up there rather than in one of the other three.

## PostgreSQL holds the durable record: users, sessions, and structured memory

PostgreSQL is this project's relational core — the place data goes when it needs foreign-key integrity, transactional writes, and a schema that Alembic migrations can track over time. Eight tables live here, and they split naturally into four groups by what they're for.

The first group is identity and session tracking. `Users` is the root of the tenant model: every row carries a `tenant_id`, and everything else in the system that scopes to a tenant ultimately traces back to this table or to `Sessions`, which represents one conversation and carries its own `context_budget` — the per-session record of how the 128K context window gets sliced between the CAG, MAG, and RAG portions of a given turn.

The second group is MAG's long-term memory — the durable tier behind the working-memory state that lives in Redis (below) — split into three tables because episodic, semantic, and procedural memory are genuinely different kinds of data with different lifecycles. `EpisodicMemory` rows are raw, timestamped experiences: each one is tied to the session it happened in, carries a `salience_score` for how much it matters, and stores its content as JSONB because an experience's shape varies turn to turn. `SemanticMemory` rows are the distilled opposite — a `fact_key`/`fact_value` pair with a `confidence` score and a `valid_until` expiry, because a semantic fact (a user preference, a domain fact) is meant to be looked up directly rather than parsed out of a JSON blob, and because facts go stale in a way raw episodes don't. `ProceduralMemory` sits between the two: a `task_pattern` with a `success_rate` and a JSONB `workflow`, tracking not what happened or what's true, but what worked, so a proven approach can be reused instead of re-derived. `EpisodicMemory` and `SemanticMemory` both carry an `embedding` column (via the `pgvector` extension), which is what lets a query find "memories like this one" with a single SQL query against the row that's already sitting in the durable store, without a round trip to a separate vector service — `ProceduralMemory` deliberately has no `embedding` column, since retrieval against it is a `task_pattern` match (`ProceduralMemoryRepository.find_by_task_pattern`), not a similarity search, so there's no vector for it to store.

The third group is Memory Evolution's own record of what a fact used to say. Memory Evolution is MAG's mechanism for letting a fact be archived, invalidated, or corrected instead of silently going stale, and its record here is `SemanticMemory.archived_at` (added by `alembic/versions/0005_mag_memory_evolution.py`) and the `SemanticMemoryHistory` table it ships alongside. `archived_at` is deliberately distinct from `valid_until` — `valid_until` means a fact is wrong or stale (what `InvalidateMemory` sets), `archived_at` means a fact might still be true but is rarely needed (what `ArchiveMemory` sets) — and a fact can carry either, both, or neither independently. `SemanticMemoryHistory` exists because `SemanticMemory`'s own upsert-by-`(user_id, fact_key)` constraint means the current row is the *only* row for that key: when `UpdateMemory` or `RefineMemory` overwrites `fact_value`, the superseded value has nowhere else to go, so it's archived here with its own `operation` (`update` or `refine` — never `invalidate`, since invalidating flips a status on the same value rather than replacing it, so there's nothing to snapshot) and a `superseded_at` timestamp. `original_fact_id` is a real foreign key back to `SemanticMemory.id` rather than a same-shaped copy, and it stays valid across every future overwrite of the same key because `RecordSemanticFact` derives that id deterministically (a UUID5 of `user_id` and `fact_key`) — `SemanticMemory` rows are never deleted, only updated in place, so the id a history row points back to never goes stale.

The fourth group is the RAG document pipeline: `Documents` tracks an uploaded file's status and chunk count, and `Chunks` holds the actual pieces that get embedded and retrieved. `Chunks.parent_id` is a self-reference — it's what makes parent-document retrieval possible, where a small chunk is what matches a query but a larger surrounding block is what actually gets handed to the model. `Chunks` also carries its own `tenant_id` column rather than scoping to a tenant only indirectly through `document_id`: the migration that created the table (`alembic/versions/0002_documents_chunks.py`) sets `tenant_id` on every chunk and indexes it, specifically so the `tenant_isolation` row-level security policy can evaluate `tenant_id = current_setting('app.current_tenant_id', true)::uuid` directly against the `Chunks` row being read or written, without a join back through `Documents` on every query.

Every one of this schema's eight tables carries an explicit `tenant_id` column, per the tables shown below. That's not an accident of eight separate decisions: `Chunks` was the first place an indirect join (through `document_id` back to `Documents.tenant_id`) was judged not good enough on its own — the RLS policy needs the column directly rather than joining on every query — and the migration that added `EpisodicMemory` and `SemanticMemory` (`alembic/versions/0003_mag_episodic_semantic_memory.py`) followed the same reasoning for both: every tenant-scoped table added from that point on carries `tenant_id` directly and enforces it with a `tenant_isolation` RLS policy, rather than relying on an indirect join (an earlier draft of that migration scoped `SemanticMemory` through `user_id` alone with no RLS at all, on the mistaken premise that `Sessions` does the same — it doesn't, and the schema below reflects the corrected version). `ProceduralMemory` (`alembic/versions/0004_mag_procedural_memory_and_consolidation.py`) and `SemanticMemoryHistory` (`alembic/versions/0005_mag_memory_evolution.py`) both carry `tenant_id` and RLS from their first version rather than needing a follow-up fix the way `SemanticMemory` did — by the time either was built, "every tenant-scoped table gets `tenant_id` and RLS from creation" was already the established rule, not a case-by-case judgment call.

`alembic/versions/0001_users_sessions.py` gives `Users` the same `tenant_id` column and index every other table in this schema carries — but in that same migration, only `Sessions` actually gets `ENABLE ROW LEVEL SECURITY` / `FORCE ROW LEVEL SECURITY` / `CREATE POLICY tenant_isolation`. That makes `Users` tenant-scoped by column but not by policy, the one real exception to "enforces it with RLS" among this schema's eight tables — worth being direct about rather than implying uniform enforcement, and a gap the codebase itself documents rather than hides: `tests/integration/test_rls_tenant_isolation.py` notes in its own setup that inserting two users needs no tenant context of its own, "since `users` carries no RLS." Nothing downstream depends on this gap being closed — every other table's RLS policy reaches `Users` only through a `tenant_id` it already carries directly, never through a query that needs `Users`' own RLS to hold — but it means a compromised connection could still read across tenants' `Users` rows directly, which every other table in this schema is specifically built to prevent.

| Table | Column | Notes |
|---|---|---|
| Users | id | UUID, primary key |
| Users | email | — |
| Users | hashed_password | — |
| Users | tenant_id | tenant-scoping root |
| Users | created_at | — |
| Users | updated_at | — |
| Sessions | id | UUID, primary key |
| Sessions | user_id | foreign key → Users |
| Sessions | tenant_id | — |
| Sessions | title | — |
| Sessions | context_budget | per-session context-slice record |
| Sessions | created_at | — |

| Table | Column | Notes |
|---|---|---|
| EpisodicMemory | id | UUID, primary key |
| EpisodicMemory | session_id | foreign key → Sessions |
| EpisodicMemory | tenant_id | explicit column, indexed, RLS-enforced — same reasoning as `Chunks.tenant_id` below |
| EpisodicMemory | content | JSONB |
| EpisodicMemory | embedding | vector (pgvector) |
| EpisodicMemory | timestamp | — |
| EpisodicMemory | salience_score | — |
| EpisodicMemory | consolidated_at | nullable; set by Consolidation once this episode has been reflected on, excluding it from future consolidation runs regardless of whether that reflection extracted a fact |
| SemanticMemory | id | UUID, primary key |
| SemanticMemory | user_id | foreign key → Users |
| SemanticMemory | tenant_id | explicit column, indexed, RLS-enforced |
| SemanticMemory | fact_key | unique per `(user_id, fact_key)` — `RecordSemanticFact` upserts on this pair rather than appending duplicates |
| SemanticMemory | fact_value | — |
| SemanticMemory | confidence | — |
| SemanticMemory | source | — |
| SemanticMemory | valid_until | expiry for a stale fact |
| SemanticMemory | embedding | vector (pgvector) |
| ProceduralMemory | id | UUID, primary key |
| ProceduralMemory | user_id | foreign key → Users |
| ProceduralMemory | tenant_id | explicit column, indexed, RLS-enforced — carried from this table's first version, not added after review |
| ProceduralMemory | task_pattern | unique per `(user_id, task_pattern)` — `RecordProcedure` upserts on this pair rather than appending duplicates |
| ProceduralMemory | success_rate | — |
| ProceduralMemory | last_used | — |
| ProceduralMemory | workflow | JSONB; no `embedding` column — retrieval is by `task_pattern` match, not similarity search |

| Table | Column | Notes |
|---|---|---|
| SemanticMemory | archived_at | nullable; set by `ArchiveMemory` — distinct from `valid_until`, which means the fact is stale rather than merely rarely needed |
| SemanticMemoryHistory | id | UUID, primary key |
| SemanticMemoryHistory | original_fact_id | foreign key → SemanticMemory.id; stays valid across every future overwrite, since `SemanticMemory` rows are updated in place, never deleted |
| SemanticMemoryHistory | user_id | foreign key → Users |
| SemanticMemoryHistory | tenant_id | explicit column, indexed, RLS-enforced from creation |
| SemanticMemoryHistory | fact_key | the superseded fact's key |
| SemanticMemoryHistory | fact_value | the superseded value, before the overwrite |
| SemanticMemoryHistory | confidence | the superseded value's confidence at the time it was overwritten |
| SemanticMemoryHistory | source | the superseded value's recorded source |
| SemanticMemoryHistory | operation | `update` or `refine` — never `invalidate`, which flips a status on the same value rather than replacing it |
| SemanticMemoryHistory | superseded_at | when the overwrite happened |

| Table | Column | Notes |
|---|---|---|
| Documents | id | UUID, primary key |
| Documents | tenant_id | — |
| Documents | filename | — |
| Documents | mime_type | — |
| Documents | storage_path | — |
| Documents | chunk_count | — |
| Documents | status | — |
| Chunks | id | UUID, primary key |
| Chunks | document_id | foreign key → Documents |
| Chunks | content | — |
| Chunks | embedding | vector (pgvector) |
| Chunks | parent_id | self-reference, enables parent-document retrieval |
| Chunks | metadata | JSONB |
| Chunks | tenant_id | explicit column, indexed — lets `tenant_isolation` evaluate RLS directly, without a join through `document_id` |

Columns without a type noted in the "Notes" column above (`email`, `filename`, `title`, and similar) are conventional scalar fields — text, timestamps — whose exact SQL type isn't pinned down at the documentation level; that's deliberate, since this project's database-first rule puts the Alembic migration, not this document, in charge of the literal `CREATE TABLE` statement. This document records the shape of the schema — which tables exist, which columns they carry, which relationships tie them together — not the migration itself.

## Qdrant holds the vectors that need fast nearest-neighbor search across the whole corpus

Every embedding in this system could, in principle, live only in the `pgvector` columns already sitting in PostgreSQL — the previous section showed three tables that do exactly that. Qdrant exists as a second home for those same embeddings anyway, and the reason isn't raw scale: the ADR that settled this choice (`docs/decisions/adr/0002-qdrant-over-milvus.md`) is explicit that large-scale distributed deployment is the capability being given up by not choosing Milvus, not a capability being gained by choosing Qdrant. What Qdrant wins on instead is a simpler operational model than Milvus's distributed architecture would require, strong performance from its Rust-based core, native hybrid (vector + keyword) search with payload-based metadata filtering — the mechanism that makes tenant-scoped filtering possible at the vector layer — and a local-development experience easy enough to keep this project's Phase 1 foundation work unblocked. `pgvector` earns its place for a different reason entirely: it keeps an embedding transactionally consistent with the row it belongs to, which a separate vector database can't offer. Reading the schema this way suggests the same embedding effectively ends up written twice on purpose — once into PostgreSQL as the durable, relationally-linked record, and again into a Qdrant collection tuned for retrieval — though that duplication is this document's own inference from how the tables and collections line up, not something either CLAUDE.md or the ADR states outright.

Three collections exist, one per source of embeddings: `documents` holds document chunks together with their payload metadata (the RAG corpus), `semantic_memory` holds distilled facts, and `episodic_memory` holds experience embeddings — mirroring the `Chunks`, `SemanticMemory`, and `EpisodicMemory` tables above, but organized for search instead of for transactional integrity.

The index underneath all three collections is HNSW (Hierarchical Navigable Small World) — a graph-based structure where each vector links to a handful of its nearest neighbors, and a search walks that graph greedily toward the query instead of comparing against every vector in the collection. Two parameters control how that graph gets built: `m`, set to 16, is roughly how many neighbor links each vector keeps (more links mean a more richly connected graph and better recall, at the cost of more memory); `ef_construct`, set to 128, is the size of the candidate list considered while inserting each new vector during index construction (a larger list means a more thorough — and slower — build in exchange for a higher-quality graph to search later). Both are standard HNSW tuning knobs, not something specific to this project's data, and they trade index build cost for search accuracy in the usual way.

The other piece that matters here is tenant isolation: Qdrant's metadata filtering is enabled specifically so that a query can be scoped to one tenant's payloads before the nearest-neighbor search even runs, keeping the multi-tenant boundary intact at the vector layer the same way row-level security keeps it intact in PostgreSQL.

| Collection | Contents |
|---|---|
| `documents` | Document chunks, with payload metadata |
| `semantic_memory` | Distilled facts |
| `episodic_memory` | Experience embeddings |

| Index parameter | Value | Effect |
|---|---|---|
| `ef_construct` | 128 | Candidate-list size during index build; higher favors recall over build speed |
| `m` | 16 | Neighbor links per vector; higher favors recall over memory footprint |

## Redis holds whatever needs to be there in microseconds, and nothing that needs to last

Redis is this project's hot path — the store for data that a request needs on essentially every turn and that would be too slow to fetch from a relational database or a graph traversal each time. The concept documents behind this project's architecture describe Redis's role plainly: "Hot session state (working memory)" (`docs/inputs/concepts/fullstack_unified_ai_system.md`), in contrast to PostgreSQL's role as warm, structured memory — and that hot/warm split is exactly why the same conceptual data (a session's live state versus its durable episodic record) shows up differently in each store. Redis doesn't hold a copy of everything in PostgreSQL; it holds the thin, fast-changing slice of state that a request can't afford to wait on.

One key pattern from that original concept is real and built: `session:{session_id}:working_memory` is the current turn's live context — the fastest tier of MAG's memory hierarchy, read and written on nearly every request, implemented by `RedisWorkingMemoryStore` (`src/mag/infrastructure/redis_working_memory_store.py`). The other three items the concept material described for this store — `user:{user_id}:preferences`, a `cag:prefix:{hash}` prefix-cache bookkeeping key, and a `memory:updates` pub/sub channel for cross-service invalidation — were never built: no file under `src/` writes or reads any of those three, and this document previously described them as though they were. What actually replaced that pub/sub design is simpler and lives at the orchestration layer instead of in Redis: `src/orchestration/domain/sync_mixer.py`'s `reconcile()` function compares a content hash the caller already holds against the authoritative source's current hash, synchronously, inside whichever service already has both values — no broadcast, no subscriber, no Redis channel in between. `docs/architecture/OVERVIEW.md`'s "sync mixer" section and `docs/architecture/CAG.md`/`MAG.md`/`RAG.md`'s own build-status notes describe that mechanism directly; this document no longer claims a Redis channel implements it.

Auth Foundation actually built two real tenant-scoped stores on Redis instead, each with its own key prefix — something this document didn't previously mention at all: `RedisRefreshTokenStore` (`src/identity/infrastructure/redis_refresh_token_store.py`) holds refresh-token state so a token can be revoked or rotated without a database round trip, and `RedisRateLimiter` (`src/identity/infrastructure/redis_rate_limiter.py`) tracks per-identity request counts for rate limiting. Both are covered by real integration tests against a TestContainers-provisioned Redis (`tests/integration/test_redis_refresh_token_store.py`, `test_redis_rate_limiter.py`).

| Key pattern | Purpose | Status |
|---|---|---|
| `session:{session_id}:working_memory` | Current turn's live context | Built — `RedisWorkingMemoryStore` |
| Refresh-token store (own key prefix) | Revocable, rotatable refresh tokens | Built — `RedisRefreshTokenStore`, not part of the original concept material |
| Rate-limiter store (own key prefix) | Per-identity request-rate tracking | Built — `RedisRateLimiter`, not part of the original concept material |
| `user:{user_id}:preferences` | Hot user preferences | Not built |
| `cag:prefix:{hash}` | Prefix-cache metadata | Not built |
| `memory:updates` (Pub/Sub) | Cross-service invalidation | Not built — superseded by `sync_mixer.reconcile()`'s synchronous hash comparison |

## Neo4j holds the relationships a table can't express without a wall of joins

The three stores above are all, in their own way, organized around records — a row, a vector, a key. Neo4j exists because some of what MAG needs to represent isn't a record at all, it's a relationship between records, and relational joins get expensive and awkward fast once a query needs to hop across several of them — "which entities are connected to entities mentioned in this user's recent sessions" is a multi-hop question that a graph database answers by walking edges, not by chaining table joins. That's also why "graph traversal" is named directly as one of MAG's retrieval strategies alongside semantic, temporal, causal, and salience-based retrieval — it's a distinct way of finding relevant memory, not a formatting choice for data that could just as easily sit in a table.

Six node types are named in the schema — `User` and `Session` anchor it to the same identities PostgreSQL tracks, `Entity` and `Concept` represent the things a conversation is about, and `Episode` and `Fact` mirror the episodic/semantic split from PostgreSQL — but only five are actually written by any code path today. `Neo4jMemoryGraphRepository`'s own `_NODE_LABELS` constant lists all six, and its uniqueness-constraint setup covers `User`, `Session`, `Entity`, `Episode`, and `Fact`; `Concept` has no constraint and no method that instantiates one anywhere under `src/mag/`. It's schema-defined, matching the same "named but not yet populated" pattern this project uses elsewhere (`docs/decisions/adr/0003-standard-attention-cache-optimization.md`'s technique enum has the same shape), not a node type this document should describe as populated today.

The five edge types are where the graph does its real work, and their names line up closely with operations described elsewhere in this project's memory design — though, as with `Concept`, one of the five is schema-defined rather than actually written. `PARTICIPATED_IN` and `MENTIONS` are the connective tissue tying a `User` or `Session` to the `Episode`s and `Entity`s involved in it, and both are real: `CaptureEpisode` writes them on every captured episode. `ABSTRACTS_TO` is the graph's representation of consolidation — the process that turns a raw episode into a distilled semantic fact is described elsewhere as creating exactly this kind of episode-to-concept abstraction edge (`docs/inputs/concepts/advanced_mag_concepts.md`), and `ConsolidateEpisodes` genuinely writes this edge for each fact it extracts. `TEMPORALLY_FOLLOWS` links episodes in sequence, which is what makes temporal and causal retrieval possible — a query that needs "what happened after X" has to walk this edge rather than sort a table by timestamp, because the graph is what preserves the sequence as a first-class relationship rather than an incidental column value, and `CaptureEpisode` writes it too. `RELATED_TO`, the general-purpose entity-to-entity link, is the one edge type with no writer: `Neo4jMemoryGraphRepository` implements exactly four edge-writing methods (`link_participated_in`, `link_temporally_follows`, `link_mentions`, `link_abstracts_to`) and no fifth for `RELATED_TO` — no command in `src/mag/` currently establishes that a query needs to record two entities as related to each other.

The two indexes — on `Entity.name` and `Entity.embedding` — cover the two ways an entity gets found: by exact or fuzzy name lookup, and by semantic similarity through its embedding, the same nearest-neighbor pattern Qdrant uses, just scoped to entity nodes inside the graph rather than to a standalone vector collection.

| Element | Names |
|---|---|
| Nodes | `User`, `Session`, `Entity`, `Concept` (schema-only, not yet written), `Episode`, `Fact` |
| Edges | `PARTICIPATED_IN`, `MENTIONS`, `RELATED_TO` (schema-only, not yet written), `ABSTRACTS_TO`, `TEMPORALLY_FOLLOWS` |
| Indexes | `Entity.name`, `Entity.embedding` |

## The four stores stay in sync through the orchestration layer, not through the schema itself

A fact written to `SemanticMemory` in PostgreSQL, mirrored into Qdrant's `semantic_memory` collection, and potentially linked into Neo4j via an `ABSTRACTS_TO` edge is three separate writes to three separate systems, not one atomic transaction — the schema above has no mechanism of its own to keep those three writes consistent with each other. That's a deliberate consequence of choosing four specialized stores over one general-purpose one: each store is good at its job specifically because it isn't trying to also be good at the other three jobs. The cost of that specialization is that keeping them coherent is somebody else's responsibility — which is exactly what the orchestration layer's Sync Mixer exists to do. Real, built code answers this now, not a pub/sub design: `src/orchestration/domain/sync_mixer.py`'s `reconcile()` compares a content hash the caller already holds for its own cached copy against the authoritative source's current hash and reports a conflict when they differ, called synchronously from `SyncCycle` (RAG-vs-CAG), `MagSyncCycle` (RAG-vs-MAG), and `CagMagSyncCycle` (CAG-vs-MAG) — three real reconciliation paths, one per pairing, each deciding which side is authoritative for its own case (RAG wins the first two; MAG wins the third, since CAG's entry there is a cached copy of a MAG record rather than an independent fact) and evicting the stale side on a real mismatch. There's no cross-service broadcast the way a Redis pub/sub channel would provide — reconciliation happens inside whichever service already holds both hashes, not through a message every other service has to listen for. `docs/architecture/OVERVIEW.md`'s "sync mixer" section and `evaluation/reports/{rag-cag,rag-mag,cag-mag}-synthesis.md` cover the real, measured detection-to-eviction latency for each pairing. Reading this document alongside that orchestration code is the difference between seeing four unrelated databases and seeing one memory system that happens to be built out of four.
