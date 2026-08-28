# Database Schema

This system splits its data across four different stores — PostgreSQL, Qdrant, Redis, and Neo4j — instead of picking one general-purpose database and making everything fit inside it. That split isn't arbitrary: it falls directly out of the three-paradigm design at the heart of this project. RAG needs a store built for similarity search over a large, growing corpus of external documents. CAG's actual cache lives in GPU memory, not on disk, so nothing here backs it directly — but the system still needs somewhere to track which cache entries exist. MAG needs a store for state that's read and written on almost every turn, a store for durable structured facts, and a store for the relationships between those facts, and no single database is good at all three of those jobs at once. What follows walks through each of the four stores, what it holds, and why that particular data ended up there rather than in one of the other three.

## PostgreSQL holds the durable record: users, sessions, and structured memory

PostgreSQL is this project's relational core — the place data goes when it needs foreign-key integrity, transactional writes, and a schema that Alembic migrations can track over time. Seven tables live here, and they split naturally into three groups by what they're for.

The first group is identity and session tracking. `Users` is the root of the tenant model: every row carries a `tenant_id`, and everything else in the system that scopes to a tenant ultimately traces back to this table or to `Sessions`, which represents one conversation and carries its own `context_budget` — the per-session record of how the 128K context window gets sliced between the CAG, MAG, and RAG portions of a given turn.

The second group is MAG's long-term memory — the durable tier behind the working-memory state that lives in Redis (below) — split into three tables because episodic, semantic, and procedural memory are genuinely different kinds of data with different lifecycles. `EpisodicMemory` rows are raw, timestamped experiences: each one is tied to the session it happened in, carries a `salience_score` for how much it matters, and stores its content as JSONB because an experience's shape varies turn to turn. `SemanticMemory` rows are the distilled opposite — a `fact_key`/`fact_value` pair with a `confidence` score and a `valid_until` expiry, because a semantic fact (a user preference, a domain fact) is meant to be looked up directly rather than parsed out of a JSON blob, and because facts go stale in a way raw episodes don't. `ProceduralMemory` sits between the two: a `task_pattern` with a `success_rate` and a JSONB `workflow`, tracking not what happened or what's true, but what worked, so a proven approach can be reused instead of re-derived. All three memory tables carry an `embedding` column (via the `pgvector` extension), which is what lets a query find "memories like this one" with a single SQL query against the row that's already sitting in the durable store, without a round trip to a separate vector service.

The third group is the RAG document pipeline: `Documents` tracks an uploaded file's status and chunk count, and `Chunks` holds the actual pieces that get embedded and retrieved. `Chunks.parent_id` is a self-reference — it's what makes parent-document retrieval possible, where a small chunk is what matches a query but a larger surrounding block is what actually gets handed to the model. `Chunks` also carries its own `tenant_id` column rather than scoping to a tenant only indirectly through `document_id`: the migration that created the table (`alembic/versions/0002_documents_chunks.py`) sets `tenant_id` on every chunk and indexes it, specifically so the `tenant_isolation` row-level security policy can evaluate `tenant_id = current_setting('app.current_tenant_id', true)::uuid` directly against the `Chunks` row being read or written, without a join back through `Documents` on every query.

`Users`, `Sessions`, `Documents`, `Chunks`, `EpisodicMemory`, and `SemanticMemory` all carry an explicit `tenant_id` column, per the tables shown below — six of this schema's seven tables so far; only `ProceduralMemory` (not yet built) doesn't, since it has no code or migration to carry one yet. That's not an accident of six separate decisions: `Chunks` was the first place an indirect join (through `document_id` back to `Documents.tenant_id`) was judged not good enough on its own — the RLS policy needs the column directly rather than joining on every query — and the migration that added `EpisodicMemory` and `SemanticMemory` (`alembic/versions/0003_mag_episodic_semantic_memory.py`) followed the same reasoning for both: every tenant-scoped table in this schema carries `tenant_id` directly and enforces it with a `tenant_isolation` RLS policy, rather than relying on an indirect join (an earlier draft of that migration scoped `SemanticMemory` through `user_id` alone with no RLS at all, on the mistaken premise that `Sessions` does the same — it doesn't, and the schema below reflects the corrected version).

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
| ProceduralMemory | task_pattern | — |
| ProceduralMemory | success_rate | — |
| ProceduralMemory | last_used | — |
| ProceduralMemory | workflow | JSONB |

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

Three key patterns and one pub/sub channel cover that slice. `session:{session_id}:working_memory` is the current turn's live context — the fastest tier of MAG's memory hierarchy, read and written on nearly every request. `user:{user_id}:preferences` caches hot user settings so they don't require a `Users` table round trip on every request that needs them. `cag:prefix:{hash}` is metadata about the CAG layer's prefix cache — not the KV-cache tensors themselves, which live in GPU memory under vLLM's management, but the bookkeeping that lets the orchestration layer know which prefixes are currently cached before it decides whether a request can take the fast CAG path. And the `memory:updates` pub/sub channel is how a write to memory in one service gets announced to every other service that might be holding a stale copy of it — the mechanism the orchestration layer's cross-paradigm Sync Mixer relies on to keep RAG, CAG, and MAG from drifting out of agreement with each other.

| Key pattern | Purpose |
|---|---|
| `session:{session_id}:working_memory` | Current turn's live context |
| `user:{user_id}:preferences` | Hot user preferences |
| `cag:prefix:{hash}` | Prefix-cache metadata (not the KV tensors themselves) |
| `memory:updates` (Pub/Sub) | Cross-service invalidation |

## Neo4j holds the relationships a table can't express without a wall of joins

The three stores above are all, in their own way, organized around records — a row, a vector, a key. Neo4j exists because some of what MAG needs to represent isn't a record at all, it's a relationship between records, and relational joins get expensive and awkward fast once a query needs to hop across several of them — "which entities are connected to entities mentioned in this user's recent sessions" is a multi-hop question that a graph database answers by walking edges, not by chaining table joins. That's also why "graph traversal" is named directly as one of MAG's retrieval strategies alongside semantic, temporal, causal, and salience-based retrieval — it's a distinct way of finding relevant memory, not a formatting choice for data that could just as easily sit in a table.

Six node types populate the graph: `User` and `Session` anchor it to the same identities PostgreSQL tracks, `Entity` and `Concept` represent the things a conversation is about, and `Episode` and `Fact` mirror the episodic/semantic split from PostgreSQL — but here as graph nodes that can be linked, rather than as isolated rows.

The five edge types are where the graph does its real work, and their names line up closely with operations described elsewhere in this project's memory design. `PARTICIPATED_IN` and `MENTIONS` are the connective tissue tying a `User` or `Session` to the `Episode`s and `Entity`s involved in it. `RELATED_TO` is the general-purpose entity-to-entity link. `ABSTRACTS_TO` is the graph's representation of consolidation — the process that turns a raw episode into a distilled semantic fact is described elsewhere as creating exactly this kind of episode-to-concept abstraction edge (`docs/inputs/concepts/advanced_mag_concepts.md`), so this edge is what consolidation actually writes into the graph. And `TEMPORALLY_FOLLOWS` links episodes in sequence, which is what makes temporal and causal retrieval possible — a query that needs "what happened after X" has to walk this edge rather than sort a table by timestamp, because the graph is what preserves the sequence as a first-class relationship rather than an incidental column value.

The two indexes — on `Entity.name` and `Entity.embedding` — cover the two ways an entity gets found: by exact or fuzzy name lookup, and by semantic similarity through its embedding, the same nearest-neighbor pattern Qdrant uses, just scoped to entity nodes inside the graph rather than to a standalone vector collection.

| Element | Names |
|---|---|
| Nodes | `User`, `Session`, `Entity`, `Concept`, `Episode`, `Fact` |
| Edges | `PARTICIPATED_IN`, `MENTIONS`, `RELATED_TO`, `ABSTRACTS_TO`, `TEMPORALLY_FOLLOWS` |
| Indexes | `Entity.name`, `Entity.embedding` |

## The four stores stay in sync through the orchestration layer, not through the schema itself

Nothing in the schema above enforces consistency between the four stores on its own — a fact written to `SemanticMemory` in PostgreSQL, mirrored into Qdrant's `semantic_memory` collection, and potentially linked into Neo4j via an `ABSTRACTS_TO` edge is three separate writes to three separate systems, not one atomic transaction. That's a deliberate consequence of choosing four specialized stores over one general-purpose one: each store is good at its job specifically because it isn't trying to also be good at the other three jobs. The cost of that specialization is that keeping them coherent is somebody else's responsibility — which is exactly what the orchestration layer's Sync Mixer and the `memory:updates` Redis channel exist to do, propagating a change in one store to the others that hold a related copy of it, rather than relying on the schema to guarantee consistency the way a single database's foreign keys would. Reading this document alongside that orchestration behavior is the difference between seeing four unrelated databases and seeing one memory system that happens to be built out of four.
