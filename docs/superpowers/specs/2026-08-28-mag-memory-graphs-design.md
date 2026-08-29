# MAG Batch D: Memory Graphs — Design Spec

GitHub issues covered: #13 (parent), #76 (spreading activation).

## Scope, and why it's shaped differently from Batch C

Memory Graphs has only two issues, not seven — but unlike every prior MAG
batch, it needs genuinely new infrastructure. Neo4j is documented
(`docs/database/DATABASE.md` already specifies a full node/edge/index
schema) but not present anywhere in the actual project: no `docker-compose`
service, no Python driver dependency, no testcontainer fixture. Checked
before writing this spec, not assumed. This batch is closer in shape to
Batch A (new storage layer, new infrastructure) than to Batch C (new
application-layer logic over existing storage).

## Lessons applied from Batches A, B, C

- **Every review round has found a real issue** — same three-tier cadence
  (review → fix wave → scoped re-review) applies here, and if anything
  deserves MORE scrutiny than usual: this is the first time this project's
  hexagonal boundary has to hold across four independent stores with no
  shared transaction (Postgres, Qdrant, Redis, now Neo4j), and Batch C's
  worst finding was exactly a scoping bug in a cross-store read path.
- **Real end-to-end tests, not just fakes** — every new repository method
  gets a real-Neo4j integration test via testcontainers, matching the
  existing Postgres/Qdrant/Redis pattern in `tests/integration/conftest.py`.
- **Real LLM measurement where a claim depends on one** — not directly
  applicable to graph writes (mechanical, not judgment calls), but
  spreading activation's relevance-decay behavior gets a real, live
  Neo4j-backed test proving the traversal actually reaches a multi-hop
  answer a single-hop or vector search would miss (this batch's version of
  Batch B/C's live-model measurements).
- **No new ADR was ever written for choosing Neo4j** (checked: `docs/decisions/adr/` has no entry touching graph databases, unlike ADR-0002 for Qdrant). Out of scope to retroactively justify a decision this project already made in `DATABASE.md`/`OVERVIEW.md` — not re-litigated here, just noted as a pre-existing gap this batch doesn't need to close.

## Part 1: Neo4j infrastructure (does not exist yet)

- **`docker/docker-compose.yml`**: new `neo4j` service, `neo4j:5-community`
  image (matching `OVERVIEW.md`'s `neo4j >=5.20.0` driver-version guidance),
  ports `7687` (Bolt) and `7474` (browser, dev convenience only), a named
  volume for data persistence, `NEO4J_AUTH` env var for credentials
  (matching how `postgres`/`redis`/`qdrant` services already set their own
  credentials in this file).
- **`pyproject.toml`**: add the `neo4j` Python driver (`neo4j>=5.20.0`,
  matching `OVERVIEW.md`'s own version floor) to `[project.dependencies]`.
  Uses `neo4j.AsyncGraphDatabase` — the driver's native async API — not a
  sync driver wrapped in a thread pool, matching this project's async-first
  convention everywhere else (SQLAlchemy async, `AsyncQdrantClient`).
- **`tests/integration/conftest.py`**: new `neo4j_container`/`neo4j_url`
  fixtures via `testcontainers.neo4j.Neo4jContainer` (verify this class
  exists in the installed `testcontainers` version before relying on it; if
  it doesn't, fall back to a generic `DockerContainer` wrapper configured
  for the same image, port, and auth env var), matching the existing
  `postgres_container`/`redis_container`/`qdrant_container` pattern exactly
  (module- or session-scoped per whatever the existing fixtures already do
  — read them first, match them, don't invent a new scoping convention).

## Part 2: Schema setup

`docs/database/DATABASE.md` already specifies the complete schema — this
batch implements it, not designs it:

| Element | Names |
|---|---|
| Nodes | `User`, `Session`, `Entity`, `Concept`, `Episode`, `Fact` |
| Edges | `PARTICIPATED_IN`, `MENTIONS`, `RELATED_TO`, `ABSTRACTS_TO`, `TEMPORALLY_FOLLOWS` |
| Indexes | `Entity.name`, `Entity.embedding` |

Per-node properties aren't specified beyond the two indexed `Entity`
fields, so this batch defines them by cross-referencing the existing
Postgres schema each node type mirrors, keeping field names identical
where a direct correspondence exists (no gratuitous renaming across
stores):

- `User(id)`, `Session(id)` — bare identity anchors, matching how
  `DATABASE.md` describes them ("anchor it to the same identities
  PostgreSQL tracks"). No other properties: these nodes exist so edges have
  somewhere to attach, not to duplicate `users`/`sessions` table data.
- `Episode(id, content, timestamp)` — mirrors `EpisodicMemory`'s Postgres
  columns. No `embedding` property on the node itself: `Entity.embedding`
  is the only embedding-indexed node type per `DATABASE.md`'s own index
  list, and Qdrant remains this system's embedding-bearing store for
  episodes (established convention since Batch A) — duplicating the vector
  into a third store with no index on it would be dead weight, not a
  feature.
- `Fact(id, fact_key, fact_value, confidence)` — mirrors `SemanticMemory`'s
  Postgres columns (minus `embedding`, same reasoning as `Episode` above).
- `Entity(name, embedding)` — the two indexed properties `DATABASE.md`
  names explicitly.
- `Concept(name)` — `DATABASE.md` groups `Entity`/`Concept` together as
  "the things a conversation is about" but only names `Entity.name`/
  `Entity.embedding` as indexed; `Concept` gets a `name` property with no
  index, since nothing in `DATABASE.md` or `MAG.md` asks for concept
  similarity search the way entity lookup needs one.

A migration-equivalent for Neo4j: this project's `alembic/` migrations are
Postgres-specific, so schema setup here is a small
`ensure_schema()`-style method on the concrete repository (mirroring how
`QdrantEpisodicMemoryIndex.ensure_collection()` and
`QdrantSemanticMemoryIndex.ensure_collection()` already handle
provisioning for their own store, deliberately NOT part of the abstract
port — same established convention, not a new one) issuing `CREATE
CONSTRAINT`/`CREATE INDEX IF NOT EXISTS` Cypher statements for uniqueness
constraints on each node type's id/name and the two `Entity` indexes.

## Part 3: The write path — building the graph

MAG.md's own framing: "Building the graph means constructing a node for
each new memory with its full attributes, then finding related existing
memories and establishing edges to them as each new memory arrives." This
batch wires graph writes into the SAME commands that already write to
Postgres/Qdrant, rather than a separate batch-sync job — matching
`DATABASE.md`'s own description of how the four stores stay consistent
("three separate writes to three separate systems, not one atomic
transaction... through the orchestration layer, not through the schema
itself").

New port `MemoryGraphRepository` in `src/mag/domain/ports.py`:

```python
class MemoryGraphRepository(ABC):
    async def upsert_episode_node(
        self, episode: EpisodicMemory, tenant_id: uuid.UUID
    ) -> None: ...
    async def upsert_fact_node(
        self, fact: SemanticMemory, tenant_id: uuid.UUID
    ) -> None: ...
    async def link_participated_in(
        self, user_id: uuid.UUID, session_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None: ...
    async def link_temporally_follows(
        self, earlier_episode_id: uuid.UUID, later_episode_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None: ...
    async def link_mentions(
        self, episode_id: uuid.UUID, entity_name: str, tenant_id: uuid.UUID
    ) -> None: ...
    async def link_abstracts_to(
        self, episode_id: uuid.UUID, fact_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None: ...
```

`RELATED_TO` (entity-to-entity) is deliberately NOT written by this batch
— every other edge type has an obvious, single writer (an episode being
captured, a fact being consolidated), but "which entities are related to
which" needs its own inference step (co-occurrence within a session? an
LLM judgment, like `CausalRetrieval`'s classifier?) that no existing
command naturally produces as a side effect, and inventing one is scope
this batch's two issues don't ask for. Documented as deferred, not
silently dropped.

Every Neo4j write is tenant-scoped by a `tenant_id` property on every node
(Neo4j has no native row-level-security equivalent to Postgres's RLS, so
this is an application-level filter on every write AND every read query —
consistent with how the write path is already "three separate writes, no
shared transaction," so it's honest to the store's own consistency model
rather than pretending Neo4j inherits Postgres's guarantees). Every read
query in Part 4 filters on `tenant_id` explicitly, and every new
integration test proves tenant isolation the same way this project already
tests it — including a same-tenant-different-session isolation test, per
the lesson Batch C's own review had to teach through a real finding.

**Wiring into existing commands** (each addition is one or two extra calls
at the end of an already-tested command, not a rewrite):

- `CaptureEpisode` (`src/mag/application/commands/capture_episode.py`):
  after saving to Postgres/Qdrant, upserts an `Episode` node, links
  `PARTICIPATED_IN` from the session, links `TEMPORALLY_FOLLOWS` from the
  session's previous episode (if one exists — the first episode in a
  session has nothing to link from), and links `MENTIONS` for every string
  in `content.get("entities", [])` (the same field Batch C's
  `EntityRetrieval` already reads structurally — reusing it here rather
  than inventing a second entity-extraction mechanism). Gains a
  `memory_graph_repository: MemoryGraphRepository` constructor dependency.
- `RecordSemanticFact` (`src/mag/application/commands/record_semantic_fact.py`):
  after saving to Postgres/Qdrant, upserts a `Fact` node. Gains the same
  new dependency.
- `ConsolidateEpisodes` (`src/mag/application/commands/consolidate_episodes.py`):
  after `RecordSemanticFact` writes each extracted fact, links
  `ABSTRACTS_TO` from every episode in the consolidated batch to that
  fact's node — this IS the graph's representation of consolidation per
  `DATABASE.md`'s own description ("the process that turns a raw episode
  into a distilled semantic fact... creating exactly this kind of
  episode-to-concept abstraction edge"). Gains the same new dependency,
  passed through to its internal `RecordSemanticFact` instance.

A graph write failing must not roll back or block the Postgres/Qdrant
writes that already succeeded — matching the "no shared transaction, three
separate writes" model this project has already committed to. Each graph
write call is wrapped so a Neo4j-side failure logs and continues rather
than raising past the command's `execute()`; this is a real, disclosed
consistency trade-off (the graph can fall behind the source-of-truth
stores), not an oversight — this project has no distributed-transaction or
outbox mechanism yet, and building one is far outside this batch's two
issues.

## Part 4: Spreading activation retrieval (#76)

`SpreadingActivationRetrieval` in
`src/mag/application/queries/retrieve_by_spreading_activation.py`:

```python
async def execute(
    self,
    tenant_id: uuid.UUID,
    start_entity_names: list[str],
    max_hops: int = 3,
    decay_factor: float = 0.5,
    activation_threshold: float = 0.05,
) -> list[ActivatedNode]
```

Per #76's own description: "start from the node (or nodes) that match the
query, activate them, and propagate a relevance score outward along their
edges to connected nodes... until relevance decays past whatever threshold
the retrieval step sets." Implemented as a single Cypher query using
Neo4j's variable-length path traversal (`MATCH path = (start)-[*1..max_hops]-(end)`)
rather than hand-rolled multi-round-trip breadth-first search in Python —
the whole reason this project chose a native graph database over
relational joins (`DATABASE.md`'s own justification) is that traversal is
cheap for Neo4j to run natively; re-implementing BFS with N round trips
from the application layer would throw that reason away.

Activation score for a node reached via a path of length `k` from a start
node: `decay_factor ** k` (each hop multiplies relevance by the decay
factor, so activation decays geometrically with distance — matching "the
most-activated subgraph... returned as context," not just the nearest
node). A node reachable via multiple paths (from multiple start entities,
or by multiple routes to the same node) takes the MAX of its path
activations, not a sum — spreading activation models "how strongly is this
node connected to what I'm looking for," and summing would let a node with
many weak, redundant paths outscore a node with one strong, direct path,
which inverts what activation is supposed to measure. Nodes below
`activation_threshold` are excluded from the result.

New entity `ActivatedNode` in `src/mag/domain/entities.py`:

```python
@dataclass(frozen=True)
class ActivatedNode:
    node_id: str
    node_type: str  # "Episode" | "Fact" | "Entity" | "Concept" | "User" | "Session"
    properties: dict[str, Any]
    activation: float
    hops: int
```

`node_type`/`properties` are deliberately generic (not a typed union of
five different dataclasses) because spreading activation's whole point is
traversing across heterogeneous node types in one pass — the caller gets
back whatever the graph actually reached, not a pre-filtered single type,
matching MAG.md's own worked example reaching through `User` → `Entity`
(Paris) → `Entity` (vegan food) → `Fact` (the restaurant recommendation) in
one traversal.

**Start-node resolution**: `start_entity_names` is caller-supplied
(matching this batch's siblings' "explicit parameters, no automatic query
decomposition" convention established in Batch C) — the caller already
knows which entity/entities to start from (e.g., "User X" from a
structured query), not asking this method to parse free text into a
starting node.

## Testing plan

- Unit tests for `SpreadingActivationRetrieval`'s activation-score math
  (decay per hop, max-not-sum on multi-path nodes, threshold cutoff) against
  a fake graph repository.
- Integration tests (real Neo4j via testcontainers) for every
  `MemoryGraphRepository` write method: node upserts (idempotent — upserting
  the same episode twice doesn't create a duplicate node, matching this
  project's established upsert-by-key discipline), every edge type,
  tenant isolation, and same-tenant-different-session isolation (applying
  Batch C's own review lesson from the start this time, not after a finding
  catches its absence).
- Integration tests for the three extended commands (`CaptureEpisode`,
  `RecordSemanticFact`, `ConsolidateEpisodes`) proving each one's graph
  side-effect actually lands in real Neo4j alongside its existing
  Postgres/Qdrant writes.
- A live, real-Neo4j integration test for spreading activation
  reproducing MAG.md's own worked example structurally: seed a small graph
  (`User` → `PARTICIPATED_IN` → `Episode` → `MENTIONS` → `Entity` "Paris"
  → ... → `Fact` "recommended Le Potager du Marais"), run spreading
  activation from the `User` node, and assert the multi-hop `Fact` node is
  reached with the expected decayed activation — proving multi-hop
  traversal succeeds where a single-hop or keyword match would fail,
  matching #76's own restaurant-recommendation example directly.

## What this batch does not do

- **No `RELATED_TO` edge writer** — deferred, see Part 3, no existing
  command naturally produces entity-to-entity relatedness as a side effect.
- **No `Concept` node writer** — Part 2's schema table names `Concept`
  alongside the five node types this batch does write, but no command
  produces one: nothing in this batch's scope (episode capture, fact
  recording, consolidation) naturally distinguishes "an abstract concept"
  from "an entity" as a separate write target the way `MENTIONS`/
  `ABSTRACTS_TO` already cover entities and facts. `ensure_schema()` also
  creates no constraint for it. A review caught this batch's own scope
  list not disclosing it — corrected here rather than left implicit.
- **No graph-write transactional consistency with Postgres/Qdrant** — a
  Neo4j write failure is logged and swallowed, not retried or reconciled;
  this project has no outbox/saga mechanism, and building one is out of
  scope for two issues about building and querying a graph.
- **No natural-language query parsing into `start_entity_names`** — same
  explicit-parameters boundary Batch C established for its own five
  strategies.
- **No ADR for the Neo4j-over-alternatives decision** — that decision
  predates this batch (`DATABASE.md`/`OVERVIEW.md`) and retroactively
  justifying it isn't what #13/#76 ask for.
