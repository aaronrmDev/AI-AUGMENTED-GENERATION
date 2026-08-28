# MAG Foundation (Batch A): Memory Hierarchy, Episodic Memory, Semantic Memory — Design Spec

## Goal

Stand up the first working slice of MAG: a hexagonal, CQRS-split `src/mag/` module (per ADR-0004) that can capture episodic memories, record semantic facts, and demonstrate the memory-hierarchy tiering claim (fast working-memory tier vs. durable long-term store) with real, live-measured numbers — not a description of what the architecture would do, but code that does it, backed by real Postgres, Redis, and Qdrant.

This is the first MAG code in the repository. Everything downstream (Batches B–G: Procedural Memory, Consolidation, Retrieval Strategies, Memory Graphs, Memory Gating, Memory Evolution, and the four combinations) builds on the entities, ports, and CQRS shape this batch establishes.

## Scope for this batch

In scope: Memory Hierarchy (#4/#46), Episodic Memory (#7/#47), Semantic Memory (#9/#49).

Deliberately out of scope, deferred to later batches: Procedural Memory (Batch B), Consolidation — turning episodes into facts via an LLM (Batch B), the six advanced Retrieval Strategies beyond plain similarity (Batch C), Memory Graphs / Neo4j (Batch D), Memory Gating (Batch E), Memory Evolution (Batch F). Semantic facts in this batch are recorded directly (a command takes a `fact_key`/`fact_value` pair), not extracted from episodes by an LLM — that extraction is Consolidation's job.

## Architecture

`src/mag/` follows the same three-layer hexagonal shape as `src/rag/` and `src/identity/`, with the CQRS split ADR-0004 requires:

```
src/mag/
  domain/
    entities.py       # EpisodicMemory, SemanticMemory (frozen dataclasses)
    ports.py           # repository + store interfaces (ABCs)
  application/
    commands/
      capture_episode.py       # CaptureEpisode — writes a new episodic memory
      record_working_turn.py   # RecordWorkingTurn — appends to the fast session tier
      record_semantic_fact.py  # RecordSemanticFact — writes/updates a semantic fact
    queries/
      retrieve_episodes.py     # RetrieveEpisodes — plain similarity + session-scoped read
      retrieve_working_memory.py  # RetrieveWorkingMemory — reads the fast tier
      find_semantic_facts.py   # FindSemanticFacts — by key or by similarity
  infrastructure/
    postgres_episodic_memory_repository.py
    postgres_semantic_memory_repository.py
    redis_working_memory_store.py
    qdrant_episodic_memory_index.py   # mirrors QdrantVectorStore's shape, scoped to the episodic_memory collection
    qdrant_semantic_memory_index.py   # same, scoped to semantic_memory collection
```

Commands live separately from queries per ADR-0004's reasoning: writes here are frequent and simple (append an episode, record a fact, push a working-memory turn), reads are what need to combine multiple signals later (Batch C's retrieval strategies plug into the query side without ever touching the command side). Splitting now, even though Batch A's queries are still simple, is what lets Batch C add new query logic without touching how memory gets written.

`src.rag.domain.ports.EmbeddingModel` is reused as-is for MAG's embeddings — it's already the right shape (`embed(text: str) -> list[float]`), and this project's `DATABASE.md` schema uses the same 384-dimension vector space for `EpisodicMemory.embedding` and `SemanticMemory.embedding` that RAG's `Chunks.embedding` already uses (`sentence-transformers`, confirmed by the existing `Vector(384)` column type in `alembic/versions/0002_documents_chunks.py`). No new embedding port is needed.

## Domain entities

```python
@dataclass(frozen=True)
class EpisodicMemory:
    id: uuid.UUID
    session_id: uuid.UUID
    content: dict[str, Any]   # input, reasoning trace, tool_calls, output, outcome, actors, entities
    embedding: list[float]
    timestamp: datetime
    salience_score: float

@dataclass(frozen=True)
class SemanticMemory:
    id: uuid.UUID
    user_id: uuid.UUID
    fact_key: str
    fact_value: str
    confidence: float
    source: str
    valid_until: datetime | None
    embedding: list[float]
```

`EpisodicMemory.content` carries the five required properties `docs/architecture/MAG.md` names for something to count as an episode at all: long-term storage is satisfied by persisting to Postgres (survives the session); explicit reasoning is satisfied by storing the reasoning trace inside `content`, not just the final output, so a later Consolidation pass (Batch B) has something to reflect on; single-shot capture and instance-specific detail are satisfied by `CaptureEpisode` taking the full event once, not accumulating it incrementally; contextual binding is satisfied by `content` carrying actors/entities/timestamp alongside the raw event rather than storing it context-free.

## Ports

```python
class EpisodicMemoryRepository(ABC):
    async def save(self, episode: EpisodicMemory, tenant_id: uuid.UUID) -> None: ...
    async def get_by_session(self, session_id: uuid.UUID, tenant_id: uuid.UUID) -> list[EpisodicMemory]: ...
    async def search_by_similarity(
        self, query_embedding: list[float], tenant_id: uuid.UUID, top_k: int
    ) -> list[EpisodicMemory]: ...

class SemanticMemoryRepository(ABC):
    async def save(self, fact: SemanticMemory) -> None: ...
    async def find_by_key(self, user_id: uuid.UUID, fact_key: str) -> SemanticMemory | None: ...
    async def search_by_similarity(
        self, query_embedding: list[float], user_id: uuid.UUID, top_k: int
    ) -> list[SemanticMemory]: ...

class WorkingMemoryStore(ABC):
    async def push_turn(self, session_id: uuid.UUID, turn: dict[str, Any]) -> None: ...
    async def get_recent_turns(self, session_id: uuid.UUID, limit: int) -> list[dict[str, Any]]: ...
```

`EpisodicMemoryRepository.search_by_similarity` and `SemanticMemoryRepository.search_by_similarity` are plain vector similarity for this batch — a `Retriever`-shaped decorator chain (matching RAG's pattern) is exactly how Batch C adds the six advanced strategies on top of this without changing the port.

## Infrastructure

**Postgres** (`postgres_episodic_memory_repository.py`, `postgres_semantic_memory_repository.py`): raw SQL via `text()`, matching `PostgresUserRepository`'s existing convention — no ORM. Row-level security follows the `tenant_isolation` policy pattern from migration `0002`: `EpisodicMemory` gets an explicit `tenant_id` column (mirroring `Chunks`, since MAG rows are read/written per-request and the RLS policy needs to evaluate directly against the row without a join, same reasoning `DATABASE.md` gives for `Chunks.tenant_id`); `SemanticMemory` scopes through `user_id` → `Users.tenant_id` indirectly, matching how `Sessions` scopes today, since a semantic fact is a per-user long-term record rather than a per-request one.

**Redis** (`redis_working_memory_store.py`): key pattern `session:{session_id}:working_memory`, exactly as `DATABASE.md` specifies — a Redis list (`RPUSH`/`LRANGE`), each entry a JSON-encoded turn, with a TTL refreshed on every push (working memory is explicitly "seconds-minutes" to "hours-days" per the tier table, not permanent). `get_recent_turns` reads the last N entries without touching Postgres at all — this is the fast tier's whole point.

**Qdrant** (`qdrant_episodic_memory_index.py`, `qdrant_semantic_memory_index.py`): two new collections, `episodic_memory` and `semantic_memory`, matching `DATABASE.md`'s table exactly (RAG's `documents` collection is the only one that exists today). Same HNSW parameters as the existing `documents` collection (`m=16`, `ef_construct=128`) — no evidence yet that MAG's access pattern needs different tuning, and matching RAG's proven values is the honest default until a real measurement says otherwise.

Postgres stays the durable, transactionally-consistent record (an embedding written alongside its row via `pgvector`, same as `Chunks`); Qdrant stays the fast nearest-neighbor search path. Both get written on every `save()` — the same "written twice on purpose" pattern `DATABASE.md` already documents for RAG's chunks, extended to MAG's two memory tables.

## Migration

New Alembic migration `0003_mag_episodic_semantic_memory.py`, revises `0002`. Creates `episodic_memory` (id, session_id → sessions.id, tenant_id, content JSONB, embedding vector(384), timestamp, salience_score) and `semantic_memory` (id, user_id → users.id, fact_key, fact_value, confidence, source, valid_until nullable, embedding vector(384)), with RLS enabled on `episodic_memory` (direct `tenant_id` column, same `tenant_isolation` policy as `chunks`) — `semantic_memory` does not get its own RLS policy in this migration, since it scopes through `user_id` and this project hasn't yet established a `user_id`-based RLS pattern anywhere (today's only two RLS tables, `sessions` and `chunks`/`documents`, both key off `tenant_id` directly); adding one now would be inventing a policy shape nothing else in the schema uses yet rather than following an established pattern. `GRANT` matches `app_user`, matching migration `0002`'s convention.

## Evaluation approach for this batch

Unlike RAG's techniques, Memory Hierarchy's actual claim ("hot data belongs in context, warm data in RAM, cold data in a persistent store" — reading recent session state from Redis instead of Postgres is *faster*) is a systems/latency claim, not an answer-quality claim — there's no LLM output to judge qualitatively here, so this batch's evaluation is a live latency benchmark, not a RAG-style baseline/treatment answer comparison:

- **Baseline**: read the last N turns of a session by querying Postgres's `episodic_memory` table directly (no working-memory tier at all — every read is a durable-store round trip, including for state generated seconds ago in the current turn).
- **Treatment**: read the same N turns via `WorkingMemoryStore.get_recent_turns` (Redis).
- **Measured**: real latency (p50/p95) for repeated recent-state reads against a live Postgres + Redis (via `docker/docker-compose.yml`, the same stack RAG's integration tests already use via testcontainers) — this is a genuine infrastructure comparison, honestly scoped to what it actually tests (Redis-vs-Postgres read latency for hot data), not dressed up as an LLM-judged quality comparison it isn't.

Episodic and Semantic Memory get correctness + latency validation, not a quality judge either, for the same reason: this batch is "can the system capture and retrieve memory correctly," not "does having episodic memory make an agent's answers better" — that comparison belongs to the combination batches (G) once retrieval strategies and gating exist to make a full pipeline worth judging end-to-end. A minimal live script captures N real episodes (with real Ollama-embedded content, not fixture vectors) and confirms `search_by_similarity` returns them in a sane order, reporting real p50/p95 write and read latency.

## Testing

Per `docs/testing/TESTING.md`'s rule that MAG specifically needs real-dependency integration tests: unit tests (`tests/unit/`) cover the command/query use cases against fakes (`FakeEpisodicMemoryRepository`, `FakeWorkingMemoryStore`, etc., following the existing `tests/unit/rag_fakes.py` pattern); integration tests (`tests/integration/`) exercise `PostgresEpisodicMemoryRepository`, `PostgresSemanticMemoryRepository`, `RedisWorkingMemoryStore`, and both Qdrant indexes against real `testcontainers`-launched instances, matching how `tests/integration/test_postgres_document_repository.py` and `tests/integration/test_redis_refresh_token_store.py` already do it for RAG and Identity.

## Global constraints carried into the plan

- Database-first: migration `0003` lands before any repository code that depends on it.
- No ORM in repositories — raw SQL via `text()`, matching every existing repository in this codebase.
- RLS on every tenant-scoped table; `episodic_memory` gets the direct-column pattern `chunks` established.
- 384-dimension embeddings throughout, matching the existing `sentence-transformers` model RAG already uses — no new embedding model introduced.
- CQRS: commands and queries in separate modules, never a shared "service" class that does both.
