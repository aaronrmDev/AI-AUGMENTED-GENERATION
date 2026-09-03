# MAG Batch F: Memory Evolution — Design Spec

**Scope:** #16 (parent), #62 (Update), #63 (Invalidate), #64 (Archive), #66 (Refine).

## What the four child issues actually ask for, versus what the parent narrates

The parent issue (#16) narrates memory evolution as one end-to-end pipeline — detect a new memory that's semantically similar to an existing one, compare the two with an LLM, branch three ways, then propagate the change to linked memories in the graph. Taken completely literally, that narrative doesn't map cleanly onto four independent operations: its "three-way branch" (overwrite / store-both-prefer-recent / store-both-with-context-tags) doesn't line up 1:1 with the four child issues' four operations, and "store both with timestamps" sounds like it's describing something closer to episodic versioning than any single child Task.

The four child issues are each precisely scoped with one trigger and one action, and — matching this project's own established pattern (the parent issue narrates the source concept, the child issues define exactly what to build; see Batch C, D, and E's own design specs) — those four are what this batch actually implements:

| Issue | Operation | Trigger | Action |
|---|---|---|---|
| #62 | Update | New info directly contradicts old info | Overwrite the fact; keep the old value, not delete it |
| #63 | Invalidate | Old info is no longer true at all | Mark the fact stale; exclude it from retrieval |
| #64 | Archive | Fact is rarely accessed | Move to cold storage; keep it available for explicit reference |
| #66 | Refine | New info adds nuance, doesn't contradict | Merge old + new into one richer fact |

All four operate on `SemanticMemory` (Batch A/C's fact storage) — none of the four issues' worked examples touch episodic or procedural memory, and this batch doesn't extend to them. Archive's "rarely accessed" framing could in principle apply to episodic memory too, but scoping it to facts only, alongside its three siblings, keeps this batch's blast radius matched to what all four issues actually describe; episodic memory already has its own lifecycle via Consolidation (Batch B).

## The detection/comparison piece the parent issue asks for

Update, Invalidate, and Refine share a genuine content-comparison shape: given an existing fact and a piece of new information, an LLM judgment decides which of the three applies (or that nothing applies — the new information is about a different context entirely, and belongs as its own fact under a different key rather than touching this one). This batch builds that judgment as `ClassifyFactEvolution`, then composes it with the three operations into one orchestrator, `EvolveMemory` — directly answering the parent issue's "detection, comparison, decision" description, the same way Batch E's `GateMemories` composed its own siblings into the pipeline its parent issue described.

Archive's trigger — access frequency — has nothing to do with comparing new information against old. It's not part of `EvolveMemory`'s dispatch; it stays independently invocable, exactly like Batch E's `TopKSelection` stayed fully built but outside `GateMemories`'s default composition because its own trigger didn't fit that pipeline's shape.

"Detection" (finding an existing fact that's semantically similar to new information) is not a new primitive this batch builds — it's exactly what `PostgresSemanticMemoryRepository.search_by_similarity` (Batch A) already does. A caller runs that search, decides which candidate fact (if any) the new information is about, and hands `EvolveMemory` a specific `fact_key` — this batch does not build automatic "scan the whole store and match" detection, matching the explicit-parameters convention every prior MAG batch has followed (Batch C's retrieval strategies, Batch D's graph writes, Batch E's gating strategies all take an explicit target rather than discovering one themselves).

## A real, pre-existing gap this batch has to close to make Invalidate/Archive mean anything

`PostgresSemanticMemoryRepository.search_by_similarity` and `QdrantSemanticMemoryIndex.search` both currently ignore `valid_until` entirely — a fact past its expiry is still returned by similarity search. This predates this batch (it's Batch A/C infrastructure), but Invalidate's own contract ("exclude it from retrieval") is meaningless if the retrieval path doesn't honor it. This batch fixes both search paths to exclude a fact when either `valid_until` has passed or the new `archived_at` column is set — closing a real, disclosed gap rather than building Invalidate on top of a filter that silently doesn't work.

`QdrantSemanticMemoryIndex.search` currently has zero callers anywhere in this codebase (`PostgresSemanticMemoryRepository.search_by_similarity` is the path Batch C's retrieval strategies and Batch E's `GateMemories` integration test actually use) — but it implements the same `SemanticMemoryIndex` port contract DATABASE.md documents as a live search surface, so leaving it silently inconsistent with the Postgres path would be exactly the kind of undisclosed gap this project's review process has caught before. Both paths get fixed.

## Schema: one new column, one new table

`semantic_memory` gains one column: `archived_at TIMESTAMPTZ NULL` — parallel to the existing `valid_until` column, but semantically distinct (`valid_until` means "this fact is wrong/stale," `archived_at` means "this fact might still be true but is rarely needed"). A fact can be either, both, or neither independently.

A new table, `semantic_memory_history`, holds snapshots of values Update and Refine overwrite — Update's own worked example is explicit that the old value is "archived with its timestamp rather than simply deleted," and `semantic_memory`'s existing upsert-by-`(user_id, fact_key)` unique constraint means the current row is the *only* row for that key, so a superseded value has nowhere else to live. `original_fact_id` is a real foreign key to `semantic_memory.id`, not merely a same-shaped column: `RecordSemanticFact` already derives that id deterministically (`uuid5(NAMESPACE_OID, f"semantic_memory:{user_id}:{fact_key}")`), so it stays stable and valid across every future overwrite of the same key, exactly like a foreign key into a row that's about to be updated but never deleted needs to be.

```sql
CREATE TABLE semantic_memory_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_fact_id UUID NOT NULL REFERENCES semantic_memory(id),
    user_id UUID NOT NULL REFERENCES users(id),
    tenant_id UUID NOT NULL,
    fact_key TEXT NOT NULL,
    fact_value TEXT NOT NULL,      -- the value being superseded
    confidence FLOAT NOT NULL,
    source TEXT NOT NULL,
    operation TEXT NOT NULL,       -- 'update' | 'refine' -- which operation superseded this value
    superseded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Row-level security, same `tenant_id`-scoped policy as every other MAG table.

Invalidate and Archive do **not** write to this history table — neither issue's own description mentions preserving a value (Invalidate flips a status on the *same* value; Archive doesn't touch the value at all), unlike Update and Refine, which both replace `fact_value` outright.

## Domain changes

`SemanticMemory` gains one field:

```python
@dataclass(frozen=True)
class SemanticMemory:
    id: uuid.UUID
    user_id: uuid.UUID
    fact_key: str
    fact_value: str
    embedding: list[float]
    confidence: float = 1.0
    source: str = ""
    valid_until: datetime | None = None
    archived_at: datetime | None = None
```

New entity for a history snapshot:

```python
@dataclass(frozen=True)
class SemanticMemoryHistoryEntry:
    id: uuid.UUID
    original_fact_id: uuid.UUID
    user_id: uuid.UUID
    fact_key: str
    fact_value: str
    confidence: float
    source: str
    operation: str  # "update" | "refine"
    superseded_at: datetime
```

New port method return type for the classification judgment:

```python
@dataclass(frozen=True)
class FactEvolutionClassification:
    operation: str  # "update" | "invalidate" | "refine" | "no_conflict"
    reasoning: str
```

## Port and repository changes

`SemanticMemoryRepository` gains three methods, alongside `search_by_similarity`'s filtering fix:

- `invalidate(user_id, fact_key, tenant_id, invalidated_at) -> None` — targeted `UPDATE ... SET valid_until = :invalidated_at`, not a full re-upsert (no embedding change, no reason to touch Qdrant's vector).
- `archive(user_id, fact_key, tenant_id, archived_at) -> None` — targeted `UPDATE ... SET archived_at = :archived_at`.
- `save_history_entry(entry, tenant_id) -> None` — a plain insert into `semantic_memory_history`.
- `find_history(user_id, fact_key, tenant_id) -> list[SemanticMemoryHistoryEntry]` — ordered newest-first, so a caller (or a test proving Update/Refine actually preserved the old value) can inspect what a fact used to say.

`find_by_key` stays intentionally unfiltered by `valid_until`/`archived_at` — it's a direct, keyed lookup a command needs to fetch the *current* row regardless of status (Update needs to read a fact to overwrite it even if it happens to already be archived), not a "what's currently retrievable" query. Only `search_by_similarity` gets the exclusion filter, since "exclude from retrieval" is specifically about the similarity-search path gating and orchestration actually call.

`SemanticMemoryIndex` (the Qdrant port) gains one method: `update_status(fact_id, tenant_id, valid_until, archived_at) -> None`. `upsert()` always replaces the whole point, including the vector — reusing it from `InvalidateMemory`/`ArchiveMemory` with the embedding-less entity `find_by_key` returns (Qdrant's read path is the embedding-bearing one; Postgres reads always come back with `embedding=[]`, an established convention since Batch A) would silently blank out the stored vector. `update_status` uses Qdrant's `set_payload` instead, which updates named payload fields in place and leaves the vector untouched.

## The five application-layer classes

**`UpdateMemory`** (`src/mag/application/commands/update_memory.py`) — explicit parameters: `tenant_id, user_id, fact_key, new_fact_value, confidence=1.0, source=""`. Fetches the current fact via `find_by_key` (raises `ValueError` if none exists — updating a fact that was never recorded is a caller bug, not a valid degenerate case, matching `RecencyWeightedSampling`'s precedent for rejecting a nonsensical input at the entry point rather than failing cryptically deeper in). Writes a history entry for the current value (`operation="update"`), then delegates the actual overwrite to `RecordSemanticFact` (already handles the Postgres upsert, the Qdrant upsert, and the best-effort Neo4j sync in one place — no reason to duplicate that dance here).

**`RefineMemory`** (`refine_memory.py`) — same shape as Update, but instead of a caller-supplied replacement value, takes `new_information: str` and LLM-merges it with the current `fact_value` into a single richer value (new `_refine_prompt.py`, same retry/validation/fail-safe shape as `capture_episode.py`'s `_score_salience` — three attempts, then falls back to a simple concatenation of old and new rather than silently dropping the new information). Writes a history entry (`operation="refine"`) for the pre-merge value, then delegates to `RecordSemanticFact` with the merged value.

**`InvalidateMemory`** (`invalidate_memory.py`) — `tenant_id, user_id, fact_key, invalidated_at=None` (defaults to now). Fetches the current fact, calls `repository.invalidate(...)`, best-effort syncs both Qdrant (`index.update_status(...)`) and Neo4j (`graph.upsert_fact_node(...)` called with `dataclasses.replace(existing, valid_until=invalidated_at)` — `upsert_fact_node`'s Cypher gains `valid_until`/`archived_at` properties on the `Fact` node so this actually changes something observable in the graph).

**`ArchiveMemory`** (`archive_memory.py`) — same shape as Invalidate, setting `archived_at` instead of `valid_until`.

**`ClassifyFactEvolution`** (a query-shaped judgment, not a command — `src/mag/application/queries/classify_fact_evolution.py`) — `tenant_id, user_id, fact_key, new_information: str`. Fetches the current fact, prompts an LLM to classify the relationship between the existing `fact_value` and `new_information` as one of `update` / `invalidate` / `refine` / `no_conflict`, with reasoning. Same validation shape as `_score_salience`: retry loop, reject a non-string/out-of-enum response, fail safe to `no_conflict` (the least destructive outcome — a caller that gets `no_conflict` back takes no action, which is always safe, unlike guessing `update` and overwriting a fact incorrectly) after exhausting retries.

**`EvolveMemory`** (`evolve_memory.py`) — the orchestrator answering the parent issue's end-to-end description. `tenant_id, user_id, fact_key, new_information`. Runs `ClassifyFactEvolution`, dispatches to `UpdateMemory` (new_fact_value = `new_information`), `RefineMemory` (new_information passed through), or `InvalidateMemory` (no replacement value — matching #63's own "without necessarily replacing it with anything"), or does nothing for `no_conflict`. Returns the classification alongside whatever the dispatched operation returned, so a caller can see *why* a particular action was taken. `ArchiveMemory` is not part of this dispatch, for the reason given above.

## Propagation to the graph

`upsert_fact_node`'s Cypher gains two properties, `valid_until` and `archived_at` (ISO strings or null, same convention `QdrantSemanticMemoryIndex` already uses), so every operation that changes either field and calls it afterward keeps the `Fact` node's observable state in sync with Postgres.

**What this batch does not do:** multi-hop propagation to *other* linked memories in the graph. The parent issue's own language ("pushing the change out to any linked memories") implies walking edges to find related facts/entities and updating them too — but this schema has no edge type representing "these two facts are related" at all (Batch D explicitly scoped `RELATED_TO`, entity-to-entity relatedness, out of its own batch, and nothing built since has added it). Building a new edge type and a graph-walking propagation algorithm neither of the four child issues asks for would be scope invention, not implementation of what's specified. This batch syncs the one node that actually changed; multi-hop propagation is disclosed here as future work, same as Batch D disclosed `RELATED_TO` itself.

## Testing plan

Unit tests (fakes, no real infra) for: `ClassifyFactEvolution`'s validation/retry/fail-safe behavior (mirroring `capture_episode.py`'s existing salience tests); `EvolveMemory`'s dispatch logic for all four classification outcomes; each of the four operations' own explicit-parameter behavior against fake repositories, including the "fact doesn't exist" `ValueError` path for Update/Invalidate/Archive/Refine.

Integration tests (real Postgres, real Qdrant, real Ollama) for: a real Update round-trip (old value lands in `semantic_memory_history`, new value is what `search_by_similarity` now returns); a real Invalidate round-trip proving the fact is genuinely excluded from both `search_by_similarity` and `QdrantSemanticMemoryIndex.search` after invalidation, and still reachable via `find_by_key`; a real Archive round-trip with the same exclude-but-still-findable-by-key shape; a real Refine round-trip against a live Ollama model reproducing MAG.md's own "prefers Python" → "prefers Python, especially for data analysis, though open to Go for CLI tools" example structurally (asserting the merged value's real semantic content covers both the original preference and the new nuance, not exact string matching against a hardcoded model output); a real `EvolveMemory` dispatch test using a live Ollama model to classify a genuine contradiction versus a genuine refinement, reproducing MAG.md's own "New York" → "Berlin" Update example and its "prefers Python" → "prefers Python, especially for..." Refine example through the full classify-then-dispatch path, not just the operations in isolation.
