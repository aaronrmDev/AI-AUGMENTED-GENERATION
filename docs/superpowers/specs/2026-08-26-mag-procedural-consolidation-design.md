# MAG Batch B: Procedural Memory + Consolidation — Design Spec

## Goal

Add the two MAG concepts that turn Batch A's storage foundation into something that learns: Procedural Memory (#17/#51, storing reusable workflows) and Consolidation (#11/#50, the process that turns raw episodes into durable semantic facts). Consolidation composes Batch A's existing `RecordSemanticFact` and `EpisodicMemoryRepository` rather than reimplementing fact-writing — the same "reuse what already exists" pattern this project's RAG combination batches established.

## Lessons applied from Batch A's three review rounds, up front

Batch A's reviews found the same two defect classes twice each: a new table missing `tenant_id`+RLS from the start (with a factually-wrong justification for the gap), and an upsert-by-key operation using a random id instead of a deterministic one, orphaning a stale Qdrant point on re-record. Both are designed out of this batch from the start rather than fixed after review:
- `procedural_memory` (the only new table this batch adds) gets a direct `tenant_id` column and RLS in the same migration that creates it — no separate justification section, because there's no case in this schema anymore where a new tenant-scoped table doesn't get one.
- `RecordProcedure`'s id is `uuid5(user_id, task_pattern)` from the first version, not `uuid4()` — matching `RecordSemanticFact`'s corrected shape, since procedures upsert by `(user_id, task_pattern)` the same way facts upsert by `(user_id, fact_key)`.
- Every new command gets a real integration test constructing it against real Postgres (and Ollama, for Consolidation's LLM call) from the first commit, not added after a review finds the unit-tests-only seam hiding a bug.

## Scope

In scope: Procedural Memory (#17/#51) — capture and retrieve reusable workflows. Consolidation (#11/#50) — reflect on a batch of episodes, extract durable facts, write them to semantic memory, mark the source episodes consolidated.

Deliberately out of scope, deferred to later batches: the six advanced retrieval strategies (Batch C) — Consolidation's episode selection here is oldest-unconsolidated-first, a backlog drain rather than the full recency+relevance+salience+abstraction weighting MAG.md describes as the Generative Agents pattern (see the "Correction, post-review" note under Consolidation's Port changes below for why "oldest-first" and MAG.md's own "the last N turns" framing aren't quite the same claim). Memory Evolution's Archive operation (Batch F) — this batch's "mark or archive the source episodes as consolidated" is implemented as a `consolidated_at` timestamp flag (excludes them from future consolidation runs), not a move to cold storage.

## Procedural Memory

### Entity

```python
@dataclass(frozen=True)
class ProceduralMemory:
    id: uuid.UUID
    user_id: uuid.UUID
    task_pattern: str
    workflow: dict[str, Any]  # JSONB -- steps, tool sequence, whatever the caller wants to record
    success_rate: float = 0.0
    last_used: datetime | None = None
```

No `embedding` field: `docs/database/DATABASE.md`'s `ProceduralMemory` table has no `embedding` column (unlike `EpisodicMemory`/`SemanticMemory`), so retrieval is by `task_pattern` match, not vector similarity — this is a real, intentional schema difference, not an oversight to fix. `docs/database/DATABASE.md` gets a one-line update noting this explicitly once the migration lands, since the schema doc currently doesn't call out why this table alone lacks the column every other memory table has.

### Port

```python
class ProceduralMemoryRepository(ABC):
    async def save(self, procedure: ProceduralMemory, tenant_id: uuid.UUID) -> None: ...
    async def find_by_task_pattern(
        self, user_id: uuid.UUID, task_pattern: str, tenant_id: uuid.UUID
    ) -> ProceduralMemory | None: ...
```

### Application

`RecordProcedure` (command): takes `tenant_id, user_id, task_pattern, workflow, success_rate=0.0`. Builds a `ProceduralMemory` with `id=uuid.uuid5(uuid.NAMESPACE_OID, f"procedural_memory:{user_id}:{task_pattern}")`, calls `save()`. No embedding step, no Qdrant write -- this command is simpler than `RecordSemanticFact` specifically because the entity has nowhere to put a vector.

`FindProcedure` (query): `by_task_pattern(user_id, task_pattern, tenant_id)` -- thin delegate.

### Infrastructure

`PostgresProceduralMemoryRepository`: raw SQL via `text()`, `ON CONFLICT (user_id, task_pattern) DO UPDATE` from the first version (matching `PostgresSemanticMemoryRepository`'s corrected shape, not its first-pass one).

### Migration (0004, part 1)

`procedural_memory`: id, user_id → users.id, tenant_id (direct column, RLS from creation), task_pattern, success_rate, last_used nullable, workflow JSONB. `UNIQUE (user_id, task_pattern)`. RLS enabled + forced + `tenant_isolation` policy, same shape as `episodic_memory`/`semantic_memory`.

## Consolidation

### What it does

Given a session, collect the last N episodes for that session that haven't been consolidated yet, ask the chat model to reflect on them and extract any durable facts, write each extracted fact via the existing `RecordSemanticFact` command, and mark the source episodes as consolidated so a later run doesn't re-process them.

### Entity change

`EpisodicMemory` gains `consolidated_at: datetime | None = None`.

### Port changes

`EpisodicMemoryRepository` gains:
```python
async def get_unconsolidated_by_session(
    self, session_id: uuid.UUID, tenant_id: uuid.UUID, limit: int
) -> list[EpisodicMemory]: ...

async def mark_consolidated(self, episode_ids: list[uuid.UUID], tenant_id: uuid.UUID) -> None: ...
```

> **Correction, post-review:** this section's "collect the last N episodes" and this batch's initial implementation used "last N" and "recency" language loosely. What `get_unconsolidated_by_session` actually does is `ORDER BY timestamp ASC LIMIT :limit` on the unconsolidated set — **oldest**-unconsolidated-first, a backlog drain, not a recency window. In steady state (Consolidation run at least as often as episodes accumulate) the two coincide, since the whole unconsolidated set IS the recent tail — but if a backlog ever exceeds one run's `batch_size`, oldest-first is the behavior that actually drains it; a genuinely recency-first read (`ORDER BY timestamp DESC`) would instead keep reprocessing whatever's newest while older unconsolidated episodes waited forever. The implementation's choice is arguably the more correct one; the spec's wording just didn't say so. `EpisodicMemoryRepository.get_unconsolidated_by_session`'s docstring now states this precisely.

### Application

`ConsolidateEpisodes` (command). Constructor: `episodic_memory_repository, semantic_memory_repository, semantic_memory_index, embedding_model, chat_model` (the last four are exactly `RecordSemanticFact`'s dependencies -- `ConsolidateEpisodes` builds one internally and calls it per extracted fact, rather than re-deriving fact-writing logic).

`execute(tenant_id, user_id, session_id, batch_size=10) -> list[SemanticMemory]`:
1. `episodes = await self._episodes.get_unconsolidated_by_session(session_id, tenant_id, limit=batch_size)`. If empty, return `[]` -- nothing to do is a normal, expected outcome (a session with fewer than `batch_size` new episodes since the last run), not an error.
2. Build a reflection prompt from the episodes' `content` (rendered as numbered entries, timestamp + a JSON dump of content per episode).
3. Call `chat_model.complete()` with a prompt instructing the model to respond with ONLY a JSON object shaped `{"facts": [{"fact_key": str, "fact_value": str, "confidence": float}, ...]}`, empty list if nothing durable is worth extracting.
4. Parse the response. **Same lesson as the just-fixed #149**: `complete()` has no guaranteed JSON mode (unlike `OllamaJudge`'s raw client call, which sets `format="json"`), so retry the completion up to 3 attempts total on a parse failure, matching `OllamaJudge.score()`'s corrected shape. If every attempt still fails to parse, treat it the same as "nothing extracted" (empty facts list) rather than raising -- a failed reflection pass shouldn't crash the caller, but it also shouldn't silently pretend to have consolidated something it didn't. **Correction, post-review: this parenthetical originally said the episodes "stay unconsolidated and get retried on the next run" -- that's wrong, and contradicts step 6 below, which is the accurate description. The implementation marks the episodes consolidated even after retry exhaustion; a parse-failed batch is NOT retried on a later run.** (Per-element validation was also added inside this same retry loop after review found the outer envelope check alone let a malformed individual fact -- not just malformed JSON -- reach `RecordSemanticFact` uncaught; see `_validate_and_dedupe_facts` in `consolidate_episodes.py`.)
5. For each parsed fact, call `RecordSemanticFact.execute(tenant_id, user_id, fact_key, fact_value, confidence=...)`.
6. `await self._episodes.mark_consolidated([e.id for e in episodes], tenant_id)` -- marked regardless of whether any facts were extracted (an episode reflected on and found to contain nothing durable is still consolidated; the alternative -- leaving it eligible for re-reflection forever -- would mean a genuinely fact-free episode gets re-sent to the LLM on every future run).
7. Return the list of written `SemanticMemory` facts.

### Reflection prompt

```
Reflect on the following episodes from a user's conversation history. Extract
any durable facts about the user's preferences, interests, or characteristics
that would be worth remembering long-term -- generalized, timeless facts, not
a record of what happened in this specific conversation.

Episodes (oldest first):
{numbered episode contents}

Respond with ONLY this JSON shape, no other text, no markdown fencing:
{"facts": [{"fact_key": <str>, "fact_value": <str>, "confidence": <float 0-1>}, ...]}
If nothing durable is worth extracting, respond with {"facts": []}.
```

Matches MAG.md's own worked example shape (three Python questions + one Go question consolidating into two facts: primary language, secondary interest) -- the prompt is deliberately built to produce that kind of output, not a verbatim transcript summary.

### Migration (0004, part 2)

`ALTER TABLE episodic_memory ADD COLUMN consolidated_at timestamptz NULL`. No new index: `get_unconsolidated_by_session` filters by `session_id` (already indexed) and `consolidated_at IS NULL`, which is selective enough on a per-session query without a dedicated partial index at this scale -- worth revisiting once a real access pattern says otherwise, not speculatively now.

## Testing

Same discipline as Batch A: unit tests against fakes for every command/query, real testcontainers Postgres integration tests for both new repository methods and the new table, and -- learning from Batch A's actual defect history -- a real integration test constructing `ConsolidateEpisodes` and `RecordProcedure` end to end against real Postgres (and, for Consolidation specifically, real Ollama via `OllamaChatModel`, since that's the one dependency a fake genuinely cannot stand in for: whether a real model's JSON output is actually parseable by the retry logic is an empirical question, not one a fake can answer).

## Evaluation approach for this batch

Procedural Memory: correctness validation (capture, upsert-by-key, retrieve), same shape as Batch A's semantic memory section -- no baseline/treatment comparison, since there's no prior state to compare against.

Consolidation is the first MAG concept in this project with a real LLM-answer-quality angle: does an agent with consolidated semantic memory answer a preference question better than one reading raw episodes directly? That comparison is deliberately **not** built in this batch -- it needs Retrieval Strategies (Batch C) and Gating (Batch E) to make "reading from memory" a fair, fully-built alternative to raw episode access, matching this project's already-stated reasoning for deferring quality comparisons to the combination batches. This batch's live measurement is narrower and honest about it: a real Ollama-backed consolidation run over a real multi-episode corpus, reporting what facts actually got extracted, whether they match what a human reading the same episodes would extract (a qualitative spot-check, not an automated score), and real latency for the reflection call.
