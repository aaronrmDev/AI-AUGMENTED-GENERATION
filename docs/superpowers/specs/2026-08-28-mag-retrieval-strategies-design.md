# MAG Batch C: Retrieval Strategies — Design Spec

GitHub issues covered: #12 (parent), #67 (semantic similarity), #69 (temporal),
#70 (causal), #72 (entity-based), #74 (salience scoring), #75 (recency-decay
fusion).

## Lessons applied from Batches A & B

- **Deterministic upsert IDs, RLS from a table's first migration** — not
  applicable here: this batch adds no tables and no migration. It's read-path
  and one small write-path addition (salience scoring at capture time) on top
  of `episodic_memory`, which already has RLS from migration 0003.
- **Untrusted-LLM-JSON validation pattern** (Batch B's Consolidation review) —
  applies twice in this batch: Causal retrieval's relevance classification and
  the new Salience scorer both call `ChatModel.complete()`, which has no
  forced JSON mode. Both get the same shape: bounded retry (3 attempts,
  matching `_MAX_REFLECTION_ATTEMPTS`/`_MAX_SCORE_ATTEMPTS` precedent),
  markdown-fence stripping, per-field validation *inside* the retry loop (not
  after it), and a safe fallback on exhausted retries rather than a crash —
  score 0.0 / not-causally-relevant, never an exception propagating out of a
  retrieval call.
- **Every review round has found a real issue** — budget for a fix wave on
  this batch same as every prior one; do not skip the three-tier review
  cadence because the design feels well-reasoned going in.
- **Real end-to-end integration tests per command/query, not just fakes** —
  every new query class gets a real-Postgres (+ real-Qdrant for semantic
  similarity, + real-Ollama for causal retrieval and salience scoring)
  integration test, not just a fake-backed unit test. This is the seam that
  hid Batch A's Qdrant-orphan bug.

## The problem this batch resolves (issue I8, deferred from Batch A)

Batch A's first review flagged that all four `search_by_similarity`/`search`
implementations (2 Postgres repos, 2 Qdrant indexes) discard the similarity
score entirely — the ports return bare entity lists. That review explicitly
deferred the fix to "whichever later batch needs multi-strategy score
fusion." This is that batch: #12's "full retrieval pass" requires fusing
scores across strategies, which is impossible without a score to fuse.

### Score-carrying foundation

Two new frozen dataclasses in `src/mag/domain/entities.py`, matching RAG's
existing precedent (`SearchResult` in `src/rag/domain/entities.py`, which
already pairs retrieved content with a `score: float` field) rather than a
bare tuple:

```python
@dataclass(frozen=True)
class ScoredEpisode:
    episode: EpisodicMemory
    score: float

@dataclass(frozen=True)
class ScoredFact:
    fact: SemanticMemory
    score: float
```

RAG's `SearchResult` flattens fields because only content+score are needed at
that layer; MAG's retrieval strategies need the full entity (timestamp,
salience_score, content) for downstream ranking and fusion, so this wraps the
whole entity instead of flattening it. That's an adaptation of the same
precedent, not a contradiction of it.

Four port signatures change to return these instead of bare entity lists:

- `EpisodicMemoryRepository.search_by_similarity` → `list[ScoredEpisode]`
- `SemanticMemoryRepository.search_by_similarity` → `list[ScoredFact]`
- `EpisodicMemoryIndex.search` → `list[ScoredEpisode]`
- `SemanticMemoryIndex.search` → `list[ScoredFact]`

Blast radius (checked before writing this spec): both Postgres
implementations, both Qdrant implementations, two thin application-layer
pass-throughs (`RetrieveEpisodes.by_similarity`,
`FindSemanticFacts.by_similarity` — neither does anything with the return
value beyond returning it, so this is a mechanical type change), the two
fakes in `tests/unit/mag_fakes.py`, and the unit/integration tests for all of
the above. No production wiring exists yet for these query classes (checked:
only test files instantiate `CaptureEpisode`, `RetrieveEpisodes`, etc.), so
there is no DI container to update.

**Score computation, same scale on both backends:**

- Postgres: pgvector's `<=>` operator is cosine *distance*. Select
  `1 - (embedding <=> CAST(:query_embedding AS vector)) AS score` alongside
  the existing columns, so Postgres's score is cosine *similarity* — the same
  quantity and the same scale Qdrant already returns natively for a
  COSINE-distance collection (`hit.score` from qdrant-client is cosine
  similarity directly, confirmed in Batch A's live testing). This matters for
  fusion: it lets scores from the same nominal strategy (semantic similarity)
  compare meaningfully regardless of which backend supplied them, and it
  means the min-max normalization fusion does per-strategy (below) starts
  from genuinely comparable raw values, not two different distance
  conventions mislabeled as the same thing.
- Qdrant: both index `search()` methods already receive `hit.score` from the
  qdrant-client response and simply discarded it before this batch. Plumb it
  through instead of computing anything new.

## The six strategies

Per #12's own worked example ("Why did the deployment fail last Tuesday?"),
a full retrieval pass decomposes a query into semantic intent, temporal
constraints, entities, and causal triggers, then runs strategies in
parallel. This batch implements that decomposition as **explicit structured
parameters the caller supplies** (a query embedding, an optional time
window, an entity string, a causal query string) rather than automatic NLU
that extracts them from free text like "last Tuesday" — turning natural
language into those parameters is a query-understanding capability that
belongs to the orchestration layer described in `OVERVIEW.md` (the layer
that decides which paradigm and which data source answers a query at all),
not to MAG's own retrieval strategies. This is a scope boundary, not an
oversight, and the report for this batch will disclose it as such rather
than claim the worked example's English sentence is parsed automatically.

All five non-fusion strategies scope their candidate set to a single
`session_id` (matching how `EpisodicMemoryRepository.get_by_session` already
scopes episodic reads) — consistent with #70 and #72's worked-example framing
of an ongoing troubleshooting conversation, not a cross-session search.

### #67 — Semantic similarity

`src/mag/application/queries/retrieve_by_semantic_similarity.py`,
`SemanticSimilarityRetrieval.execute(tenant_id, query_embedding, top_k)`.
Thin wrapper over `EpisodicMemoryIndex.search` (Qdrant — the embedding-bearing
real-ANN path per Batch A's established convention, not the Postgres repo).
Returns `list[ScoredEpisode]` directly from the index. This formalizes
existing code as a named strategy now that it carries a real score; the
Batch A/B code was already structurally correct here, just unscored.

### #69 — Temporal retrieval

`src/mag/application/queries/retrieve_by_temporal_window.py`,
`TemporalRetrieval.execute(tenant_id, session_id, top_k, within=None)` where
`within: tuple[datetime, datetime] | None`.

- `within` given: new repository method
  `get_by_session_in_window(session_id, tenant_id, start, end) -> list[EpisodicMemory]`
  (`WHERE timestamp BETWEEN :start AND :end ORDER BY timestamp DESC`). Score
  = `1.0` uniformly — being inside the requested window is a binary match,
  not a graded one.
- `within` omitted: new repository method
  `get_recent_by_session(session_id, tenant_id, limit) -> list[EpisodicMemory]`
  (`ORDER BY timestamp DESC LIMIT :limit`). Score = linear rank decay,
  `(limit - rank) / limit` for `rank` in `0..limit-1`, computed in the query
  class — graded so fusion has something to differentiate, computed in the
  application layer (not SQL) since it's ranking logic, not a data-access
  concern.

### #70 — Causal retrieval

`src/mag/application/queries/retrieve_by_causal_relevance.py`,
`CausalRetrieval.execute(tenant_id, session_id, query, top_k, now=None)`.
Candidates come from the existing `get_by_session` (no new repository method
needed — this is exactly what it already returns). One batched LLM call
(mirroring `ConsolidateEpisodes`'s single-call-over-a-batch shape, not one
call per candidate) scores every candidate's causal relevance to `query` in
one round trip.

New `src/mag/infrastructure/_causal_prompt.py`, mirroring
`_consolidation_prompt.py`'s structure: a system prompt instructing the model
to identify which episodes describe a cause-effect chain relevant to the
query (error traces, root causes, fixes — matching #70's own "prioritizes
episodes that actually contain an error trace" framing) and to score each
0.0–1.0, plus a `build_causal_user_message(query, episodes)` function.
Response shape: `{"scores": [{"episode_index": <int>, "score": <float>}, ...]}`,
validated field-by-field inside the retry loop exactly like Consolidation's
`_validate_and_dedupe_facts`. On exhausted retries: every candidate scores
`0.0` (a real, disclosed degenerate case — "causal retrieval ran but the
LLM's output was unusable" is different from "causal retrieval crashed," and
the pattern established in Batch B is to fail safe, not fail loud, for a
retrieval-time LLM call). Episodes are ranked by score and the top `top_k`
returned, regardless of the 0.0 floor — if every score is genuinely 0.0
(exhausted retries, or the LLM commits to nothing being causally relevant),
the caller gets back `top_k` episodes all scored 0.0 rather than an empty
list, and fusion's min-max normalization (below) handles an all-zero input
band the same way it handles any other flat distribution.

### #72 — Entity-based retrieval

`src/mag/application/queries/retrieve_by_entity.py`,
`EntityRetrieval.execute(tenant_id, session_id, entity, top_k)`. New
repository method
`get_by_session_matching_entity(session_id, tenant_id, entity, top_k) -> list[EpisodicMemory]`.
SQL matches on the `content` JSONB column two ways in one `WHERE`, since
`content`'s documented shape (`entities.py`'s own docstring: "input,
reasoning trace, tool_calls, output, outcome, actors, entities") names an
`entities` key but doesn't guarantee every episode populates it:
`content->'entities' @> to_jsonb(ARRAY[:entity]::text[])` (structured
containment) `OR content::text ILIKE :pattern` (substring fallback across
the whole serialized episode), `ORDER BY timestamp DESC LIMIT :top_k`. Score
= `1.0` uniformly for every match. Considered a graded score (higher
confidence for a structured-field match than a substring-only match) and
rejected it: nothing in this system currently produces a confidence value
for *how* an entity was mentioned, and inventing a number (e.g. "0.7 for
substring matches") to look more precise than the underlying signal actually
is would misrepresent what's being measured. Binary relevance, honestly
labeled, beats fabricated graduation.

### #74 — Salience scoring

`src/mag/application/queries/retrieve_by_salience.py`,
`SalienceRetrieval.execute(tenant_id, session_id, top_k)`. New repository
method `get_by_session_ranked_by_salience(session_id, tenant_id, top_k) ->
list[EpisodicMemory]` (`ORDER BY salience_score DESC LIMIT :top_k`). Score =
the entity's own `salience_score` field directly — it's already a meaningful
continuous signal once populated (see below), nothing to compute on top of
it.

**A real problem found while designing this, not after:** `salience_score`
has existed on `EpisodicMemory` since Batch A's migration but nothing has
ever written a non-default value to it — `CaptureEpisode` (checked before
writing this spec) always saves `salience_score=0.0`. A `SalienceRetrieval`
strategy built on top of that would be real, tested code that always returns
an arbitrarily-tied ordering in practice — correct but hollow, the same
failure shape the "no vacuous test assertions" lesson from Batch B's review
was about, just at the design stage instead of the test stage. This batch
therefore also gives `CaptureEpisode` a real salience-computation step:

- New `src/mag/infrastructure/_salience_prompt.py`: a system prompt asking
  the model to rate, 0.0–1.0, how much this single episode looks like a
  critical decision or a failure/error rather than a routine turn — #74's
  own framing, directly — plus `build_salience_user_message(content)`.
- `CaptureEpisode` gains a `chat_model: ChatModel` constructor dependency
  (matching `ConsolidateEpisodes`'s existing use of the same port) and calls
  it once per captured episode, same 3-attempt retry / fence-stripping /
  field-validation pattern as Causal retrieval above, defaulting to `0.0` on
  exhausted retries (a fail-safe default, not a crash on capture — capture is
  a hot, synchronous path and an unscored episode is still a valid episode).
  The computed value replaces the current hardcoded `salience_score=0.0` at
  construction time.
- This is a real behavior change to existing Batch A code, not new
  standalone code — `CaptureEpisode`'s two existing call sites (both test
  files, no production wiring exists yet) need a `chat_model` fixture added,
  and its existing unit tests need a fake `ChatModel` the same way
  `ConsolidateEpisodes`'s tests already use one.

### #75 — Recency-decay fusion

`src/mag/application/queries/retrieve_with_recency_decay_fusion.py`,
`RecencyDecayFusionRetrieval`. Per #75's own text this "combin[es] the
outputs of the other strategies rather than acting as an independent one" —
so it's built as an orchestrator over the four query classes above (semantic
similarity, temporal, causal, entity) plus salience, not a sixth independent
data-access path.

```python
async def execute(
    self,
    tenant_id: uuid.UUID,
    session_id: uuid.UUID,
    top_k: int,
    query_embedding: list[float] | None = None,
    causal_query: str | None = None,
    entity: str | None = None,
    within: tuple[datetime, datetime] | None = None,
    weights: dict[str, float] | None = None,
    decay_half_life_hours: float = 24.0,
    now: datetime | None = None,
) -> list[ScoredEpisode]:
```

1. Build the set of strategies to run: temporal and salience always run
   (they only need `session_id`); semantic similarity runs only if
   `query_embedding` is given, causal only if `causal_query` is given, entity
   only if `entity` is given. This is the same "explicit parameters, no
   auto-decomposition" boundary as above — fusion runs whatever the caller
   gave it enough information to run, and silently skips what it wasn't
   given, rather than guessing.
2. Run the included strategies concurrently (`asyncio.gather`), each
   returning `top_k`-or-more candidates so fusion has enough to rank across.
3. **Per-strategy min-max normalization**: within each strategy's own result
   set, rescale scores to `[0, 1]` (`(score - min) / (max - min)`; if
   `max == min` — including the causal all-zero-floor case above — every
   score normalizes to `1.0`, since a genuinely flat distribution means every
   candidate was equally (ir)relevant by that strategy's own measure, not
   that they should drop out). This step is why the score-carrying
   foundation above insists Postgres and Qdrant report cosine similarity on
   the same scale for semantic similarity specifically — normalization
   happens *within* a strategy's own output, so cross-backend scale
   consistency for that one strategy still matters for reproducibility
   between the two search paths, even though every strategy gets
   independently normalized before fusion regardless.
4. **Recency decay**, applied to every candidate from every strategy
   uniformly (decay is about the episode's age, not about which strategy
   found it): `decayed = normalized_score * exp(-ln(2) * age_hours /
   decay_half_life_hours)`, standard half-life decay, `age_hours` computed
   against `now` (defaults to `datetime.now(UTC)` if not given — accepting it
   as a parameter, not reading the clock internally, is what makes the decay
   math testable with a fixed reference time rather than a real wall clock).
5. **Combine**: for each unique episode `id` across all strategies' decayed
   results, `fused_score = sum(strategy_weight[s] * decayed_score[s]` for
   every strategy `s` that returned this episode`)`. `strategy_weight`
   defaults to `1 / len(included_strategies)` (equal weighting) unless the
   caller passes `weights`. An episode surfaced by more than one strategy
   accumulates more than one strategy's weighted contribution — that's
   deliberate: agreement across strategies is itself a relevance signal, and
   summing (rather than averaging) rewards it instead of diluting it.
6. **Dedupe and rank**: the combine step in (5) already dedupes by episode
   id as a side effect of summing into one entry per id; sort by
   `fused_score` descending, return the top `top_k` as `list[ScoredEpisode]`.

This directly satisfies #12's "run each strategy in parallel; fuse the
resulting scores with learned or heuristic weights; deduplicate overlapping
hits; and rank the survivors by composite relevance."

## New Postgres repository methods (summary)

Added to `EpisodicMemoryRepository` port and
`PostgresEpisodicMemoryRepository`:

- `get_by_session_in_window(session_id, tenant_id, start, end) -> list[EpisodicMemory]`
- `get_recent_by_session(session_id, tenant_id, limit) -> list[EpisodicMemory]`
- `get_by_session_ranked_by_salience(session_id, tenant_id, top_k) -> list[EpisodicMemory]`
- `get_by_session_matching_entity(session_id, tenant_id, entity, top_k) -> list[EpisodicMemory]`

All follow the existing `_SELECT_COLUMNS` + `_row_to_episode` pattern already
in the file; all raw SQL via `text()` with bound parameters, no ORM, no
string interpolation of untrusted values (the `entity` and window bounds are
bound parameters like everything else in this file).

`FakeEpisodicMemoryRepository` in `tests/unit/mag_fakes.py` gets matching
in-memory implementations of all four, plus updated
`search_by_similarity` returning `list[ScoredEpisode]` (fake cosine
similarity computed the same way the real Postgres/Qdrant backends compute
it, so ordering assertions in unit tests are meaningful, not arbitrary).

## Testing plan

- Unit tests per strategy (fake-backed): correct filtering, correct scoring
  formula, correct ranking, `top_k` respected, tenant/session isolation
  respected (reusing the existing fakes' tenant-scoping behavior).
- Integration tests per strategy against real Postgres (+ real Qdrant for
  semantic similarity) via testcontainers, matching the existing
  `test_postgres_episodic_memory_repository.py` pattern for the four new
  repository methods specifically (real SQL, real ordering, real tenant
  isolation via RLS backstop where a new query path is added).
- **Live-LLM integration tests** (no mocks, matching the Batch B precedent):
  one for `CausalRetrieval` against a real Ollama model with a small
  constructed session containing one clearly-causal episode (an error trace
  followed by a fix) and one clearly-unrelated episode, asserting the causal
  one ranks first; one for the `CaptureEpisode` salience step, asserting a
  constructed "critical failure" episode scores higher than a constructed
  "routine turn" episode. Both report their actual observed scores in the
  evaluation report and the closing issue comments, honestly, including if a
  run's ordering is borderline or the scores are close — matching the
  "expected LLM variance, not a bug" framing Batch B used for Consolidation's
  two differently-worded but both-correct live runs.
- Full existing unit + integration suite re-run after implementation to
  confirm the port signature changes didn't silently break anything outside
  this batch's own new files (RAG's suite is untouched by this batch, but the
  full run is the only way to be sure).

## What this batch does not do

- No automatic natural-language query decomposition (turning "last Tuesday"
  into a `within` window, or free text into an `entity` string) — disclosed
  above as an orchestration-layer concern, not a MAG retrieval-strategy
  concern.
- No cross-session retrieval — all five non-fusion strategies scope to one
  `session_id`, matching the worked example's single ongoing conversation
  framing. A user-wide ("what do I know about User X across all sessions")
  variant of entity-based retrieval is a plausible future extension but isn't
  what #72's own text asks for, and isn't implemented here.
- No learned fusion weights — `weights` in `RecencyDecayFusionRetrieval` is
  caller-supplied heuristic weighting only, matching #12's "learned *or*
  heuristic weights" — a learned-weights mechanism (training a model to pick
  weights from feedback) is out of scope for this batch and has no obvious
  home in the current architecture yet.
