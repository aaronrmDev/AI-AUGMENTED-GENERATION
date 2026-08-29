# MAG Batch E: Memory Gating — Design Spec

GitHub issues covered: #15 (parent), #53 (Top-K selection), #54 (Token
budget allocation), #55 (Hierarchical assembly), #57 (Recency-weighted
sampling), #58 (Task-specific filtering), #60 (Dynamic re-ranking).

## Scope, and why it's shaped differently from Batches C and D

Six children under one parent, exactly mirroring Batch C's structure (one
issue per named strategy). Unlike Batch C (new application logic over
existing storage) or Batch D (new storage entirely), gating needs **no new
infrastructure and no new database or schema at all** — checked before
writing this spec, not assumed: `docs/database/DATABASE.md` has nothing
provisioned for gating, and MAG.md itself frames gating as a pure
compute-time filter: "retrieval decides what candidate memories get pulled
out of storage... gating decides what survives from those candidates into
the context window." Every input gating needs already exists on
`ScoredEpisode`, `ScoredFact`, and `ActivatedNode` (Batches C and D) — this
batch is pure application-layer logic, fully unit-testable, no
testcontainers required for the strategies themselves.

## Lessons applied from Batches A–D

- **Every review round has found a real issue** — same three-tier cadence
  applies here regardless of how "simple" pure functions might look going
  in.
- **Real data, not synthetic toy inputs, wherever a live measurement is
  possible** — this batch has no LLM calls to live-measure, but it does
  have a real embedding model (`DynamicReranking`) and real token counting
  (`tiktoken`, already a project dependency). The integration test for the
  full pipeline runs real Batch C/D retrieval outputs (real Postgres,
  real Qdrant embeddings) through gating, not hand-built fixtures alone.
- **Don't silently degrade** — `DynamicReranking` can't compute a
  meaningful re-score for a candidate with no embedding (Postgres-sourced
  reads carry `embedding=[]` per the established convention); this is
  disclosed explicitly as a documented fallback (keep the original score),
  not silently wrong output.
- **Reuse an established pattern rather than reinvent it** — recency
  weighting uses the same half-life decay formula
  `RecencyDecayFusionRetrieval` (Batch C) already established for the same
  purpose, applied here to a different concern (gating a mixed candidate
  pool, not fusing retrieval-strategy scores).

## The unified candidate type

MAG.md's own worked example treats facts and episodes as one pool being
gated together: "places user preferences and task-critical facts first,
adds supporting episodic context after" — one selection-and-ordering pass
over a mixed set, not three separate per-type passes. `ScoredEpisode`,
`ScoredFact`, and `ActivatedNode` are three different dataclasses with
different shapes, so gating needs a normalized wrapper the same way
`ActivatedNode` already normalizes across six heterogeneous *node* types
for spreading activation — this batch applies that same precedent one
level up, across heterogeneous *retrieval strategies*.

New entity in `src/mag/domain/entities.py`:

```python
@dataclass(frozen=True)
class GatingCandidate:
    content_text: str          # what token-counting strategies measure
    score: float                # composite relevance, strategy-dependent scale
    salience: float              # 0.0 if the source has no salience signal
    timestamp: datetime | None  # None for graph nodes with no timestamp
    source_type: str            # "episode" | "fact" | "graph_node"
    origin: EpisodicMemory | SemanticMemory | ActivatedNode
    embedding: list[float]      # [] if the source carries none (Postgres reads)
```

`content_text`/`embedding` are pulled from `origin` by the (new) adapter
functions below, not duplicated data — `origin` stays the single source of
truth, `content_text`/`embedding`/`salience`/`timestamp` are a flattened,
uniform read of it so gating strategies never need `isinstance` branching
on three different types internally.

`src/mag/application/gating/_candidates.py` (new small conversion module,
not a class — three pure functions, one per source type):

```python
def from_scored_episode(scored: ScoredEpisode) -> GatingCandidate: ...
def from_scored_fact(scored: ScoredFact) -> GatingCandidate: ...
def from_activated_node(node: ActivatedNode) -> GatingCandidate: ...
```

- `from_scored_episode`: `content_text = json.dumps(episode.content,
  sort_keys=True)` (matching `CaptureEpisode`'s own embedding-input
  convention for the same field), `salience = episode.salience_score`,
  `timestamp = episode.timestamp`, `source_type = "episode"`.
- `from_scored_fact`: `content_text = fact.fact_value`, `salience =
  fact.confidence` (the closest analogous secondary signal a
  `SemanticMemory` carries — disclosed as a deliberate substitution, not an
  oversight: facts have no `salience_score` field, and confidence plays the
  same "how much should this be trusted/weighted" role), `timestamp = None`
  (`SemanticMemory` has no timestamp field), `source_type = "fact"`.
- `from_activated_node`: `content_text = json.dumps(node.properties,
  sort_keys=True)`, `salience = 0.0` (a graph node carries no salience
  signal of its own), `timestamp = None` unless `node.properties` happens
  to carry one (an `Episode`-typed activated node's properties do, per
  `Neo4jMemoryGraphRepository._node_properties`'s existing deserialization
  — extracted opportunistically when present, not required), `source_type
  = "graph_node"`.

## Where token counting lives

`tiktoken` is already a dependency, already used once (`CompressingRetriever`,
RAG), with no shared abstraction — this is the second independent need for
the identical `tiktoken.get_encoding("cl100k_base")` pattern, which is
where this project's own token-economy guidance says repetition becomes
the signal to extract, not duplicate again. New minimal module, the first
thing under a `src/shared/` package (nothing currently needs one; this is
the first genuinely cross-paradigm utility neither RAG nor MAG owns
exclusively):

```python
# src/shared/tokenization.py
def count_tokens(text: str) -> int: ...
```

One function, no class, no configuration surface beyond the encoding
`CompressingRetriever` already committed to (`cl100k_base`) — matching it
exactly rather than introducing a second convention. `CompressingRetriever`
itself is NOT refactored to use this in this batch (it already works,
touching it isn't what these seven issues ask for); only MAG's new gating
code uses the shared function.

## The six strategies

Every strategy lives in `src/mag/application/gating/`, one file each,
matching this project's established one-class-per-file convention for
application-layer use cases. Every strategy has the same shape —
`execute(candidates: list[GatingCandidate], ...) -> list[GatingCandidate]`
— specifically so a pipeline can chain them (Part on `GateMemories` below),
and every strategy is a pure function with no injected dependencies (no
repository, no embedding model) — the caller supplies whatever a strategy
needs (a token budget, a query embedding, an allowed-type set) as explicit
parameters, continuing this batch's own "explicit parameters, no
auto-decomposition" convention from Batch C.

### #53 — Top-K selection

`TopKSelection.execute(candidates, k) -> list[GatingCandidate]`. Sorts by
`score` descending, returns the first `k`. "The simplest and fastest
gating option... the right default when speed matters more than nuance and
the scoring function is already trustworthy" — no additional logic beyond
that sort and slice.

### #54 — Token budget allocation

`TokenBudgetAllocation.execute(candidates, token_budget) ->
list[GatingCandidate]`. Sorts by `score` descending, then greedily
accumulates candidates (via `src/shared/tokenization.count_tokens`) until
adding the next one would exceed `token_budget`, skipping (not stopping
at) any single candidate that doesn't fit — the same greedy-by-score,
skip-oversized-items pattern `CompressingRetriever` already established for
an analogous problem, applied here to whole candidates instead of
sentences. "Maximizes information density rather than memory count... a
better fit than Top-K when memories vary wildly in length."

### #55 — Hierarchical assembly

`HierarchicalAssembly.execute(candidates) -> list[GatingCandidate]`. Not a
selection strategy — MAG.md's own framing is explicit: "the strategy to
reach for when what's included is already right but the order isn't
protecting the critical facts," i.e. this reorders an already-selected
list, it doesn't shrink one. Stable-sorts by `(source_type priority,
score)` descending, where `source_type` priority is `fact > episode >
graph_node` — matching the worked example's own ordering ("places user
preferences and task-critical facts first, adds supporting episodic
context after") directly: facts (which is where "user preferences" live in
this schema, per `SemanticMemory`) are ranked ahead of episodes, addressing
"lost in the middle" by putting the highest-priority *type* at the front
regardless of how the pool was originally interleaved.

### #57 — Recency-weighted sampling

`RecencyWeightedSampling.execute(candidates, half_life_hours, now=None) ->
list[GatingCandidate]`. Reuses `RecencyDecayFusionRetrieval`'s established
half-life decay formula (`score * exp(-ln(2) * age_hours /
half_life_hours)`) — same math, different purpose: there it fuses
retrieval-strategy scores, here it re-weights gating candidates so a
strategy "biased toward recent memories while still making room for
old-but-salient ones" doesn't discard genuinely important old material
outright. Candidates with `timestamp = None` (facts, graph nodes) are
**not decayed** — their score passes through unchanged, since there's no
age to compute and MAG.md's own framing ("old-but-salient ones") is about
not over-penalizing age, which a timestamp-less candidate has none of to
begin with. Returns candidates re-scored and re-sorted descending by the
decayed score, not a subset — "sampling" in MAG.md's name refers to the
re-weighting shifting which candidates end up on top when a later
selection strategy (Top-K/token-budget) narrows the list, not a selection
step of its own.

### #58 — Task-specific filtering

`TaskSpecificFiltering.execute(candidates, allowed_source_types) ->
list[GatingCandidate]`. Filters to candidates whose `source_type` is in
the caller-supplied set. Scoped to the three `source_type` values this
batch's `GatingCandidate` actually carries (`"episode"`, `"fact"`,
`"graph_node"`) — MAG.md's own illustrative example names "procedural
memories" as a type to exclude, but no procedural-memory retrieval
strategy in this codebase currently returns a scored list the way
episodic/semantic/graph retrieval do (`ProceduralMemoryRepository.
find_by_task_pattern` is a single exact-match lookup, not part of the
scored-candidate pipeline gating sits downstream of) — filtering by a
caller-supplied set of `source_type` strings extends to a fourth type
later with no redesign, but this batch doesn't invent one that has nothing
upstream producing it yet.

### #60 — Dynamic re-ranking

`DynamicReranking.execute(candidates, query_embedding) ->
list[GatingCandidate]`. Re-scores every candidate by real cosine similarity
between `query_embedding` and the candidate's own `embedding` field,
**replacing** (not blending with) the original score — "re-ranks the
retrieved set based on the specifics of the current query rather than
trusting the original retrieval score" is explicit that this supersedes
the prior score, not averages with it. `query_embedding` is a required
explicit parameter, not computed here from a raw query string — the caller
already has one from whatever embedded the query for the original
retrieval call, and recomputing it here would be redundant work and a
needless `EmbeddingModel` dependency this strategy would otherwise be the
only one of the six to need. A candidate with `embedding == []` (a
Postgres-sourced episode/fact, or a graph node — none of which carry a
real vector under this project's established embedding-bearing-index-only
convention) keeps its original score unchanged rather than being scored
`0.0` or dropped — an honest, disclosed fallback for "no signal available
to re-rank this one," not silently wrong output. Returns candidates
re-sorted descending by whichever score (re-ranked or original-preserved)
each ended up with.

## The pipeline: `GateMemories`

Per #15's own text, "assembling context runs all of this as a pipeline:
retrieve candidates across every tier and strategy, score them by a
composite of similarity, recency, salience, and task fit, filter by hard
constraints (max tokens, forbidden topics, required inclusions), assemble
the prompt with the resulting ordering, and inject the result." The parent
issue's own Definition of Done ("Validated against MAG.md's described
behavior") is what this orchestrator exists to satisfy — mirroring Batch
C's `RecencyDecayFusionRetrieval`, which composed that batch's five
strategies under its own numbered issue (#75); gating has no equivalent
numbered "pipeline" child, so this orchestrator is built under the parent
issue #15 directly.

`src/mag/application/gating/gate_memories.py`:

```python
class GateMemories:
    def execute(
        self,
        episodes: list[ScoredEpisode],
        facts: list[ScoredFact],
        graph_nodes: list[ActivatedNode],
        token_budget: int,
        query_embedding: list[float] | None = None,
        allowed_source_types: set[str] | None = None,
        recency_half_life_hours: float = 24.0,
        now: datetime | None = None,
    ) -> list[GatingCandidate]:
```

Stages, in order (each optional stage skipped when its parameter isn't
given — same explicit-parameters convention as every strategy above, and
as Batch C's fusion orchestrator):

1. Convert all three input lists to `GatingCandidate` via the adapter
   functions and concatenate into one pool.
2. If `query_embedding` given: `DynamicReranking` over the whole pool.
3. If `allowed_source_types` given: `TaskSpecificFiltering`.
4. `RecencyWeightedSampling` (always runs — `half_life_hours` has a
   default, matching how temporal/salience always ran in Batch C's own
   fusion regardless of which optional legs were supplied).
5. `TokenBudgetAllocation` with `token_budget` (the "hard constraint: max
   tokens" #15 names) — chosen over `TopKSelection` as the pipeline's
   default selection step because #54's own text frames it as the
   generally better fit ("a better fit than Top-K when memories vary
   wildly in length," which a mixed episode/fact/graph-node pool always
   does); `TopKSelection` remains available as its own standalone
   strategy for a caller that wants it directly, just not wired into this
   default pipeline.
6. `HierarchicalAssembly` last, ordering whatever survived selection.

`TopKSelection` is deliberately NOT one of the pipeline's own stages for
the reason in step 5 — it's still fully implemented and tested as its own
independent strategy (#53's own Definition of Done doesn't depend on being
wired into the default pipeline to be "implemented" and "validated").

"Forbidden topics" and "required inclusions" (#15's other two named hard
constraints) are **not implemented by this batch** — see "What this batch
does not do" below.

## Testing plan

- Unit tests per strategy: pure-function behavior, no fakes or infra
  needed (candidates are just constructed directly). Covers the actual
  scoring/ordering/filtering math for each of the six, including edge
  cases each strategy's own design section above calls out (timestamp-less
  candidates in recency weighting, embedding-less candidates in dynamic
  re-ranking, empty candidate lists).
- Unit tests for the three `_candidates.py` adapter functions, and for
  `GateMemories`'s stage-skipping behavior (mirroring Batch C fusion's own
  "only supplied legs run" test coverage).
- One integration test reproducing MAG.md's own worked example
  structurally, with real data: real Postgres-backed episodic/semantic
  retrieval (Batch A/C infrastructure) and a real embedding model feeding
  `GateMemories`, asserting the token budget is genuinely respected (via
  real `tiktoken` counts, not estimated) and that facts are ordered ahead
  of episodes in the final assembly.

## What this batch does not do

- **No "forbidden topics" or "required inclusions" hard constraints** —
  #15's own pipeline description names these alongside the token-budget
  constraint, but none of the six numbered child issues asks for either
  one specifically, and neither has an obvious data source in this
  schema yet (what makes a topic "forbidden," or a fact "required," isn't
  defined anywhere in `DATABASE.md` or `MAG.md`). Deferred, not silently
  dropped.
- **No procedural-memory candidate type** — see #58's section above;
  nothing upstream currently produces a scored, listable procedural
  memory result the way episodic/semantic/graph retrieval do.
- **No automatic query-string embedding inside `DynamicReranking`** — the
  caller supplies `query_embedding` directly, consistent with this
  project's established explicit-parameters boundary.
- **`CompressingRetriever` is not refactored** to use the new shared
  `src/shared/tokenization.py` utility — it already works; this batch adds
  the shared utility for its own new code, not as an excuse to touch
  RAG's already-reviewed, working implementation.
