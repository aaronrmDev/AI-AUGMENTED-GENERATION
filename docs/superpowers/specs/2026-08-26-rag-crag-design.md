# RAG CRAG (Corrective RAG) — Design Spec

**Status:** Approved (continuation of the standing "full RAG backlog, case by case and
combinations" directive; process compressed per this session's established
"tomorrow morning" pacing — lean spec, no separate brainstorming dialogue — rigor is not
compressed: real code, real tests, real live Ollama/Qdrant measurement, honest reporting).

## What CRAG does (from `docs/architecture/RAG.md` and issue #73)

CRAG is the Post-Retrieval validation step: retrieve, evaluate each candidate for
relevance, decide, and — only if the set as a whole fails — correct via a refined
re-search. RAG.md's stated correction actions are "refining the query, trying an
alternative search, expanding and re-ranking, or filtering out the noise." Expected
impact: 40–60% reduction in hallucinations (RAG.md's largest single-technique figure).

## Composition shape: `Retriever` decorator, not a sibling use case

CRAG's `execute(tenant_id, query, top_k) -> list[SearchResult]` never needs a
fundamentally different return type and never needs to skip retrieval entirely (unlike
Self-RAG) — it always ends in a list of `SearchResult`, whether that list is the
relevance-filtered original results or a fresh re-search's results. It composes exactly
like `RerankingRetriever`, `HybridSearchDocuments`, `ParentDocumentRetriever`,
`CompressingRetriever`, `MultiQueryRetriever`, and `HyDERetriever`: wraps an inner
`Retriever`, decides what to return.

```python
class CorrectiveRetriever(Retriever):
    def __init__(self, inner: Retriever, chat_model: ChatModel) -> None: ...
    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]: ...
```

## The loop

1. **Retrieve**: call `inner.execute(tenant_id, query, top_k)`.
2. **Evaluate**: for each result, one `chat_model.complete(prompt)` call asking "does
   this passage directly help answer the query? Respond with ONLY YES or NO." —
   `complete()`, not `generate()`: this is a classification task, not a "use only the
   provided context" QA task, and Batch C's final review already found that exact
   confusion silently breaks two techniques (HyDE, Self-RAG) by routing a non-QA prompt
   through `generate()`'s hardcoded RAG-answering system prompt. Parsed with the same
   `\b(yes|no)\b` regex Batch C's review fix established, for the same reason (a fixed
   prefix window both misses a late "YES"/"NO" and misreads a substring of an unrelated
   word). Default for an ambiguous/unparseable response: **not relevant** — CRAG's whole
   purpose is trustworthiness, so an ambiguous judgment errs toward excluding rather than
   including questionable content, the opposite direction of Self-RAG's gate (which
   defaulted to retrieving, since skipping a needed retrieval risks a worse failure than
   an unnecessary one — here, including untrustworthy content is the worse failure).
3. **Decide**: if anything passed relevance review, return the relevant subset (RAG.md's
   "filtering out the noise"). **Revised after this batch's final review**: the first
   implementation used a strict-majority threshold (`> len(results) / 2`), which reads
   RAG.md's "documents that pass get used, and if the set as a whole fails, CRAG
   triggers a correction" too strictly. Live measurement caught the real consequence —
   in the ordinary case where a precise factual query's answer lives in exactly one of
   `top_k` retrieved chunks, a single correct match can never be a majority, so the
   threshold discarded a correctly-identified answer chunk and replaced it with an
   unvalidated re-search on 5 of the 7 questions this batch measured. "The set as a
   whole fails" is the more faithful reading of zero results passing, not
   less-than-a-majority passing, and returning any non-empty relevant subset is also the
   simpler, more defensible rule: it never discards a validated, correctly-identified
   answer chunk it already has in hand.
4. **Correct** (only when nothing passed relevance review): generate one refined/
   alternative phrasing of the query via `chat_model.complete()` (same style as
   HyDE/Multi-Query's prompt templates; falls back to the original query if the
   completion is blank), then call `inner.execute()` once more with that refined query at
   the same `top_k`, and return those results directly — RAG.md's "trying an alternative
   search." Single-shot: no retry loop, no unbounded correction chain, so latency stays
   boundable and the behavior stays testable. The corrected re-search's results are
   returned as-is, with no second relevance pass — a disclosed scope boundary, not an
   oversight: verifying the correction actually helped would need a second full
   evaluate step, doubling worst-case latency for a technique whose cost is already
   substantial (see the report's latency figures).

`top_k == 0` and an empty inner result list are both edge cases the implementation
handles by short-circuiting: an empty result list has nothing to evaluate or be wrong
about, so it returns the empty list directly, with no evaluation calls made.

## Composability into "combinations" (Batch E work, not this batch)

CRAG naturally slots into the position `HybridSearchDocuments` / reranking already
occupy — it wraps whatever retriever came before it. Once merged, three of the five
still-unbuilt RAG combinations become buildable with zero new production code, exactly
like Batches A and B's combinations "fell out for free":
- **Reranking + CRAG** (#125/#139): `CorrectiveRetriever(inner=RerankingRetriever(...))`
  or the reverse order — RAG.md's own text ("CRAG evaluates... reranking runs again on
  the new results") implies CRAG wraps outermost, so a correction's re-search still gets
  reranked.
- **Production Grade** (#132/#141): Hybrid Search → Reranking → CRAG → Context
  Compression, chaining `CompressingRetriever(inner=CorrectiveRetriever(inner=
  RerankingRetriever(inner=HybridSearchDocuments(...))))`.
- **Fort Knox** (#135/#142): the six-concept chain, built the same nesting way.

## Out of scope for this batch

- The "expanding and re-ranking" correction action (would require a `Reranker`
  dependency injected into `CorrectiveRetriever` just for the correction path) — the
  simpler "alternative search" action already satisfies RAG.md's stated correction
  behaviors and keeps this class's dependencies to exactly `Retriever` + `ChatModel`,
  consistent with every other decorator in this codebase.
- Batch E's combination scenarios — tracked separately, unblocked by this batch's merge.
