# RAG Combinations (Batch E) — Design Spec

**Status:** Approved (continuation of the standing "full RAG backlog, case by case and
combinations" directive; process compressed per this session's established pacing — lean
spec, no separate brainstorming dialogue — rigor is not compressed).

## Scope

Five of RAG.md's documented combinations have no dedicated implementation yet — every
underlying `Retriever`/`Reranker`/use-case piece they need already exists and is already
individually tested and measured (Batches A–D):

| Combination | Issues | Ingredients (all pre-existing) |
|---|---|---|
| Multi-Query + HyDE | #124/#138 | `MultiQueryRetriever`, `HyDERetriever` |
| Speed Demon | #136/#143 | `SelfRAGAnswerQuestion`, `HybridSearchDocuments`, `RerankingRetriever`, `CompressingRetriever` |
| Reranking + CRAG | #125/#139 | `RerankingRetriever`, `CorrectiveRetriever` |
| Production Grade | #132/#141 | `HybridSearchDocuments`, `RerankingRetriever`, `CorrectiveRetriever`, `CompressingRetriever` |
| Fort Knox | #135/#142 | `HybridSearchDocuments`, `MultiQueryRetriever`, `HyDERetriever`, `RerankingRetriever`, `CorrectiveRetriever`, `ParentDocumentRetriever` |

Per this project's own established pattern (Batches A and B's own combinations "fell out
for free"), none of these need new production classes — only nested composition, plus a
comparison scenario per combination.

## Composition order for each combination

Every `Retriever` decorator's constructor takes `inner: Retriever` — nesting order is
read innermost-first (what runs first) to outermost-last (what runs last, right before
`top_k` results reach the LLM).

- **Multi-Query + HyDE**: `MultiQueryRetriever(inner=HyDERetriever(inner=SearchDocuments))`.
  RAG.md: *"a pipeline can generate multiple hypothetical answers, or treat HyDE's
  hypothetical answer as one query among several diverse ones — dual enrichment."*
  Multi-Query generates N diverse phrasings of the question; each phrasing is
  independently run through HyDE (generating its own hypothetical answer, embedding
  that instead of the phrasing itself) before searching. This is what makes it "dual
  enrichment" rather than either technique alone — every one of Multi-Query's N searches
  also gets HyDE's vague-query rescue.
- **Speed Demon**: `SelfRAGAnswerQuestion(search_documents=CompressingRetriever(inner=
  RerankingRetriever(inner=HybridSearchDocuments(...))))`. Self-RAG isn't a `Retriever`
  (per Batch C's design), so it wraps the whole retrieval chain as its `search_documents`
  dependency — its gate decides whether to run any of this at all. Reranking then
  Compression, matching the established Batch A/B ordering (search/rerank innermost,
  compression outermost, right before the LLM).
- **Reranking + CRAG**: `CorrectiveRetriever(inner=RerankingRetriever(inner=
  SearchDocuments))`. RAG.md: *"when [CRAG] rejects a batch and triggers a corrective
  retrieval loop, reranking runs again on the new results"* — nesting CRAG outside
  Reranking means `CorrectiveRetriever`'s correction path re-invokes `inner.execute()`,
  which is the reranked search, automatically satisfying "reranking runs again on the
  new results" with zero special-case code.
- **Production Grade**: `CompressingRetriever(inner=CorrectiveRetriever(inner=
  RerankingRetriever(inner=HybridSearchDocuments(...))))`. RAG.md's own archetype text,
  in order: *"Hybrid Search casts the wide net, Reranking narrows it to true relevance,
  CRAG validates the result and triggers correction when needed, and Context Compression
  optimizes what's left before it reaches the LLM."* — a direct, literal pipeline.
- **Fort Knox**: `ParentDocumentRetriever(inner=CorrectiveRetriever(inner=
  RerankingRetriever(inner=MultiQueryRetriever(inner=HyDERetriever(inner=
  HybridSearchDocuments(...))))))`. Reasoning: Hybrid Search + HyDE + Multi-Query form
  the same "dual enrichment" search-and-rewrite stage as the standalone Multi-Query+HyDE
  combination above, just searching via Hybrid instead of plain vector; Reranking then
  CRAG follow the same Reranking+CRAG ordering established above (so CRAG's correction
  path re-invokes the entire enriched-search-then-rerank chain); Parent Document
  expansion goes outermost, matching Batch B's established "expansion operates on the
  already-final ranked set" ordering. This requires parent-linked chunks in the
  document repository, so Fort Knox's scenario uploads via `UploadDocumentWithParents`
  (Batch B), not the plain `UploadDocument` every other combination in this batch uses.

## A real, disclosed interaction: BM25 keyword search over a parent-chunked corpus

Fort Knox is the first combination in this project to combine `HybridSearchDocuments`
(whose keyword arm, `BM25KeywordSearch`, reads every chunk `DocumentRepository.
get_chunks_for_tenant` returns) with `UploadDocumentWithParents` (which saves *both*
parent and child chunks to that same repository). BM25's keyword search can therefore
surface a parent chunk as a search result — and `ParentDocumentRetriever` expects its
input to be *child* chunks: for a result whose `chunk_id` resolves to a parent chunk
(`parent_id is None`), it silently skips that result rather than expanding or erroring.
This is measured and disclosed honestly in Fort Knox's report as a real behavior, not
engineered around with new production code at this stage.

**Revised after this batch's final review, which measured the actual impact rather than
assuming it was bounded**: the mechanism stated in an earlier version of this section was
wrong, and so was the conclusion. `UploadDocumentWithParents`
(`src/rag/application/upload_document_with_parents.py`) never upserts parent chunks to
the vector store *at all* — the `[0.0] * 384` placeholder embedding exists solely to
satisfy the Postgres `Vector(384) NOT NULL` column, not to keep parents out of vector
search results (they were never eligible for the vector arm in the first place, upsert
or no). That means this interaction is not "only what BM25 can return" in some small,
bounded sense — parents are BM25's *natural favorites*: at roughly 969 tokens each
against children's roughly 182, they accumulate far more of BM25Plus's per-matched-term
score contribution. Measured directly against this batch's real corpus and question set:
parents occupied 6–7 of the 7 available parent slots in BM25's own top-20 on every one of
the 7 questions (47 of 49 possible appearances), with a parent's best BM25 rank landing
at position 2 (0-indexed 1) on all 7 questions. After `HybridSearchDocuments`' RRF fusion
and `RerankingRetriever`'s cross-encoder pass, parents still made up 13 of the 35 results
(37%) reaching `ParentDocumentRetriever` across a live 7-question sample — meaning over a
third of what CRAG and Reranking worked to surface got silently dropped at the very last
stage, not a rare edge case. This can degenerate a treatment sample into an empty
context (see Fort Knox's report notes) when too few of the survivors are children.
Genuinely fixing this — e.g. having `BM25KeywordSearch` accept an optional
child-only filter, or having `ParentDocumentRetriever` fall back to a parent's own
content instead of dropping it — is real production work, out of scope for this
composition-only batch; the point of this section is that the interaction is
disclosed honestly with its measured size, not "bounded" by assumption.

## Reranker and corpus choice

Every combination needing a `Reranker` uses `CrossEncoderReranker` — RAG.md's own
"beginner implementation path" recommendation ("a lightweight cross-encoder reranker...
specifically because it's fast enough to run on every request"), and the same choice
Batch A's own `hybrid-rerank-cross-encoder` combination already used.

All five combinations measure against the same corpus every batch in this project uses
(`docs/architecture/RAG.md`) and the same shared 7-question set (`queries.yaml`),
covering material that spans the contributing techniques of all five combinations, so
one scenario script (`run_combinations_comparison.py <combination>`) serves all five —
matching Batch C's multi-strategy-CLI-argument pattern rather than five separate
scenario directories with near-duplicate scaffolding.

## Out of scope

- New production classes — none needed.
- Re-deriving distinct question sets per combination — one shared set, reused across all
  five arms, keeps the comparison apples-to-apples and avoids 5x the quote-verification
  overhead for material already exercised in Batches A–D's own reports.
