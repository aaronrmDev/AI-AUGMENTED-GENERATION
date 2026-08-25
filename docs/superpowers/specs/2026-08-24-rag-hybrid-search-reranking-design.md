# RAG Hybrid Search + Reranking — Design Spec

Compressed process, per the user's own timeline pressure (2026-08-24): a bounded spec, not the full architectural ceremony, since the shape here follows an already-established precedent (the `Chunker` port from RAG Chunking Strategies) almost exactly. Real code, real tests, real live measurement throughout — the compression is in ceremony, not in rigor.

## Why this exists

RAG.md names Hybrid Search (#44/#102) and Reranking (#40/#90/#94/#98) as the two techniques its own "beginner implementation path" builds first, alongside chunking — "~70-80% of production quality" from just these three. Chunking is done; this batch builds the other two, plus measures each in isolation and as the "Hybrid Search + Reranking" combination RAG.md calls "the production standard" (§ High-synergy combinations) — sets up issue #123/#137 (that specific combination issue) almost for free once both pieces exist.

## A new `Retriever` port, and why it replaces `SearchDocuments`'s concrete type

`AnswerQuestion.__init__` currently takes a concrete `SearchDocuments`. Hybrid Search and Reranking are both alternative-or-composable retrieval strategies, exactly the situation the RAG Pipeline final review flagged for `Chunker` before it existed — same fix, same shape:

```python
class Retriever(ABC):
    @abstractmethod
    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]: ...
```

This is `SearchDocuments.execute`'s existing exact signature — `SearchDocuments(Retriever)` costs nothing, same pattern as `FixedSizeChunker(Chunker)`. `AnswerQuestion`'s `search_documents: SearchDocuments` param narrows to `search_documents: Retriever`.

## Hybrid Search (#44, #102)

**`BM25KeywordSearch(Retriever)`** — new infrastructure adapter. Fetches every chunk for the tenant from Postgres (new `DocumentRepository.get_chunks_for_tenant(tenant_id) -> list[Chunk]` method — chunk content is already stored there, nothing new to persist), tokenizes each chunk's content and the query (lowercase, split on non-alphanumeric via a simple regex — no need for a heavier tokenizer at this scale), and scores with `rank_bm25`'s `BM25Okapi` (new dependency — pure-Python, no external service, matches this project's "no new infra for a proof" bias). Returns the top-`k` chunks as `SearchResult`s with the BM25 score in `.score`. Rebuilding the BM25 index per call is O(corpus size) and fine at this project's proof-of-concept scale (tens to low hundreds of chunks per tenant); caching the index is a real production concern this spec explicitly defers, since nothing in this batch's measurement needs it.

**`HybridSearchDocuments(Retriever)`** — composes an existing `Retriever` (the vector search, `SearchDocuments`) and `BM25KeywordSearch`, runs both concurrently (`asyncio.gather`) against a wider candidate pool (`candidate_k`, default 20) than the final `top_k`, and merges via Reciprocal Rank Fusion — RAG.md's own named default merge method (it also lists Weighted Score Combination, Relative Score Fusion, and Rank Based Fusion as alternatives; RRF is what this batch builds, matching "typically via Reciprocal Rank Fusion"). RRF score per chunk: `sum over the result lists containing it of 1 / (k_rrf + rank)`, `k_rrf = 60` (the standard constant from the original RRF paper, which the source doesn't override). A chunk appearing in both lists gets both terms added — this is the mechanism that lets hybrid search reward a chunk that both search methods agree on. Dedup key is `chunk_id`. Sorted descending by RRF score, truncated to `top_k`.

## Reranking (#40, #90, #94, #98)

```python
class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...
```

**`RerankingRetriever(Retriever)`** — a decorator, not a new retrieval mechanism: wraps any inner `Retriever`, asks it for a wider candidate pool (`candidate_k`, default 20) than the final `top_k`, then reranks that pool down to `top_k`. Wrapping `SearchDocuments` measures "Reranking alone"; wrapping `HybridSearchDocuments` (same class, no new code) is what a later combination measurement needs — this is why the decorator shape is worth the small extra indirection over bolting reranking directly onto `AnswerQuestion`.

Three `Reranker` implementations, matching RAG.md's own accuracy/speed/cost table (§ Post-Retrieval, Reranking):

- **`CrossEncoderReranker`** (#90) — `sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`, a real cross-encoder model (already have `sentence-transformers` as a dependency; this is one more model from the same library, not a new one). Scores every `(query, chunk.content)` pair jointly — the textbook cross-encoder mechanic RAG.md itself describes ("looking at the query and each retrieved chunk together, rather than scoring them independently"). Sorts descending, takes `top_k`. This is the source's own beginner-path pick (`BAAI/bge-reranker-base` named there specifically; `ms-marco-MiniLM-L-6-v2` is used here as the equivalent well-known, small, fast cross-encoder already common in this ecosystem — swapping the exact checkpoint doesn't change what's being measured, the reranking *mechanism*).
- **`BiEncoderRerankReranker`** (#94) — the "balanced" middle option. Re-embeds the query and each candidate with the same bi-encoder already in this stack (`SentenceTransformersEmbedder`, no new model) for a semantic score, and adds a lexical term-overlap score (Jaccard similarity between the query's and the chunk's lowercased token sets) as a second, cheap signal — blended `0.7 * semantic + 0.3 * lexical`. A pure re-embed-and-cosine pass alone would just reproduce the initial vector search's own ordering (same model, same math), so the lexical blend is what makes this a genuinely distinct second pass rather than a no-op — while staying deliberately lighter than the full cross-encoder, which is the accuracy/speed/cost tradeoff RAG.md's table describes for this row.
- **`LLMReranker`** (#98) — uses the existing `ChatModel` port (Ollama for this batch's live runs, matching every prior comparison this project has run). Prompts the model once per candidate with the query and the chunk's content, asking for a single integer relevance score 0-10; parses the score (a malformed response scores 0, logged rather than crashing the whole rerank — the one place in this batch that talks to a model per-candidate rather than once per query, so it has to tolerate one bad response without losing the rest). Sorts descending by score. This is the source's own "highest accuracy, slowest, costliest" row — genuinely paying for an LLM call per candidate is what makes it that.

## Scenario and measurement

New scenario `evaluation/scenarios/rag-hybrid-reranking/` — same corpus (`docs/architecture/RAG.md`, already proven byte-identical and content-grounded from the chunking-strategies batch) and the same `queries.yaml` shape (5 real, source-grounded questions with substring `success_check`s), reused rather than re-derived where the existing 5 chunking-strategy questions still apply, extended with 2-3 more that specifically probe Hybrid Search's and Reranking's own RAG.md content (e.g. RRF, the three reranker types' accuracy/speed/cost tradeoff) so the questions aren't accidentally about the wrong technique.

Five comparisons this batch measures, all against the same no-RAG baseline, all `rag=True, cag=False, mag=False`, chunker held constant at `FixedSizeChunker` (already measured, not the variable under test here):
1. Vector-only (`SearchDocuments`) — already effectively measured in the chunking batch's `fixed-size` report, not re-run.
2. Hybrid Search (`HybridSearchDocuments`) vs. vector-only baseline.
3. Cross-Encoder reranking (`RerankingRetriever` wrapping `SearchDocuments`).
4. Bi-Encoder+Rerank reranking (same wrapper, different `Reranker`).
5. LLM reranking (same wrapper, different `Reranker`).
6. Hybrid Search + Cross-Encoder reranking (`RerankingRetriever` wrapping `HybridSearchDocuments`) — this is #123/#137's actual combination measurement, landing as a side effect of the decorator shape rather than needing its own separate build.

Per this project's own tracked follow-up (#147), the qualitative judge is known to need an independent model and a fixed reference context before its scores are trustworthy for between-strategy comparison — this batch's own qualitative results carry the same caveat already added to every chunking-strategy report, not a fresh problem to solve here.

## Non-goals

No changes to `UploadDocument` or the `Chunker` port — chunking is already measured and held constant. No BM25 index caching or incremental updates — rebuilt per query, a real production gap named and deferred, not hidden. No new evaluation-harness changes — Batch A reuses `RunComparison`/`OllamaJudge` exactly as the chunking-strategies batch left them.
