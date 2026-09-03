# RAG Parent Document Chunking + Retrieval + Context Compression — Design Spec

Compressed process, same as the Hybrid Search + Reranking batch — bounded spec, real code/tests/measurement throughout.

## Why this exists, and why these three together

The chunking-strategies batch deliberately deferred Parent Document Chunking (#85): RAG.md frames its effect as showing up at *retrieval* time, when a matched child chunk maps back to its parent — measuring #85 alone, without Parent Document Retrieval (#56/#105), would show no signal for the wrong reason. Both are built here, together, plus Context Compression (#61/#107) — RAG.md's own "Parent Document + Context Compression" pairing (§ High-synergy combinations, "the perfect balance") is directly what this batch measures as its combination case, matching #129/#140.

## Composition order — resolved from RAG.md's own text, not assumed

RAG.md's own description of the mechanic: "search against small chunks for precision, then map each **top result** back to its parent... before sending anything to the LLM." Note the order: search returns its *already-narrowed* top results, and expansion to parent happens *after* that, immediately before the LLM sees anything. That means parent expansion is the **outermost** step relative to search/rerank, and — because Compression's whole job is trimming the redundancy Parent Document's expansion introduces (RAG.md: "Compression trims exactly that expansion back down") — Compression is outermost of all three, wrapping the parent-expanded result.

Composition order, outermost to innermost: **Compression → Parent Document expansion → Reranking → (Hybrid/Vector search)**. Concretely: `CompressingRetriever` wraps `ParentDocumentRetriever` wraps (optionally) `RerankingRetriever` wraps `SearchDocuments`/`HybridSearchDocuments`. Reranking, when present, judges the small child-level candidates (cheap, and matches RAG.md's own "precision" framing for why children are searched in the first place); expansion and compression both operate on the already-decided final set, never on a wider candidate pool.

## Parent Document Chunking (#85) + Retrieval (#56/#105)

New entity, `src/rag/domain/entities.py`:
```python
@dataclass(frozen=True)
class ParentChildChunks:
    parents: list[str]
    children: list[tuple[str, int]]  # (child content, index into parents)
```

New `src/rag/infrastructure/parent_document_chunker.py`, `ParentDocumentChunker` — not a `Chunker` (its output shape is two-tiered, not a flat `list[str]`):
```python
class ParentDocumentChunker:
    def __init__(self, parent_chunk_size_tokens: int = 1000, child_chunk_size_tokens: int = 200) -> None:
        self._parent_chunker = FixedSizeChunker(chunk_size_tokens=parent_chunk_size_tokens)
        self._child_chunker = FixedSizeChunker(chunk_size_tokens=child_chunk_size_tokens)

    def chunk_with_parents(self, text: str) -> ParentChildChunks:
        parents = self._parent_chunker.chunk(text)
        children: list[tuple[str, int]] = []
        for i, parent in enumerate(parents):
            children.extend((child, i) for child in self._child_chunker.chunk(parent))
        return ParentChildChunks(parents=parents, children=children)
```
1000/200 tokens matches RAG.md's own recommendation range for child chunks ("very small chunks (100-300 tokens)") and gives parents enough room to be a meaningfully larger section.

New use case `src/rag/application/upload_document_with_parents.py`, `UploadDocumentWithParents` — mirrors `UploadDocument.execute()`'s shape (extract → store file → save `Document`) but saves two tiers: parent `Chunk`s (`parent_id=None`, `embedding=[]` — parents are never searched directly, only fetched by id) and child `Chunk`s (`parent_id=<the real parent chunk's id>`, real embedding). Only child chunks get upserted to the vector store — that's what keeps retrieval precise while parents stay fetchable.

New repository method: `DocumentRepository.get_chunk_by_id(chunk_id: uuid.UUID) -> Chunk | None` (Postgres: `SELECT ... FROM chunks WHERE id = :id`).

New decorator `src/rag/infrastructure/parent_document_retriever.py`:
```python
class ParentDocumentRetriever(Retriever):
    def __init__(self, inner: Retriever, document_repository: DocumentRepository) -> None:
        self._inner = inner
        self._documents = document_repository

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        child_results = await self._inner.execute(tenant_id=tenant_id, query=query, top_k=top_k)
        expanded: list[SearchResult] = []
        seen_parent_ids: set[uuid.UUID] = set()
        for child in child_results:
            child_chunk = await self._documents.get_chunk_by_id(child.chunk_id)
            if child_chunk is None or child_chunk.parent_id is None:
                continue
            if child_chunk.parent_id in seen_parent_ids:
                continue
            parent_chunk = await self._documents.get_chunk_by_id(child_chunk.parent_id)
            if parent_chunk is None:
                continue
            seen_parent_ids.add(child_chunk.parent_id)
            expanded.append(SearchResult(
                document_id=child.document_id, chunk_id=child_chunk.parent_id,
                content=parent_chunk.content, score=child.score,
            ))
        return expanded
```
Note `execute` requests `top_k` from `self._inner`, not a wider pool — matches the resolved composition order above (expansion operates on the already-final ranked set). Dedup-by-parent-id is deliberate: RAG.md's own caveat is "avoid pulling in too many parents at once"; two matched children from the same parent section return that parent once, not twice, which also means `len(expanded) <= top_k` but can be *less* than `top_k` when children collapse onto shared parents — a real, expected, and disclosed property of this technique, not a bug to paper over.

## Context Compression (#61, #107)

RAG.md names "extractive compression that keeps the top-N sentences by relevance" as its own building method. New `src/rag/infrastructure/extractive_compressor.py`, composed as a decorator matching the `RerankingRetriever`/`HybridSearchDocuments` shape already established:
```python
class CompressingRetriever(Retriever):
    def __init__(self, inner: Retriever, embedding_model: EmbeddingModel, target_tokens: int = 2000) -> None:
        ...
    async def execute(self, tenant_id, query, top_k) -> list[SearchResult]:
        results = await self._inner.execute(tenant_id=tenant_id, query=query, top_k=top_k)
        # split every result's content into sentences (reuse
        # src/rag/infrastructure/_sentence_splitter.split_sentences), score
        # each sentence's cosine similarity to the query embedding, greedily
        # keep the highest-scoring sentences (across ALL results pooled
        # together, not per-result) until target_tokens is spent, then
        # rebuild each result's .content from only its kept sentences in
        # their original order -- a result that contributes zero kept
        # sentences is dropped from the output entirely.
```
Pooling sentence selection across all results (not per-result) is what actually removes redundancy across chunks, not just within one — matching RAG.md's own description of the payoff ("removes duplicate chunks... redundancy removal").

## Scenario and measurement

New scenario `evaluation/scenarios/rag-parent-doc-compression/`, same corpus/harness/judge pattern as every prior batch. Strategies:
1. `parent-document` — `ParentDocumentRetriever` wrapping `SearchDocuments`, using `UploadDocumentWithParents`/`ParentDocumentChunker` instead of the fixed-size upload every other strategy in this project has used.
2. `context-compression` — `CompressingRetriever` wrapping `SearchDocuments`, chunker held at `FixedSizeChunker` (compression doesn't need the two-tier upload).
3. `parent-document-compression` — `CompressingRetriever` wrapping `ParentDocumentRetriever` wrapping `SearchDocuments`, using the two-tier upload. This is #129/#140's actual measurement, falling out of the decorator composition with zero new production code, same trick Batch A's `hybrid-rerank-cross-encoder` already proved.

5 questions grounded in the live `docs/architecture/RAG.md` at measurement time (read it fresh, don't reuse an old batch's quotes) — cover: Parent Document Retrieval's own mechanic and expected impact figure (+15-20% completeness), Context Compression's two payoff numbers (the -75% worked-example figure vs. the more conservative -50%/+10% general figure — RAG.md is explicit these shouldn't be conflated, a good disambiguation test), the "Parent Document + Context Compression" combination's named quality ("the perfect balance"), and 2 more of the implementer's choice.

## Non-goals

No changes to `UploadDocument`, `Chunker`, or any Batch-A retriever/reranker — `ParentDocumentRetriever`/`CompressingRetriever` are new, additive decorators. No wiring into `src/api/` — matches every prior batch's measurement-only scope. No re-litigating the qualitative-judge limitation (#147) or the `candidate_k`-vs-corpus-size disclosure pattern Batch A established — both apply identically here and get the same caveat treatment, not a fresh investigation.
