# Orchestration Meta-Layer — RAG Retrievers in a PARALLEL Route

Every question is forced into a PARALLEL route across CAG, MAG, and RAG, once per RAG composition behind RagTier: 10 questions x 5 repeats each, default TierTimeouts, after one untimed warm-up, on the corpus and thresholds of orchestration-meta-layer-cascade.md. The pipeline's shared CachingEmbeddingModel serves the question's embedding to every tier and to SearchDocuments, so SearchDocuments alone embeds nothing new. The other compositions embed through the uncached MiniLM model on every attempt: CompressingRetriever its query and every candidate sentence, BiEncoderRerankReranker its query and every candidate, HyDERetriever's search its generated passage, and CacheWarmedRetrieve its query, against its own warmed copy of the 4 frozen documents. BM25KeywordSearch, inside HybridSearchDocuments, ranks an in-memory copy of the same 6 chunks, and CrossEncoderReranker runs the real ms-marco MiniLM cross-encoder on CPU. HyDE's passage comes from a stub chat model that returns a new passage at once: generation is awaited network I/O that yields the loop, so the stub isolates the embedding. A timed-out CAG match keeps its worker thread running to completion, so an attempt measured right after a CAG timeout can include that overlap. Measured on commit f641b0d, after #168 and #181 moved every composition's CPU work off the event loop.

## Tier latency against Concept 5's budgets, per RAG retriever

| Retriever | Tier | Attempts | Outcomes | p50 ms | p95 ms | Budget ms | Within budget |
|---|---|---|---|---|---|---|---|
| SearchDocuments | CAG | 50 | hit 30, miss 20 | 1.75 | 2.30 | 10 | 100% |
| SearchDocuments | MAG | 50 | hit 20, partial 5, miss 25 | 5.98 | 7.18 | 50 | 100% |
| SearchDocuments | RAG | 50 | hit 50 | 5.48 | 6.75 | 2000 | 100% |
| CompressingRetriever | CAG | 50 | hit 30, miss 20 | 2.02 | 2.15 | 10 | 100% |
| CompressingRetriever | MAG | 50 | hit 20, partial 5, miss 25 | 9.18 | 12.01 | 50 | 100% |
| CompressingRetriever | RAG | 50 | hit 50 | 62.25 | 76.23 | 2000 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | CAG | 50 | hit 30, miss 20 | 2.01 | 2.18 | 10 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | MAG | 50 | hit 20, partial 5, miss 25 | 9.08 | 11.20 | 50 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | RAG | 50 | hit 50 | 85.48 | 95.88 | 2000 | 100% |
| HyDERetriever | CAG | 50 | hit 30, miss 20 | 1.43 | 1.62 | 10 | 100% |
| HyDERetriever | MAG | 50 | hit 20, partial 5, miss 25 | 6.35 | 8.21 | 50 | 100% |
| HyDERetriever | RAG | 50 | hit 50 | 18.63 | 20.83 | 2000 | 100% |
| HybridSearchDocuments (SearchDocuments + BM25KeywordSearch) | CAG | 50 | hit 30, miss 20 | 1.87 | 2.43 | 10 | 100% |
| HybridSearchDocuments (SearchDocuments + BM25KeywordSearch) | MAG | 50 | hit 20, partial 5, miss 25 | 6.27 | 7.30 | 50 | 100% |
| HybridSearchDocuments (SearchDocuments + BM25KeywordSearch) | RAG | 50 | hit 50 | 5.99 | 6.91 | 2000 | 100% |
| RerankingRetriever + CrossEncoderReranker | CAG | 50 | hit 30, miss 20 | 2.08 | 2.39 | 10 | 100% |
| RerankingRetriever + CrossEncoderReranker | MAG | 50 | hit 20, partial 5, miss 25 | 7.23 | 8.48 | 50 | 100% |
| RerankingRetriever + CrossEncoderReranker | RAG | 50 | hit 50 | 32.26 | 35.59 | 2000 | 100% |
| CacheWarmedRetrieve | CAG | 50 | hit 30, miss 20 | 1.10 | 1.49 | 10 | 100% |
| CacheWarmedRetrieve | MAG | 50 | hit 20, partial 5, miss 25 | 7.77 | 30.02 | 50 | 100% |
| CacheWarmedRetrieve | RAG | 50 | hit 50 | 15.04 | 30.74 | 2000 | 100% |
