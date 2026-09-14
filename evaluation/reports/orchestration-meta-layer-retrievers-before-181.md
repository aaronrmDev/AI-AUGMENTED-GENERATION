# Orchestration Meta-Layer — RAG Retrievers in a PARALLEL Route

Every question is forced into a PARALLEL route across CAG, MAG, and RAG, once per RAG composition behind RagTier: 10 questions x 5 repeats each, default TierTimeouts, after one untimed warm-up, on the corpus and thresholds of orchestration-meta-layer-cascade.md. The pipeline's shared CachingEmbeddingModel serves the question's embedding to every tier and to SearchDocuments, so SearchDocuments alone embeds nothing new. The other compositions embed through the uncached MiniLM model on every attempt: CompressingRetriever its query and every candidate sentence, BiEncoderRerankReranker its query and every candidate, HyDERetriever's search its generated passage, and CacheWarmedRetrieve its query, against its own warmed copy of the 4 frozen documents. BM25KeywordSearch, inside HybridSearchDocuments, ranks an in-memory copy of the same 6 chunks, and CrossEncoderReranker runs the real ms-marco MiniLM cross-encoder on CPU. HyDE's passage comes from a stub chat model that returns a new passage at once: generation is awaited network I/O that yields the loop, so the stub isolates the embedding. A timed-out CAG match keeps its worker thread running to completion, so an attempt measured right after a CAG timeout can include that overlap. Measured on commit 774e270, before #181 moved BM25KeywordSearch, CrossEncoderReranker, and CacheWarmedRetrieve.execute off the event loop; the first four compositions were already fixed by #168.

## Tier latency against Concept 5's budgets, per RAG retriever

| Retriever | Tier | Attempts | Outcomes | p50 ms | p95 ms | Budget ms | Within budget |
|---|---|---|---|---|---|---|---|
| SearchDocuments | CAG | 50 | hit 30, miss 20 | 1.84 | 2.50 | 10 | 100% |
| SearchDocuments | MAG | 50 | hit 20, partial 5, miss 25 | 6.41 | 7.61 | 50 | 100% |
| SearchDocuments | RAG | 50 | hit 50 | 5.86 | 7.33 | 2000 | 100% |
| CompressingRetriever | CAG | 50 | hit 30, miss 20 | 1.99 | 2.43 | 10 | 100% |
| CompressingRetriever | MAG | 50 | hit 20, partial 5, miss 25 | 9.62 | 14.49 | 50 | 100% |
| CompressingRetriever | RAG | 50 | hit 50 | 77.50 | 112.35 | 2000 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | CAG | 50 | hit 30, miss 20 | 2.00 | 2.37 | 10 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | MAG | 50 | hit 20, partial 5, miss 25 | 10.50 | 17.53 | 50 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | RAG | 50 | hit 50 | 105.19 | 147.67 | 2000 | 100% |
| HyDERetriever | CAG | 50 | hit 30, miss 20 | 1.55 | 1.94 | 10 | 100% |
| HyDERetriever | MAG | 50 | hit 20, partial 5, miss 25 | 7.16 | 9.01 | 50 | 100% |
| HyDERetriever | RAG | 50 | hit 50 | 20.61 | 24.69 | 2000 | 100% |
| HybridSearchDocuments (SearchDocuments + BM25KeywordSearch) | CAG | 50 | hit 30, miss 20 | 1.80 | 2.59 | 10 | 100% |
| HybridSearchDocuments (SearchDocuments + BM25KeywordSearch) | MAG | 50 | hit 20, partial 5, miss 25 | 6.97 | 9.57 | 50 | 100% |
| HybridSearchDocuments (SearchDocuments + BM25KeywordSearch) | RAG | 50 | hit 50 | 6.53 | 8.49 | 2000 | 100% |
| RerankingRetriever + CrossEncoderReranker | CAG | 50 | hit 30, miss 20 | 3.81 | 4.65 | 10 | 100% |
| RerankingRetriever + CrossEncoderReranker | MAG | 50 | timeout 50 | 76.97 | 87.62 | 50 | 0% |
| RerankingRetriever + CrossEncoderReranker | RAG | 50 | hit 50 | 81.04 | 88.47 | 2000 | 100% |
| CacheWarmedRetrieve | CAG | 50 | timeout 50 | 13.86 | 18.12 | 10 | 0% |
| CacheWarmedRetrieve | MAG | 50 | timeout 50 | 60.11 | 67.43 | 50 | 0% |
| CacheWarmedRetrieve | RAG | 50 | hit 50 | 15.56 | 66.16 | 2000 | 100% |
