# Orchestration Meta-Layer — RAG Retrievers in a PARALLEL Route

Every question is forced into a PARALLEL route across CAG, MAG, and RAG, once per RAG composition behind RagTier: 10 questions x 5 repeats each, default TierTimeouts, after one untimed warm-up, on the corpus and thresholds of orchestration-meta-layer-cascade.md. The pipeline's shared CachingEmbeddingModel serves the question's embedding to every tier and to SearchDocuments, so SearchDocuments alone embeds nothing new. The other compositions embed through the uncached MiniLM model on every attempt: CompressingRetriever its query and every candidate sentence, BiEncoderRerankReranker its query and every candidate, and HyDERetriever's search its generated passage. That passage comes from a stub chat model that returns a new passage at once: generation is awaited network I/O that yields the loop, so the stub isolates the embedding. A timed-out CAG match keeps its worker thread running to completion, so an attempt measured right after a CAG timeout can include that overlap. Measured on commit 099c99d, after #168 moved these retrievers' embedding off the event loop.

## Tier latency against Concept 5's budgets, per RAG retriever

| Retriever | Tier | Attempts | Outcomes | p50 ms | p95 ms | Budget ms | Within budget |
|---|---|---|---|---|---|---|---|
| SearchDocuments | CAG | 50 | hit 30, miss 20 | 1.75 | 2.73 | 10 | 100% |
| SearchDocuments | MAG | 50 | hit 20, partial 5, miss 25 | 6.68 | 8.32 | 50 | 100% |
| SearchDocuments | RAG | 50 | hit 50 | 5.95 | 7.72 | 2000 | 100% |
| CompressingRetriever | CAG | 50 | hit 30, miss 20 | 1.86 | 2.14 | 10 | 100% |
| CompressingRetriever | MAG | 50 | hit 20, partial 5, miss 25 | 9.57 | 13.37 | 50 | 100% |
| CompressingRetriever | RAG | 50 | hit 50 | 65.17 | 82.18 | 2000 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | CAG | 50 | hit 30, miss 20 | 1.82 | 1.97 | 10 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | MAG | 50 | hit 20, partial 5, miss 25 | 9.32 | 14.40 | 50 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | RAG | 50 | hit 50 | 83.98 | 94.00 | 2000 | 100% |
| HyDERetriever | CAG | 50 | hit 30, miss 20 | 1.37 | 1.88 | 10 | 100% |
| HyDERetriever | MAG | 50 | hit 20, partial 5, miss 25 | 6.63 | 7.76 | 50 | 100% |
| HyDERetriever | RAG | 50 | hit 50 | 18.29 | 20.40 | 2000 | 100% |
