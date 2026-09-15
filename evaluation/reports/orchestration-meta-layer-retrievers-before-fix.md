# Orchestration Meta-Layer — RAG Retrievers in a PARALLEL Route

Every question is forced into a PARALLEL route across CAG, MAG, and RAG, once per RAG composition behind RagTier: 10 questions x 5 repeats each, default TierTimeouts, after one untimed warm-up, on the corpus and thresholds of orchestration-meta-layer-cascade.md. The pipeline's shared CachingEmbeddingModel serves the question's embedding to every tier and to SearchDocuments, so SearchDocuments alone embeds nothing new. The other compositions embed through the uncached MiniLM model on every attempt: CompressingRetriever its query and every candidate sentence, BiEncoderRerankReranker its query and every candidate, and HyDERetriever's search its generated passage. That passage comes from a stub chat model that returns a new passage at once: generation is awaited network I/O that yields the loop, so the stub isolates the embedding. A timed-out CAG match keeps its worker thread running to completion, so an attempt measured right after a CAG timeout can include that overlap. Measured on commit e5bf5f0, before #168 moved these retrievers' embedding off the event loop.

## Tier latency against Concept 5's budgets, per RAG retriever

| Retriever | Tier | Attempts | Outcomes | p50 ms | p95 ms | Budget ms | Within budget |
|---|---|---|---|---|---|---|---|
| SearchDocuments | CAG | 50 | hit 30, miss 20 | 2.18 | 2.66 | 10 | 100% |
| SearchDocuments | MAG | 50 | hit 20, partial 5, miss 25 | 7.21 | 8.50 | 50 | 100% |
| SearchDocuments | RAG | 50 | hit 50 | 5.97 | 7.60 | 2000 | 100% |
| CompressingRetriever | CAG | 50 | hit 30, miss 20 | 3.86 | 4.64 | 10 | 100% |
| CompressingRetriever | MAG | 50 | timeout 50 | 99.73 | 112.93 | 50 | 0% |
| CompressingRetriever | RAG | 50 | hit 50 | 99.84 | 112.37 | 2000 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | CAG | 50 | hit 30, miss 20 | 4.09 | 4.68 | 10 | 100% |
| RerankingRetriever + BiEncoderRerankReranker | MAG | 50 | timeout 50 | 120.79 | 136.04 | 50 | 0% |
| RerankingRetriever + BiEncoderRerankReranker | RAG | 50 | hit 50 | 122.58 | 135.97 | 2000 | 100% |
| HyDERetriever | CAG | 50 | timeout 50 | 14.98 | 17.15 | 10 | 0% |
| HyDERetriever | MAG | 50 | timeout 50 | 59.34 | 64.16 | 50 | 0% |
| HyDERetriever | RAG | 50 | hit 50 | 20.63 | 63.53 | 2000 | 100% |
