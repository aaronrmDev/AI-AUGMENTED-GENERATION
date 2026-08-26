# rag-crag — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-crag/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 17959ms / 45942ms | 86% | — | corpus=docs/architecture/RAG.md, 14 chunks. Baseline = plain vector search (SearchDocuments), Treatment = CorrectiveRetriever-wrapped search (relevance-filters retrieved results, single-shot corrected re-search on a majority failure). CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: this treatment issues one extra complete() call per retrieved result (relevance evaluation) plus, on a correction, one more for the refined query -- expect the same class of latency overhead every prior batch's extra-LLM-call techniques showed. CAVEAT 3: this corpus is a single 14-chunk document, which makes it hard to manufacture genuinely irrelevant top-k results for a well-targeted query -- questions 6-7 are a deliberate stress test of CRAG's value on vague, non-technical phrasing, but a null result there (no measurable difference from baseline) is a legitimate, disclosed finding about this corpus's small size, not evidence CorrectiveRetriever itself is broken. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 119193ms / 167518ms | 86% | n/a output tokens, +563.7% p50 latency | corpus=docs/architecture/RAG.md, 14 chunks. Baseline = plain vector search (SearchDocuments), Treatment = CorrectiveRetriever-wrapped search (relevance-filters retrieved results, single-shot corrected re-search on a majority failure). CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: this treatment issues one extra complete() call per retrieved result (relevance evaluation) plus, on a correction, one more for the refined query -- expect the same class of latency overhead every prior batch's extra-LLM-call techniques showed. CAVEAT 3: this corpus is a single 14-chunk document, which makes it hard to manufacture genuinely irrelevant top-k results for a well-targeted query -- questions 6-7 are a deliberate stress test of CRAG's value on vague, non-technical phrasing, but a null result there (no measurable difference from baseline) is a legitimate, disclosed finding about this corpus's small size, not evidence CorrectiveRetriever itself is broken. |

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 3 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context describes CLAUDE.md's RAG paradigm rather than a separate 'RAG.md' document.

### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 4 | 3 | 3 |
| Treatment (B) | 5 | 5 | 4 | 5 |

- Baseline unverifiable claims: Self-RAG (30-50% reduction in retrieval costs)

### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 6

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 3 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: Self-RAG acts as a gate where the LLM decides 'YES or NO' on whether to search at all

### Question 7

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 4 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

