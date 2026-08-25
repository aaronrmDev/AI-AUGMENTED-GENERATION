# rag-hybrid-reranking — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-hybrid-reranking/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 10200ms / 20645ms | 0% | — | strategy=hybrid, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. CAVEAT 2: candidate_k=20 exceeds this corpus's 14 chunks, so every retrieval arm always returns the full corpus in this run -- Hybrid Search's recall-union property and Reranking's candidate-narrowing were not exercised at the scale they're meant for; this measures reordering over the whole corpus, not filtering a wider pool. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 15847ms / 66948ms | 80% | n/a output tokens, +55.4% p50 latency | strategy=hybrid, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. CAVEAT 2: candidate_k=20 exceeds this corpus's 14 chunks, so every retrieval arm always returns the full corpus in this run -- Hybrid Search's recall-union property and Reranking's candidate-narrowing were not exercised at the scale they're meant for; this measures reordering over the whole corpus, not filtering a wider pool. |

> **Caveat on the qualitative scores below:** each cell is a single (n=1)
> draw from a self-grading judge -- Ollama/qwen3.5, the same model family
> generating the treatment's own answers, not an independent judge -- called
> once per question, not averaged across this run's repeats. As with every
> prior batch's qualitative scores in this evaluation harness, these numbers
> are not statistically usable for ranking strategies against each other; the
> task-success and latency numbers above are the load-bearing evidence from
> this batch. Tracked as issue #147.

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 3 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to your question.

### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer.
