# rag-hybrid-reranking — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-hybrid-reranking/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 10116ms / 13696ms | 0% | — | strategy=rerank-cross-encoder, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 25524ms / 94248ms | 80% | n/a output tokens, +152.3% p50 latency | strategy=rerank-cross-encoder, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. |

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

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 2 | 3 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer

### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 1 | 3 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to your question.

### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 3 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer.
