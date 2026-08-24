# rag-chunking-strategies — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-chunking-strategies/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 7587ms / 21860ms | 0% | — | strategy=fixed-size, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 15739ms / 69884ms | 60% | n/a output tokens, +107.5% p50 latency | strategy=fixed-size, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. |

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 3 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 1 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 3 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to your question.

### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 4 | 3 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: Parent Document Retrieval does not have an expected impact figure in the provided context

### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 2 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

