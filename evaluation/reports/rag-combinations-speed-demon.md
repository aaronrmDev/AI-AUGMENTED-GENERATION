# rag-combinations-speed-demon — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-combinations/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 8891ms / 15093ms | 14% | — | combination=speed-demon, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: baseline calls generate() with empty context deliberately (strict context-only methodology, same as every prior batch) -- expected to refuse or answer from general training regardless of what a bare model might know. CAVEAT 3: the qualitative judge scores both arms against the TREATMENT's own retrieved context (evaluation/application/run_comparison.py, pre-existing, tracked as #148) -- baseline can be penalized on 'groundedness' for citing facts sourced from context it genuinely retrieved but that differs from treatment's; no quality conclusion should be drawn from the qualitative table alone until #148 is fixed. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 26888ms / 67775ms | 100% | n/a output tokens, +202.4% p50 latency | combination=speed-demon, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: baseline calls generate() with empty context deliberately (strict context-only methodology, same as every prior batch) -- expected to refuse or answer from general training regardless of what a bare model might know. CAVEAT 3: the qualitative judge scores both arms against the TREATMENT's own retrieved context (evaluation/application/run_comparison.py, pre-existing, tracked as #148) -- baseline can be penalized on 'groundedness' for citing facts sourced from context it genuinely retrieved but that differs from treatment's; no quality conclusion should be drawn from the qualitative table alone until #148 is fixed. |

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

- Baseline unverifiable claims: The provided context does not contain the answer to your question.

### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 2 | 1 | 4 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 2 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the information needed to answer the question.

### Question 6

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 7

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question.
