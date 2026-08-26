# rag-combinations-speed-demon — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-combinations/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 10686ms / 17196ms | 0% | — | combination=speed-demon, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: baseline calls generate() with empty context deliberately (strict context-only methodology, same as every prior batch) -- expected to refuse or answer from general training regardless of what a bare model might know. Baseline's task-success figure can differ slightly run to run for the identical call: task_success_rate is computed from only the LAST of repeat_count=3 samples per question, not an average, so a small amount of run-to-run variance on borderline questions is expected and not itself evidence of a methodology difference between combinations' reports. CAVEAT 3: the qualitative judge scores both arms against the TREATMENT's own retrieved context (evaluation/application/run_comparison.py, pre-existing, tracked as #148) -- baseline can be penalized on 'groundedness' for citing facts sourced from context it genuinely retrieved but that differs from treatment's; no quality conclusion should be drawn from the qualitative table alone until #148 is fixed. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 30026ms / 77752ms | 100% | n/a output tokens, +181.0% p50 latency | combination=speed-demon, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: baseline calls generate() with empty context deliberately (strict context-only methodology, same as every prior batch) -- expected to refuse or answer from general training regardless of what a bare model might know. Baseline's task-success figure can differ slightly run to run for the identical call: task_success_rate is computed from only the LAST of repeat_count=3 samples per question, not an average, so a small amount of run-to-run variance on borderline questions is expected and not itself evidence of a methodology difference between combinations' reports. CAVEAT 3: the qualitative judge scores both arms against the TREATMENT's own retrieved context (evaluation/application/run_comparison.py, pre-existing, tracked as #148) -- baseline can be penalized on 'groundedness' for citing facts sourced from context it genuinely retrieved but that differs from treatment's; no quality conclusion should be drawn from the qualitative table alone until #148 is fixed. |

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 2 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer.

### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer.

### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain information regarding RAG.md, the 'Speed Demon' pattern, or its tradeoffs compared to Fort Knox.

### Question 6

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question

### Question 7

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

