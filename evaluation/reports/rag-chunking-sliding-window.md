# rag-chunking-strategies — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-chunking-strategies/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 8680ms / 13131ms | 0% | — | strategy=sliding-window, corpus=docs/architecture/RAG.md, 24 chunks. CAVEAT: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 20370ms / 52095ms | 100% | n/a output tokens, +134.7% p50 latency | strategy=sliding-window, corpus=docs/architecture/RAG.md, 24 chunks. CAVEAT: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. |

> **Caveat on the qualitative scores below (added after the final whole-branch
> review found this empirically):** each cell is a single (n=1) draw from a
> self-grading judge -- Ollama/qwen3.5, the same model family generating the
> treatment's own answers, not an independent judge -- called once per
> question, not averaged across this run's repeats. Across all five of this
> batch's reports the treatment saturated at 5/5/5/5 on the overwhelming
> majority of question-runs, including cases where the same run's own
> quantitative task-success check failed for that question -- while the
> IDENTICAL no-RAG baseline condition (same model, same five questions)
> scored across nearly the full 1-5 range between different reports. That
> combination means these qualitative numbers are not statistically usable
> for ranking chunking strategies against each other, and should not be read
> as confirming RAG wins on every qualitative dimension -- only the
> task-success and latency numbers above are trustworthy load-bearing
> evidence from this batch. A trustworthy qualitative re-measurement needs a
> run-independent reference context (not the treatment grading itself) and
> an independent judge model (not the same family as the system under test)
> -- tracked as follow-up work, not completed in this batch.

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question

### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 3 | 5 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question

### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer.
