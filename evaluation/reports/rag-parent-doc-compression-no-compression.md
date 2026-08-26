# rag-parent-doc-compression — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-parent-doc-compression/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 10490ms / 17520ms | 0% | — | strategy=no-compression, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. CAVEAT 2 (corrected after the final whole-branch review found the original wording overclaimed): candidate_k-style widen-then-narrow doesn't apply here, but parent expansion creates its own confound on this small 7-parent corpus -- see the 'no-compression' control arm's report for the isolated effect. CAVEAT 3: per-strategy latency deltas across this batch's runs are dominated by Ollama generation variance at repeat_count=3 (observed p95/p50 ratios of 2.2x-6.0x on 15 samples each) and should not be read as a precise per-technique cost; only large, consistent differences are meaningful at this sample size. Context tokens actually sent to the model this run: mean=2372, range=2243-2564 (computed from the real retrieved context, not the chat model's own usage reporting -- OllamaChatModel discards that, tracked in #147). |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 13923ms / 67018ms | 20% | n/a output tokens, +32.7% p50 latency | strategy=no-compression, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. CAVEAT 2 (corrected after the final whole-branch review found the original wording overclaimed): candidate_k-style widen-then-narrow doesn't apply here, but parent expansion creates its own confound on this small 7-parent corpus -- see the 'no-compression' control arm's report for the isolated effect. CAVEAT 3: per-strategy latency deltas across this batch's runs are dominated by Ollama generation variance at repeat_count=3 (observed p95/p50 ratios of 2.2x-6.0x on 15 samples each) and should not be read as a precise per-technique cost; only large, consistent differences are meaningful at this sample size. Context tokens actually sent to the model this run: mean=2372, range=2243-2564 (computed from the real retrieved context, not the chat model's own usage reporting -- OllamaChatModel discards that, tracked in #147). |

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 3 | 4 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 2 | 2 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 2 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question

### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 3 | 4 |
| Treatment (B) | 5 | 5 | 5 | 5 |

