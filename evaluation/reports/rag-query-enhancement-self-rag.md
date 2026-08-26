# rag-query-enhancement — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-query-enhancement/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 7959ms / 16528ms | 0% | — | strategy=self-rag, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. CAVEAT 2: self-rag's gate check is itself an extra generate() call on every single question, independent of what the gate decides -- even a NO decision that skips retrieval entirely still paid for that one gate call first, so self-rag is not free relative to a plain AnswerQuestion baseline on questions where it ends up retrieving anyway; its savings are specifically in the retrieval + context-assembly + larger-context generation work it skips on a NO, not in avoiding LLM calls altogether. SELF-RAG GATE LOG: across this run's 21 treatment calls (repeat_count=3 x 7 questions), the gate said YES/retrieved 15 times and NO/skipped 6 times. This is only the aggregate count -- the real, honest, question-by-question gate decisions were printed to stdout as '[self-rag gate] <question> -> <decision>' while this run executed; read that output directly rather than treating this aggregate as the full record. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 64430ms / 120337ms | 71% | n/a output tokens, +709.5% p50 latency | strategy=self-rag, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. CAVEAT 2: self-rag's gate check is itself an extra generate() call on every single question, independent of what the gate decides -- even a NO decision that skips retrieval entirely still paid for that one gate call first, so self-rag is not free relative to a plain AnswerQuestion baseline on questions where it ends up retrieving anyway; its savings are specifically in the retrieval + context-assembly + larger-context generation work it skips on a NO, not in avoiding LLM calls altogether. SELF-RAG GATE LOG: across this run's 21 treatment calls (repeat_count=3 x 7 questions), the gate said YES/retrieved 15 times and NO/skipped 6 times. This is only the aggregate count -- the real, honest, question-by-question gate decisions were printed to stdout as '[self-rag gate] <question> -> <decision>' while this run executed; read that output directly rather than treating this aggregate as the full record. |

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 3 | 3 | 1 | 2 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer

### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to your question.

### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 2 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to your question.

### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 2 | 1 | 2 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain information to answer this question.

### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 6

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 3 | 5 |
| Treatment (B) | 5 | 3 | 3 | 5 |


### Question 7

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 4 | 3 | 5 |
| Treatment (B) | 5 | 4 | 3 | 5 |

