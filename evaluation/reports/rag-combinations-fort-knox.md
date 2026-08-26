# rag-combinations-fort-knox — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-combinations/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 9307ms / 13941ms | 0% | — | combination=fort-knox, corpus=docs/architecture/RAG.md, 41 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: baseline calls generate() with empty context deliberately (strict context-only methodology, same as every prior batch) -- expected to refuse or answer from general training regardless of what a bare model might know. CAVEAT 3: the qualitative judge scores both arms against the TREATMENT's own retrieved context (evaluation/application/run_comparison.py, pre-existing, tracked as #148) -- baseline can be penalized on 'groundedness' for citing facts sourced from context it genuinely retrieved but that differs from treatment's; no quality conclusion should be drawn from the qualitative table alone until #148 is fixed. CAVEAT 4: this combination wraps CorrectiveRetriever -- correction firing rate is instrumented and reported below, not assumed. CAVEAT 5 (FORT KNOX ONLY): BM25 keyword search reads every saved chunk, including parent chunks UploadDocumentWithParents also saves; a parent chunk surfaced by the keyword arm is silently skipped by ParentDocumentRetriever (its parent_id is None), a real disclosed interaction documented in the design spec, not engineered around. CORRECTION FIRING RATE: across this run's 21 treatment calls (repeat_count=3 x 7 questions), correction fired 0 times (0%). Real, per-question decisions were printed live as '[fort-knox correction] <question> -> FIRED/not fired'; read that output directly rather than treating this aggregate as the full record. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 219897ms / 282440ms | 86% | n/a output tokens, +2262.8% p50 latency | combination=fort-knox, corpus=docs/architecture/RAG.md, 41 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: baseline calls generate() with empty context deliberately (strict context-only methodology, same as every prior batch) -- expected to refuse or answer from general training regardless of what a bare model might know. CAVEAT 3: the qualitative judge scores both arms against the TREATMENT's own retrieved context (evaluation/application/run_comparison.py, pre-existing, tracked as #148) -- baseline can be penalized on 'groundedness' for citing facts sourced from context it genuinely retrieved but that differs from treatment's; no quality conclusion should be drawn from the qualitative table alone until #148 is fixed. CAVEAT 4: this combination wraps CorrectiveRetriever -- correction firing rate is instrumented and reported below, not assumed. CAVEAT 5 (FORT KNOX ONLY): BM25 keyword search reads every saved chunk, including parent chunks UploadDocumentWithParents also saves; a parent chunk surfaced by the keyword arm is silently skipped by ParentDocumentRetriever (its parent_id is None), a real disclosed interaction documented in the design spec, not engineered around. CORRECTION FIRING RATE: across this run's 21 treatment calls (repeat_count=3 x 7 questions), correction fired 0 times (0%). Real, per-question decisions were printed live as '[fort-knox correction] <question> -> FIRED/not fired'; read that output directly rather than treating this aggregate as the full record. |

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 3 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: the text describing the 'RAG.md' file and its contents is missing from the input

### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the information needed to answer this question.

### Question 6

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 7

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

