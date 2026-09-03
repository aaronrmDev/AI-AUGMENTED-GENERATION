# rag-crag — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-crag/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 17499ms / 75416ms | 86% | — | corpus=docs/architecture/RAG.md, 14 chunks. Baseline = plain vector search (SearchDocuments), Treatment = CorrectiveRetriever-wrapped search (relevance-filters retrieved results, single-shot corrected re-search when nothing passes). CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: this treatment issues one extra complete() call per retrieved result (relevance evaluation, run concurrently via asyncio.gather) plus, on a correction, one more for the refined query -- expect the same class of latency overhead every prior batch's extra-LLM-call techniques showed. CAVEAT 3: this corpus is a single 14-chunk document, which makes it hard to manufacture genuinely irrelevant top-k results for a well-targeted query -- questions 6-7 are a deliberate stress test of CRAG's value on vague, non-technical phrasing, but a null result there (no measurable difference from baseline) is a legitimate, disclosed finding about this corpus's small size, not evidence CorrectiveRetriever itself is broken. CAVEAT 4 (JUDGE HARNESS LIMITATION, PROJECT-WIDE, NOT SPECIFIC TO THIS BATCH): this batch's final review found evaluation/application/run_comparison.py's qualitative judge call scores BOTH the baseline and treatment answers against the TREATMENT's retrieved context only -- so whenever treatment's retrieved chunks differ from baseline's (which CorrectiveRetriever's filtering/correction makes likely by design), the baseline gets penalized on 'groundedness' for citing facts sourced from context it never actually retrieved, while treatment is graded against literally its own input and structurally cannot lose. This affects every report this project has published (treatment flagged for unverifiable claims 0 times in 17 of 18 prior reports), not just this one --any 'treatment has fewer unverifiable claims than baseline' reading from this report's qualitative table should be treated as an artifact of this harness limitation, not a real finding, until the harness itself is fixed. CORRECTION FIRING RATE: across this run's 21 treatment calls (repeat_count=3 x 7 questions), correction fired 3 times (14%). Real, honest, per-question decisions were printed live as '[crag correction] <question> -> FIRED/not fired'; read that output directly rather than treating this aggregate as the full record. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 94148ms / 163174ms | 86% | n/a output tokens, +438.0% p50 latency | corpus=docs/architecture/RAG.md, 14 chunks. Baseline = plain vector search (SearchDocuments), Treatment = CorrectiveRetriever-wrapped search (relevance-filters retrieved results, single-shot corrected re-search when nothing passes). CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: this treatment issues one extra complete() call per retrieved result (relevance evaluation, run concurrently via asyncio.gather) plus, on a correction, one more for the refined query -- expect the same class of latency overhead every prior batch's extra-LLM-call techniques showed. CAVEAT 3: this corpus is a single 14-chunk document, which makes it hard to manufacture genuinely irrelevant top-k results for a well-targeted query -- questions 6-7 are a deliberate stress test of CRAG's value on vague, non-technical phrasing, but a null result there (no measurable difference from baseline) is a legitimate, disclosed finding about this corpus's small size, not evidence CorrectiveRetriever itself is broken. CAVEAT 4 (JUDGE HARNESS LIMITATION, PROJECT-WIDE, NOT SPECIFIC TO THIS BATCH): this batch's final review found evaluation/application/run_comparison.py's qualitative judge call scores BOTH the baseline and treatment answers against the TREATMENT's retrieved context only -- so whenever treatment's retrieved chunks differ from baseline's (which CorrectiveRetriever's filtering/correction makes likely by design), the baseline gets penalized on 'groundedness' for citing facts sourced from context it never actually retrieved, while treatment is graded against literally its own input and structurally cannot lose. This affects every report this project has published (treatment flagged for unverifiable claims 0 times in 17 of 18 prior reports), not just this one --any 'treatment has fewer unverifiable claims than baseline' reading from this report's qualitative table should be treated as an artifact of this harness limitation, not a real finding, until the harness itself is fixed. CORRECTION FIRING RATE: across this run's 21 treatment calls (repeat_count=3 x 7 questions), correction fired 3 times (14%). Real, honest, per-question decisions were printed live as '[crag correction] <question> -> FIRED/not fired'; read that output directly rather than treating this aggregate as the full record. |

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 4 | 4 | 2 |
| Treatment (B) | 5 | 5 | 4 | 5 |

- Baseline unverifiable claims: "Self-RAG has a '30-50% reduction in retrieval costs'"

### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 6

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 4 | 5 |
| Treatment (B) | 5 | 5 | 4 | 5 |

- Treatment unverifiable claims: Corrective Retrieval-Augmented Generation (as full expansion of CRAG) - this abbreviation expansion is not explicitly in the provided context

### Question 7

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 3 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: Multi-Query Retrieval... Much higher recall gain compared to Query Expansion, produces queries that are "Diverse from each other" with "Much higher" recall gain
