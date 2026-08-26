# rag-combinations-fort-knox — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-combinations/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 8897ms / 13344ms | 0% | — | combination=fort-knox, corpus=docs/architecture/RAG.md, 41 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: baseline calls generate() with empty context deliberately (strict context-only methodology, same as every prior batch) -- expected to refuse or answer from general training regardless of what a bare model might know. Baseline's task-success figure can differ slightly run to run for the identical call: task_success_rate is computed from only the LAST of repeat_count=3 samples per question, not an average, so a small amount of run-to-run variance on borderline questions is expected and not itself evidence of a methodology difference between combinations' reports. CAVEAT 3: the qualitative judge scores both arms against the TREATMENT's own retrieved context (evaluation/application/run_comparison.py, pre-existing, tracked as #148) -- baseline can be penalized on 'groundedness' for citing facts sourced from context it genuinely retrieved but that differs from treatment's; no quality conclusion should be drawn from the qualitative table alone until #148 is fixed. CAVEAT 4: this combination wraps CorrectiveRetriever -- correction firing rate is instrumented and reported below, not assumed. IMPORTANT SCOPE LIMIT (added after this batch's final review): the instrument measures ONLY whether CorrectiveRetriever's re-search fired (i.e. whether zero of top_k passed relevance review) -- it does NOT measure the relevance filter's rejection rate. A 0% correction rate is fully compatible with the filter discarding most candidates on every call, as long as at least one survives; spot-checks during review found CRAG discarding roughly 3 of every 5 reranked results on individual live calls while still reporting 0% correction. A 0% rate here also isn't directly comparable to the standalone rag-crag batch's measured 14% firing rate: that batch used a different question set (including 2 deliberately vague, non-technical diagnostic questions written specifically to stress the correction path) and wrapped CorrectiveRetriever around a plain vector search with no reranker inside it, whereas every combination here already reranks before CRAG ever sees the results, structurally lowering how often nothing passes. CAVEAT 5 (FORT KNOX ONLY): BM25 keyword search reads every saved chunk, including the 7 parent chunks UploadDocumentWithParents also saves (48 total chunks in the repository; document.chunk_count below reports only the 41 children UploadDocumentWithParents returns, not both tiers). A parent chunk surfaced by the keyword arm is silently skipped by ParentDocumentRetriever (its parent_id is None) rather than expanded or erroring. MEASURED IMPACT (added after this batch's final review measured it directly, rather than assuming it was bounded): parents are BM25's natural favorites, not a rare edge case -- at roughly 969 tokens each against children's roughly 182, they accumulate far more of BM25Plus's per-matched-term score. Measured on this batch's real corpus and question set: parents occupied 6-7 of the 7 available parent slots in BM25's own top-20 on every one of the 7 questions (47 of 49 possible appearances), with a parent's best BM25 rank landing at position 2 on all 7 questions. After Hybrid Search's RRF fusion and Reranking's cross-encoder pass, parents still made up roughly 37% of results reaching ParentDocumentRetriever in a live sample -- over a third of what CRAG and Reranking worked to surface got silently dropped at the final stage. This can degenerate a treatment sample into an empty context (if too few of the survivors are children), which AnswerQuestion then answers with no retrieved material at all -- indistinguishable from a baseline call for that one sample. Not engineered around at this stage (would require production changes to BM25KeywordSearch or ParentDocumentRetriever, out of scope for this composition-only batch); disclosed here with its real measured size instead. CORRECTION FIRING RATE: across this run's 21 treatment calls (repeat_count=3 x 7 questions), correction fired 0 times (0%). Real, per-question decisions were printed live as '[fort-knox correction] <question> -> FIRED/not fired'; read that output directly rather than treating this aggregate as the full record. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 233620ms / 313707ms | 86% | n/a output tokens, +2525.9% p50 latency | combination=fort-knox, corpus=docs/architecture/RAG.md, 41 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- self-grading-bias risk, same as every prior batch in this project. CAVEAT 2: baseline calls generate() with empty context deliberately (strict context-only methodology, same as every prior batch) -- expected to refuse or answer from general training regardless of what a bare model might know. Baseline's task-success figure can differ slightly run to run for the identical call: task_success_rate is computed from only the LAST of repeat_count=3 samples per question, not an average, so a small amount of run-to-run variance on borderline questions is expected and not itself evidence of a methodology difference between combinations' reports. CAVEAT 3: the qualitative judge scores both arms against the TREATMENT's own retrieved context (evaluation/application/run_comparison.py, pre-existing, tracked as #148) -- baseline can be penalized on 'groundedness' for citing facts sourced from context it genuinely retrieved but that differs from treatment's; no quality conclusion should be drawn from the qualitative table alone until #148 is fixed. CAVEAT 4: this combination wraps CorrectiveRetriever -- correction firing rate is instrumented and reported below, not assumed. IMPORTANT SCOPE LIMIT (added after this batch's final review): the instrument measures ONLY whether CorrectiveRetriever's re-search fired (i.e. whether zero of top_k passed relevance review) -- it does NOT measure the relevance filter's rejection rate. A 0% correction rate is fully compatible with the filter discarding most candidates on every call, as long as at least one survives; spot-checks during review found CRAG discarding roughly 3 of every 5 reranked results on individual live calls while still reporting 0% correction. A 0% rate here also isn't directly comparable to the standalone rag-crag batch's measured 14% firing rate: that batch used a different question set (including 2 deliberately vague, non-technical diagnostic questions written specifically to stress the correction path) and wrapped CorrectiveRetriever around a plain vector search with no reranker inside it, whereas every combination here already reranks before CRAG ever sees the results, structurally lowering how often nothing passes. CAVEAT 5 (FORT KNOX ONLY): BM25 keyword search reads every saved chunk, including the 7 parent chunks UploadDocumentWithParents also saves (48 total chunks in the repository; document.chunk_count below reports only the 41 children UploadDocumentWithParents returns, not both tiers). A parent chunk surfaced by the keyword arm is silently skipped by ParentDocumentRetriever (its parent_id is None) rather than expanded or erroring. MEASURED IMPACT (added after this batch's final review measured it directly, rather than assuming it was bounded): parents are BM25's natural favorites, not a rare edge case -- at roughly 969 tokens each against children's roughly 182, they accumulate far more of BM25Plus's per-matched-term score. Measured on this batch's real corpus and question set: parents occupied 6-7 of the 7 available parent slots in BM25's own top-20 on every one of the 7 questions (47 of 49 possible appearances), with a parent's best BM25 rank landing at position 2 on all 7 questions. After Hybrid Search's RRF fusion and Reranking's cross-encoder pass, parents still made up roughly 37% of results reaching ParentDocumentRetriever in a live sample -- over a third of what CRAG and Reranking worked to surface got silently dropped at the final stage. This can degenerate a treatment sample into an empty context (if too few of the survivors are children), which AnswerQuestion then answers with no retrieved material at all -- indistinguishable from a baseline call for that one sample. Not engineered around at this stage (would require production changes to BM25KeywordSearch or ParentDocumentRetriever, out of scope for this composition-only batch); disclosed here with its real measured size instead. CORRECTION FIRING RATE: across this run's 21 treatment calls (repeat_count=3 x 7 questions), correction fired 0 times (0%). Real, per-question decisions were printed live as '[fort-knox correction] <question> -> FIRED/not fired'; read that output directly rather than treating this aggregate as the full record. |

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
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
| Baseline (A) | 5 | 1 | 1 | 2 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer

### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 3 | 5 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 6

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain information about RAG.md or Parent Document Retrieval

### Question 7

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

