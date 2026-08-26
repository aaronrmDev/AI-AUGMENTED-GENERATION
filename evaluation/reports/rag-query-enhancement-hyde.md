# rag-query-enhancement — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-query-enhancement/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 10046ms / 15232ms | 0% | — | strategy=hyde, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. CAVEAT 2: this strategy issues one extra LLM call per query before retrieval even runs (query-variant generation for multi-query, hypothetical-answer generation for hyde), on top of the one generate() call every strategy already pays for the final answer -- the rag-hybrid-reranking batch's rerank-llm finding already showed an extra reasoning-model call in the loop can add substantial latency; expect the same class of effect here. CAVEAT 3 (RE-RUN): this batch's final review caught HyDE and Self-RAG's no-context answer silently degrading into a refusal under ChatModel.generate()'s hardcoded RAG-answering system prompt -- the first published measurement for this strategy was invalid. Fixed by adding ChatModel.complete() (no system prompt) and rewiring HyDE, Multi-Query's variant generation, and Self-RAG's gate + no-context answer to use it; this report is the corrected re-run against that fix, not the original. CAVEAT 4 (BASELINE METHODOLOGY): this script's baseline() deliberately still calls chat_model.generate(question, context=""), the same RAG-answering-system-prompt call the bug above was found in -- left unchanged rather than fixed, because baseline is meant to measure 'strict context-only answering with nothing retrieved', consistent with every prior batch's baseline in this project. The real, disclosed consequence: baseline is expected to refuse even the 2 general-knowledge questions (Q6, Q7) that a bare, unconstrained model could trivially answer, because the system prompt instructs it to say so when 'the context' (empty) doesn't contain the answer. Baseline's 0% task success and its apparently-high qualitative judge scores on Q6/Q7 both reflect this -- the judge scores a confident, well-formed refusal as coherent and 'grounded' (it makes no false claims), which is a real, separate judge-prompt limitation, not evidence baseline actually answered correctly. Treat baseline as 'no context provided', not 'no model knowledge available'. HYDE QUERY LOG: across this run's 21 treatment calls, 0 of the generated hypothetical answers looked like a refusal (contained 'does not contain' or 'provided context') rather than an actual invented answer -- 0 is the expected/healthy value post-fix. Sample generated hypothetical answer: 'According to RAG.md, Multi-Query Retrieval differentiates itself through parallel semantic paraphrase generation rather than lexical augmentation as seen in Query Expansion, and its specific recall gain figure is reported at +41.3%.'. Full per-question log was printed to stdout as '[hyde query] <question> -> <hypothetical answer used as the search query>' while this run executed. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 0 | 0 | 43237ms / 68214ms | 57% | n/a output tokens, +330.4% p50 latency | strategy=hyde, corpus=docs/architecture/RAG.md, 14 chunks. CAVEAT 1: qualitative judge is Ollama/qwen3.5, not Claude (no API credit balance at run time) -- same model family judging its own treatment output, a real self-grading-bias risk; re-run with ClaudeJudge once credits exist before treating these judge scores as final. CAVEAT 2: this strategy issues one extra LLM call per query before retrieval even runs (query-variant generation for multi-query, hypothetical-answer generation for hyde), on top of the one generate() call every strategy already pays for the final answer -- the rag-hybrid-reranking batch's rerank-llm finding already showed an extra reasoning-model call in the loop can add substantial latency; expect the same class of effect here. CAVEAT 3 (RE-RUN): this batch's final review caught HyDE and Self-RAG's no-context answer silently degrading into a refusal under ChatModel.generate()'s hardcoded RAG-answering system prompt -- the first published measurement for this strategy was invalid. Fixed by adding ChatModel.complete() (no system prompt) and rewiring HyDE, Multi-Query's variant generation, and Self-RAG's gate + no-context answer to use it; this report is the corrected re-run against that fix, not the original. CAVEAT 4 (BASELINE METHODOLOGY): this script's baseline() deliberately still calls chat_model.generate(question, context=""), the same RAG-answering-system-prompt call the bug above was found in -- left unchanged rather than fixed, because baseline is meant to measure 'strict context-only answering with nothing retrieved', consistent with every prior batch's baseline in this project. The real, disclosed consequence: baseline is expected to refuse even the 2 general-knowledge questions (Q6, Q7) that a bare, unconstrained model could trivially answer, because the system prompt instructs it to say so when 'the context' (empty) doesn't contain the answer. Baseline's 0% task success and its apparently-high qualitative judge scores on Q6/Q7 both reflect this -- the judge scores a confident, well-formed refusal as coherent and 'grounded' (it makes no false claims), which is a real, separate judge-prompt limitation, not evidence baseline actually answered correctly. Treat baseline as 'no context provided', not 'no model knowledge available'. HYDE QUERY LOG: across this run's 21 treatment calls, 0 of the generated hypothetical answers looked like a refusal (contained 'does not contain' or 'provided context') rather than an actual invented answer -- 0 is the expected/healthy value post-fix. Sample generated hypothetical answer: 'According to RAG.md, Multi-Query Retrieval differentiates itself through parallel semantic paraphrase generation rather than lexical augmentation as seen in Query Expansion, and its specific recall gain figure is reported at +41.3%.'. Full per-question log was printed to stdout as '[hyde query] <question> -> <hypothetical answer used as the search query>' while this run executed. |

## Qualitative (per question)

### Question 1

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 2 | 1 | 2 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to your question.

### Question 2

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 1 | 2 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 3

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 4 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 4

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 5

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |


### Question 6

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 3 | 5 |
| Treatment (B) | 5 | 5 | 3 | 5 |


### Question 7

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 3 | 5 |
| Treatment (B) | 5 | 3 | 3 | 5 |

