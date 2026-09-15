# rag-chunking-strategies — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-chunking-strategies/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 360 | 4334 | 9822ms / 19633ms | 0% | — | strategy=sliding-window, corpus=docs/architecture/RAG.md, 24 chunks. #147 re-measurement: qualitative judge is now Ollama/llama3.1:8b, independent of qwen3.5 (the model generating the treatment answer) -- the original same-model-family self-grading-bias risk no longer applies. Groundedness is now judged against a run-independent reference passage (queries.yaml's gold_passage) rather than each arm's own retrieved context, so groundedness is comparable across the five chunking-strategy reports for the first time. repeat_count=5 (was 3), every repeat scored for task success, not just the last. Input/output token counts are now read from Ollama's real prompt_eval_count/eval_count, not hardcoded 0. CAVEAT (still real, unrelated to the above): baseline has no retrieved context of its own (empty string), so per the #148 fix (each arm's own context still used for the unverifiable-claims check) baseline's hallucination check runs against '(none provided)' -- this does not affect groundedness, which now uses the shared reference passage for both arms. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 13328 | 10379 | 24948ms / 82250ms | 100% | +139.5% output tokens, +154.0% p50 latency | strategy=sliding-window, corpus=docs/architecture/RAG.md, 24 chunks. #147 re-measurement: qualitative judge is now Ollama/llama3.1:8b, independent of qwen3.5 (the model generating the treatment answer) -- the original same-model-family self-grading-bias risk no longer applies. Groundedness is now judged against a run-independent reference passage (queries.yaml's gold_passage) rather than each arm's own retrieved context, so groundedness is comparable across the five chunking-strategy reports for the first time. repeat_count=5 (was 3), every repeat scored for task success, not just the last. Input/output token counts are now read from Ollama's real prompt_eval_count/eval_count, not hardcoded 0. CAVEAT (still real, unrelated to the above): baseline has no retrieved context of its own (empty string), so per the #148 fix (each arm's own context still used for the unverifiable-claims check) baseline's hallucination check runs against '(none provided)' -- this does not affect groundedness, which now uses the shared reference passage for both arms. |

## Qualitative (per question)

### Question 1

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to the question.
- Treatment answer: According to the provided context, the source recommends a chunk size of **512 tokens** with an overlap percentage of **10%** as the starting point for Fixed Size chunking in the beginner implementation phase.

- Baseline unverifiable claims: The provided context does not contain the answer to the question.

### Question 2

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 1 | 1 | 1 | 5 |
| Treatment (B) | 3 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to your question.
- Treatment answer: Based on the provided context, Semantic Chunking requires an **embedding model** to do the grouping of content. The text states that it is considered "the most complex of the six to implement" specifically at the cost of needing this embedding model.


### Question 3

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 1 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to this question.
- Treatment answer: Based on the provided context, RAG.md's "Production Grade" pipeline archetype chains **Hybrid Search**, **Reranking**, **CRAG**, and **Context Compression**. The source describes this archetype as "**the gold standard for production RAG systems**."

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 4

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 3 |

- Baseline answer: The provided context does not contain the answer to this question.
- Treatment answer: Based on the provided context, Parent Document Retrieval's expected impact when measured in isolation is **a +15-20% gain in completeness**. 

However, I should note that the context references `docs/inputs/concepts/advanced_rag_concepts.md` rather than a document specifically named "RAG.md." The information about Parent Document Retrieval's impact figures comes from the source document described in the context.

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 5

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 1 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to this question.
- Treatment answer: Based on the provided context, Sliding Window chunking makes a tradeoff where it "gives good coverage and keeps the narrative flow intact across chunk boundaries," but at the cost of doing so "multiplies how much has to be stored since the windows overlap." (The text attributes these descriptions to `docs/inputs/concepts/advanced_rag_concepts.md`).

- Baseline unverifiable claims: The provided context does not contain the answer to this question.
