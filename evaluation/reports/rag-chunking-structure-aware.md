# rag-chunking-strategies — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-chunking-strategies/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 360 | 2430 | 8198ms / 13636ms | 0% | — | strategy=structure-aware, corpus=docs/architecture/RAG.md, 18 chunks. #147 re-measurement: qualitative judge is now Ollama/llama3.1:8b, independent of qwen3.5 (the model generating the treatment answer) -- the original same-model-family self-grading-bias risk no longer applies. Groundedness is now judged against a run-independent reference passage (queries.yaml's gold_passage) rather than each arm's own retrieved context, so groundedness is comparable across the five chunking-strategy reports for the first time. repeat_count=5 (was 3), every repeat scored for task success, not just the last. Input/output token counts are now read from Ollama's real prompt_eval_count/eval_count, not hardcoded 0. CAVEAT (still real, unrelated to the above): baseline has no retrieved context of its own (empty string), so per the #148 fix (each arm's own context still used for the unverifiable-claims check) baseline's hallucination check runs against '(none provided)' -- this does not affect groundedness, which now uses the shared reference passage for both arms. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 9713 | 11788 | 33017ms / 76168ms | 100% | +385.1% output tokens, +302.7% p50 latency | strategy=structure-aware, corpus=docs/architecture/RAG.md, 18 chunks. #147 re-measurement: qualitative judge is now Ollama/llama3.1:8b, independent of qwen3.5 (the model generating the treatment answer) -- the original same-model-family self-grading-bias risk no longer applies. Groundedness is now judged against a run-independent reference passage (queries.yaml's gold_passage) rather than each arm's own retrieved context, so groundedness is comparable across the five chunking-strategy reports for the first time. repeat_count=5 (was 3), every repeat scored for task success, not just the last. Input/output token counts are now read from Ollama's real prompt_eval_count/eval_count, not hardcoded 0. CAVEAT (still real, unrelated to the above): baseline has no retrieved context of its own (empty string), so per the #148 fix (each arm's own context still used for the unverifiable-claims check) baseline's hallucination check runs against '(none provided)' -- this does not affect groundedness, which now uses the shared reference passage for both arms. |

## Qualitative (per question)

### Question 1

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain information about RAG.md or its recommendations for fixed size chunking (chunk size and overlap percentage) in a beginner implementation phase. According to the instructions, if the context lacks this information, it should be stated plainly rather than guessed.
- Treatment answer: According to the provided context, for the beginner implementation phase (beginner path), the source recommends **Fixed Size at 512 tokens with 10% overlap**.


### Question 2

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 1 | 1 | 1 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to this question.
- Treatment answer: According to the provided text, Semantic Chunking requires **an embedding model** in order to group content by semantic similarity rather than by position in the document. The document describes this as making it **the most complex of the six to implement** due to this cost ("at the cost of needing an embedding model to do the grouping").

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 3

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 4 | 5 |

- Baseline answer: The provided context does not contain the answer to your question.
- Treatment answer: Based on the provided context, the **"Production Grade" pipeline archetype** is described as consisting of **Hybrid Search**, **Reranking**, **CRAG**, and **Context Compression**. The source describes this archetype using the phrase "**the gold standard for production RAG systems**".

(Note: The provided text refers to this paradigm in relation to "CLAUDE.md's RAG paradigm" and the document header "# RAG: Retrieval-Augmented Generation", but it does not explicitly use the filename "RAG.md" within the body of the text.)


### Question 4

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the information regarding Parent Document Retrieval's expected impact figure per RAG.md.
- Treatment answer: Parent Document Retrieval's expected impact in isolation is a +15-20% gain in completeness.


### Question 5

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain information about Sliding Window chunking or its tradeoffs as described in RAG.md.
- Treatment answer: According to the provided context, Sliding Window chunking makes the tradeoff that while it "gives good coverage and keeps the narrative flow intact across chunk boundaries," it does so at the cost of a storage increase where it "multiplies how much has to be stored since the windows overlap."

