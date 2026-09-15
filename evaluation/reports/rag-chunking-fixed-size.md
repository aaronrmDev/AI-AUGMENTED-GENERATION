# rag-chunking-strategies — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-chunking-strategies/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 360 | 5035 | 9501ms / 24358ms | 0% | — | strategy=fixed-size, corpus=docs/architecture/RAG.md, 14 chunks. #147 re-measurement: qualitative judge is now Ollama/llama3.1:8b, independent of qwen3.5 (the model generating the treatment answer) -- the original same-model-family self-grading-bias risk no longer applies. Groundedness is now judged against a run-independent reference passage (queries.yaml's gold_passage) rather than each arm's own retrieved context, so groundedness is comparable across the five chunking-strategy reports for the first time. repeat_count=5 (was 3), every repeat scored for task success, not just the last. Input/output token counts are now read from Ollama's real prompt_eval_count/eval_count, not hardcoded 0. CAVEAT (still real, unrelated to the above): baseline has no retrieved context of its own (empty string), so per the #148 fix (each arm's own context still used for the unverifiable-claims check) baseline's hallucination check runs against '(none provided)' -- this does not affect groundedness, which now uses the shared reference passage for both arms. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 12744 | 16316 | 27158ms / 97165ms | 64% | +224.1% output tokens, +185.8% p50 latency | strategy=fixed-size, corpus=docs/architecture/RAG.md, 14 chunks. #147 re-measurement: qualitative judge is now Ollama/llama3.1:8b, independent of qwen3.5 (the model generating the treatment answer) -- the original same-model-family self-grading-bias risk no longer applies. Groundedness is now judged against a run-independent reference passage (queries.yaml's gold_passage) rather than each arm's own retrieved context, so groundedness is comparable across the five chunking-strategy reports for the first time. repeat_count=5 (was 3), every repeat scored for task success, not just the last. Input/output token counts are now read from Ollama's real prompt_eval_count/eval_count, not hardcoded 0. CAVEAT (still real, unrelated to the above): baseline has no retrieved context of its own (empty string), so per the #148 fix (each arm's own context still used for the unverifiable-claims check) baseline's hallucination check runs against '(none provided)' -- this does not affect groundedness, which now uses the shared reference passage for both arms. |

## Qualitative (per question)

### Question 1

Task success this question: baseline 0%, treatment 0%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 3 | 3 | 1 | 1 |
| Treatment (B) | 5 | 4 | 4 | 2 |

- Baseline answer: The provided context does not contain the answer regarding the recommended chunk size and overlap percentage.
- Treatment answer: Based on the provided context, I cannot confirm information specifically from a document called "RAG.md." The context references CLAUDE.md and docs/inputs/concepts/advanced_rag_concepts.md as the source documents, but does not mention RAG.md.

However, regarding chunk size and overlap recommendations for Fixed Size (or generally) chunking strategies that ARE in the context:

- **Starting token sizes:** 256, 512, and 1024 tokens are suggested starting points
- **Overlap percentage:** 10-20% overlap between chunks is recommended so information near a boundary isn't lost

These recommendations appear in the chunking strategies section of advanced_rag_concepts.md, but the document name "RAG.md" mentioned in your question is not confirmed in the provided context.

- Baseline unverifiable claims: The provided context does not contain the answer regarding the recommended chunk size and overlap percentage.

### Question 2

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 3 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to your question about Semantic Chunking or RAG.md.
- Treatment answer: According to the provided context, Semantic Chunking requires an **embedding model** to group content by semantic similarity rather than by position in the document. It is called the most complex of the six chunking strategies because it incurs "the cost of needing an embedding model to do the grouping."

- Baseline unverifiable claims: The provided context does not contain the answer to your question about Semantic Chunking or RAG.md.

### Question 3

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to this question.
- Treatment answer: The provided context does not mention a system or file named `RAG.md`; it refers to a system and document named **CLAUDE.md**. For the "Production Grade" pipeline archetype described in the text (for CLAUDE.md), the techniques involved are:

*   **Hybrid Search**
*   **Reranking**
*   **CRAG**
*   **Context Compression**

The source describes this archetype as "**the gold standard for production RAG systems**."

- Baseline unverifiable claims: The provided context does not contain the answer to this question.

### Question 4

Task success this question: baseline 0%, treatment 40%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain information regarding Parent Document Retrieval's expected impact figure per RAG.md.
- Treatment answer: Based on the provided context, **Parent Document Retrieval**'s expected impact figure, measured in isolation, is a **+15-20% gain in completeness**. This information appears in the section "How the five pipeline stages fit together," which aligns with the concept's listed Core Value of "Completeness" in the "The nine concepts at a glance" table.


### Question 5

Task success this question: baseline 0%, treatment 80%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 1 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain information about tradeoffs made by Sliding Window chunking according to RAG.md.
- Treatment answer: Per the provided context, Sliding Window chunking makes the tradeoff of multiplying how much has to be stored (because the windows overlap) in exchange for giving good coverage and keeping the narrative flow intact across chunk boundaries.

- Baseline unverifiable claims: The provided context does not contain information about tradeoffs made by Sliding Window chunking according to RAG.md.
