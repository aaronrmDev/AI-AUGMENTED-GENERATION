# rag-chunking-strategies — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-chunking-strategies/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 360 | 3872 | 8985ms / 14472ms | 0% | — | strategy=semantic, corpus=docs/architecture/RAG.md, 24 chunks. #147 re-measurement: qualitative judge is now Ollama/llama3.1:8b, independent of qwen3.5 (the model generating the treatment answer) -- the original same-model-family self-grading-bias risk no longer applies. Groundedness is now judged against a run-independent reference passage (queries.yaml's gold_passage) rather than each arm's own retrieved context, so groundedness is comparable across the five chunking-strategy reports for the first time. repeat_count=5 (was 3), every repeat scored for task success, not just the last. Input/output token counts are now read from Ollama's real prompt_eval_count/eval_count, not hardcoded 0. CAVEAT (still real, unrelated to the above): baseline has no retrieved context of its own (empty string), so per the #148 fix (each arm's own context still used for the unverifiable-claims check) baseline's hallucination check runs against '(none provided)' -- this does not affect groundedness, which now uses the shared reference passage for both arms. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 10150 | 13811 | 49175ms / 75254ms | 56% | +256.7% output tokens, +447.3% p50 latency | strategy=semantic, corpus=docs/architecture/RAG.md, 24 chunks. #147 re-measurement: qualitative judge is now Ollama/llama3.1:8b, independent of qwen3.5 (the model generating the treatment answer) -- the original same-model-family self-grading-bias risk no longer applies. Groundedness is now judged against a run-independent reference passage (queries.yaml's gold_passage) rather than each arm's own retrieved context, so groundedness is comparable across the five chunking-strategy reports for the first time. repeat_count=5 (was 3), every repeat scored for task success, not just the last. Input/output token counts are now read from Ollama's real prompt_eval_count/eval_count, not hardcoded 0. CAVEAT (still real, unrelated to the above): baseline has no retrieved context of its own (empty string), so per the #148 fix (each arm's own context still used for the unverifiable-claims check) baseline's hallucination check runs against '(none provided)' -- this does not affect groundedness, which now uses the shared reference passage for both arms. |

## Qualitative (per question)

### Question 1

Task success this question: baseline 0%, treatment 0%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to this question.
- Treatment answer: The provided context does not contain information about a document named "RAG.md" or a "beginner implementation phase." The text references the document `docs/inputs/concepts/advanced_rag_concepts.md` and discusses chunking guidance in the context of production systems rather than a beginner implementation phase. While it does recommend testing sizes of 256, 512, and 1024 tokens with a 10-20% overlap for this stage generally, the specific labels requested are not present in the provided text.


### Question 2

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 1 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain information about what Semantic Chunking requires to group content or why RAG.md considers it the most complex strategy.
- Treatment answer: Based on the provided text, Semantic Chunking requires an **embedding model** to group content by semantic similarity rather than position in the document. The text states that this requirement comes at a cost which makes it "the most complex of the six [strategies] to implement." (Note: The context specifically references `docs/inputs/concepts/advanced_rag_concepts.md`, not a file named `RAG.md`).

- Baseline unverifiable claims: The provided context does not contain information about what Semantic Chunking requires to group content or why RAG.md considers it the most complex strategy.

### Question 3

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to this question.
- Treatment answer: Based on the context, RAG.md's **"Production Grade"** pipeline archetype is made up of these four techniques:

1. **Hybrid Search** (casts the wide net)
2. **Reranking** (narrows it to true relevance)
3. **CRAG** (validates the result and triggers correction when needed)
4. **Context Compression** (optimizes what's left before it reaches the LLM)

The phrase RAG.md uses to describe this archetype is **"the gold standard for production RAG systems."**


### Question 4

Task success this question: baseline 0%, treatment 0%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to your question regarding Parent Document Retrieval's expected impact figure.
- Treatment answer: The provided context does not contain an expected impact figure for Parent Document Retrieval measured in isolation. While it lists impact figures for other techniques—such as Chunking Strategies (+10-20% gain in relevance), Self-RAG (30-50% reduction in retrieval costs, with specific examples ranging from -75% to -50% token usage), and CRAG (40-60% reduction in hallucinations)—no quantified metric for Parent Document Retrieval's impact in isolation is provided.


### Question 5

Task success this question: baseline 0%, treatment 80%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer.
- Treatment answer: Per the provided context, Sliding Window chunking trades off storage efficiency by "multipl[y]ing how much has to be stored since the windows overlap," whereas it provides good coverage and keeps the narrative flow intact across chunk boundaries.

- Baseline unverifiable claims: The provided context does not contain the answer.
