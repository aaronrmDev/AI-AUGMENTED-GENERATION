# rag-chunking-strategies — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/rag-chunking-strategies/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 360 | 3334 | 8053ms / 18559ms | 0% | — | strategy=sentence-based, corpus=docs/architecture/RAG.md, 13 chunks. #147 re-measurement: qualitative judge is now Ollama/llama3.1:8b, independent of qwen3.5 (the model generating the treatment answer) -- the original same-model-family self-grading-bias risk no longer applies. Groundedness is now judged against a run-independent reference passage (queries.yaml's gold_passage) rather than each arm's own retrieved context, so groundedness is comparable across the five chunking-strategy reports for the first time. repeat_count=5 (was 3), every repeat scored for task success, not just the last. Input/output token counts are now read from Ollama's real prompt_eval_count/eval_count, not hardcoded 0. CAVEAT (still real, unrelated to the above): baseline has no retrieved context of its own (empty string), so per the #148 fix (each arm's own context still used for the unverifiable-claims check) baseline's hallucination check runs against '(none provided)' -- this does not affect groundedness, which now uses the shared reference passage for both arms. |
| Treatment | ✓ | ✗ | ✗ | qwen3.5, Ollama | 12344 | 10902 | 32684ms / 77883ms | 76% | +227.0% output tokens, +305.9% p50 latency | strategy=sentence-based, corpus=docs/architecture/RAG.md, 13 chunks. #147 re-measurement: qualitative judge is now Ollama/llama3.1:8b, independent of qwen3.5 (the model generating the treatment answer) -- the original same-model-family self-grading-bias risk no longer applies. Groundedness is now judged against a run-independent reference passage (queries.yaml's gold_passage) rather than each arm's own retrieved context, so groundedness is comparable across the five chunking-strategy reports for the first time. repeat_count=5 (was 3), every repeat scored for task success, not just the last. Input/output token counts are now read from Ollama's real prompt_eval_count/eval_count, not hardcoded 0. CAVEAT (still real, unrelated to the above): baseline has no retrieved context of its own (empty string), so per the #148 fix (each arm's own context still used for the unverifiable-claims check) baseline's hallucination check runs against '(none provided)' -- this does not affect groundedness, which now uses the shared reference passage for both arms. |

## Qualitative (per question)

### Question 1

Task success this question: baseline 0%, treatment 0%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to this question.
- Treatment answer: Based on the provided context, there is no specific information about "Fixed Size chunking" or a "beginner implementation phase" with RAG.md. 

The context does mention general chunk size recommendations:
- Test sizes of 256, 512, and 1024 tokens as starting points
- Use 10-20% overlap between chunks

However, these appear to be general guidance for chunking rather than specifically tied to "Fixed Size chunking" in a beginner implementation phase as referenced in the question. The context notes that six strategies were named by the project brief including "fixed," but doesn't provide specific parameters for Fixed Size chunking separate from other strategies like Sliding Window or Semantic Chunking.

The source document referenced is `advanced_rag_concepts.md`, not explicitly "RAG.md." If you're looking for those specific RAG.md beginner phase Fixed Size recommendations, they are not contained in the provided context.


### Question 2

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 1 | 1 | 1 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to the question.
- Treatment answer: Semantic Chunking requires an **embedding model** to do the grouping. It is called the most complex of the six chunking strategies because it incurs the cost of needing that embedding model for the grouping process.


### Question 3

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer.
- Treatment answer: Based on the context provided, RAG.md's **"Production Grade" pipeline archetype** consists of these four techniques:

1. **Hybrid Search** — casts the wide net by returning candidates from two different notions of relevance (semantic similarity and keyword overlap)
2. **Reranking** — narrows to true relevance after Hybrid Search provides a noisy candidate pool
3. **CRAG** — validates the result and triggers correction when needed through its corrective retrieval loop
4. **Context Compression** — optimizes what's left before it reaches the LLM by trimming redundancy and noise

The phrase RAG.md uses to describe this archetype is **"the gold standard for production RAG systems."**


### Question 4

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 1 | 1 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer.
- Treatment answer: According to the provided context, Parent Document Retrieval's expected impact in isolation is a **+15-20% gain in completeness**.

- Baseline unverifiable claims: The provided context does not contain the answer.

### Question 5

Task success this question: baseline 0%, treatment 80%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to this question.
- Treatment answer: According to the provided context, Sliding Window chunking makes the tradeoff that it **"multiplies how much has to be stored since the windows overlap"** to achieve **"good coverage and keeps the narrative flow intact across chunk boundaries."** The text attributes these descriptions to `advanced_rag_concepts.md` rather than explicitly identifying the document as "RAG.md."

