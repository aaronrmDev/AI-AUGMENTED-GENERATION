# orchestration-meta-layer — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/orchestration-meta-layer/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 1235 | 8948 | 8366ms / 34632ms | 78% | — | Baseline = RAG-only AnswerQuestion (top_k=3). Treatment = UnifiedAnswerQuestion routed by the MiniLM prototype classifier, with the CAG/MAG/RAG cascade under generous timeouts, the 128K budget allocator, and a real session budget record. CAVEAT 1: judge and generator are both qwen3.5 (self-grading risk, as in every earlier batch). CAVEAT 2: CAG is a CPU distilgpt2 proxy whose lookup, not generation speed, is what the treatment exercises. CAVEAT 3: question 1 is a static policy question, so Concept 1's rule sends it to CAG, which holds the superseded thirty-day policy in this corpus; when CAG answers it alone, the stale value is what the model sees. That is a Sync Mixer failure routing cannot fix, and it is reported, not relabeled. ROUTES (last repeat): 'What is the return window for unopened items?': routed ['cag', 'mag', 'rag'], attempts [('cag', 'hit'), ('mag', 'miss'), ('rag', 'hit')]; 'What changed in the return policy today?': routed ['cag', 'rag'], attempts [('cag', 'hit'), ('rag', 'hit')]; 'How long does standard shipping take?': routed ['cag', 'rag'], attempts [('cag', 'hit'), ('rag', 'hit')]; 'How do I set up an account?': routed ['cag'], attempts [('cag', 'hit')]; 'How long is the product warranty?': routed ['cag'], attempts [('cag', 'hit')]; 'Are the blue running shoes in stock today?': routed ['rag'], attempts [('rag', 'hit')]; 'Is there a sale on backpacks today?': routed ['cag', 'mag', 'rag'], attempts [('cag', 'miss'), ('mag', 'miss'), ('rag', 'hit')]; 'What shipping speed did I say I prefer?': routed ['cag', 'rag'], attempts [('cag', 'hit'), ('rag', 'hit')]; 'What shoe size do I wear?': routed ['cag', 'rag'], attempts [('cag', 'miss'), ('rag', 'hit')]; 'Are the running shoes in my size in stock right now?': routed ['cag', 'rag'], attempts [('cag', 'miss'), ('rag', 'hit')] |
| Treatment | ✓ | ✓ | ✓ | qwen3.5, Ollama | 1338 | 11868 | 8515ms / 43067ms | 76% | +32.6% output tokens, +1.8% p50 latency | Baseline = RAG-only AnswerQuestion (top_k=3). Treatment = UnifiedAnswerQuestion routed by the MiniLM prototype classifier, with the CAG/MAG/RAG cascade under generous timeouts, the 128K budget allocator, and a real session budget record. CAVEAT 1: judge and generator are both qwen3.5 (self-grading risk, as in every earlier batch). CAVEAT 2: CAG is a CPU distilgpt2 proxy whose lookup, not generation speed, is what the treatment exercises. CAVEAT 3: question 1 is a static policy question, so Concept 1's rule sends it to CAG, which holds the superseded thirty-day policy in this corpus; when CAG answers it alone, the stale value is what the model sees. That is a Sync Mixer failure routing cannot fix, and it is reported, not relabeled. ROUTES (last repeat): 'What is the return window for unopened items?': routed ['cag', 'mag', 'rag'], attempts [('cag', 'hit'), ('mag', 'miss'), ('rag', 'hit')]; 'What changed in the return policy today?': routed ['cag', 'rag'], attempts [('cag', 'hit'), ('rag', 'hit')]; 'How long does standard shipping take?': routed ['cag', 'rag'], attempts [('cag', 'hit'), ('rag', 'hit')]; 'How do I set up an account?': routed ['cag'], attempts [('cag', 'hit')]; 'How long is the product warranty?': routed ['cag'], attempts [('cag', 'hit')]; 'Are the blue running shoes in stock today?': routed ['rag'], attempts [('rag', 'hit')]; 'Is there a sale on backpacks today?': routed ['cag', 'mag', 'rag'], attempts [('cag', 'miss'), ('mag', 'miss'), ('rag', 'hit')]; 'What shipping speed did I say I prefer?': routed ['cag', 'rag'], attempts [('cag', 'hit'), ('rag', 'hit')]; 'What shoe size do I wear?': routed ['cag', 'rag'], attempts [('cag', 'miss'), ('rag', 'hit')]; 'Are the running shoes in my size in stock right now?': routed ['cag', 'rag'], attempts [('cag', 'miss'), ('rag', 'hit')] |

## Qualitative (per question)

### Question 1

Task success this question: baseline 100%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The return window for unopened items is within forty-five days of purchase.
- Treatment answer: Based on the retrieved documents, the return window for unopened items is forty-five days (extended today).


### Question 2

Task success this question: baseline 100%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The return window was extended to forty-five days today.
- Treatment answer: The return window was extended to forty-five days today.


### Question 3

Task success this question: baseline 100%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: Standard shipping takes five to seven business days.
- Treatment answer: Standard shipping takes five to seven business days.


### Question 4

Task success this question: baseline 100%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 5 | 3 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: To set up an account, provide your email address and choose a password of at least twelve characters. A verification email is sent immediately after registration.
- Treatment answer: To set up an account, provide your email address and choose a password of at least twelve characters. A verification email is sent immediately after registration.

- Baseline unverifiable claims: Our return policy allows customers to return unopened items within forty-five days of purchase for a full refund., The return window was extended to forty-five days today., A flash sale today takes 20% off every backpack until midnight.

### Question 5

Task success this question: baseline 100%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The product warranty is two years long.
- Treatment answer: Every product carries a two-year limited warranty covering manufacturing defects.


### Question 6

Task success this question: baseline 100%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 3 | 5 |
| Treatment (B) | 5 | 5 | 3 | 5 |

- Baseline answer: As of this morning, the blue running shoes are out of stock in sizes 9 and 10.
- Treatment answer: As of this morning, the blue running shoes are out of stock in sizes 9 and 10.


### Question 7

Task success this question: baseline 100%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: Yes, there is a flash sale today that takes 20% off every backpack until midnight.
- Treatment answer: Yes, according to the retrieved documents, there is a flash sale on every backpack today that takes 20% off until midnight.


### Question 8

Task success this question: baseline 0%, treatment 0%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 1 | 1 |
| Treatment (B) | 5 | 5 | 1 | 1 |

- Baseline answer: The provided context does not mention any preference for a specific shipping speed.
- Treatment answer: The provided context does not contain information about what shipping speed you prefer.


### Question 9

Task success this question: baseline 20%, treatment 0%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 2 | 2 |
| Treatment (B) | 5 | 5 | 2 | 2 |

- Baseline answer: The provided context does not contain information about what shoe size you wear.
- Treatment answer: The provided context does not contain information about what shoe size you wear.


### Question 10

Task success this question: baseline 60%, treatment 60%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 3 | 2 | 1 |
| Treatment (B) | 5 | 3 | 2 | 1 |

- Baseline answer: The provided context states that blue running shoes are out of stock in sizes 9 and 10, but it does not specify what your shoe size is.
- Treatment answer: The provided context does not contain the answer because it does not specify what your shoe size is, only that blue running shoes in sizes 9 and 10 are out of stock.

