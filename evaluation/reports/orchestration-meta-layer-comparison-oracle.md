# orchestration-meta-layer — Comparison Result

**Model:** qwen3.5, Ollama
**Success criterion:** see evaluation/scenarios/orchestration-meta-layer/queries.yaml

## Quantitative

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | qwen3.5, Ollama | 1235 | 9140 | 7720ms / 28510ms | 80% | — | Baseline = RAG-only AnswerQuestion (top_k=3). Treatment = UnifiedAnswerQuestion routed by each question's labeled route in queries.yaml (an oracle), which separates what the pipeline adds from classifier routing errors, with the CAG/MAG/RAG cascade under generous timeouts, the 128K budget allocator, and a real session budget record. CAVEAT 1: judge and generator are both qwen3.5 (self-grading risk, as in every earlier batch). CAVEAT 2: CAG is a CPU distilgpt2 proxy whose lookup, not generation speed, is what the treatment exercises. CAVEAT 3: question 1 is a static policy question, so Concept 1's rule sends it to CAG, which holds the superseded thirty-day policy in this corpus; when CAG answers it alone, the stale value is what the model sees. That is a Sync Mixer failure routing cannot fix, and it is reported, not relabeled. ROUTES (last repeat): 'What is the return window for unopened items?': routed ['cag'], attempts [('cag', 'hit')]; 'What changed in the return policy today?': routed ['rag'], attempts [('rag', 'hit')]; 'How long does standard shipping take?': routed ['cag'], attempts [('cag', 'hit')]; 'How do I set up an account?': routed ['cag'], attempts [('cag', 'hit')]; 'How long is the product warranty?': routed ['cag'], attempts [('cag', 'hit')]; 'Are the blue running shoes in stock today?': routed ['rag'], attempts [('rag', 'hit')]; 'Is there a sale on backpacks today?': routed ['rag'], attempts [('rag', 'hit')]; 'What shipping speed did I say I prefer?': routed ['mag'], attempts [('mag', 'hit')]; 'What shoe size do I wear?': routed ['mag'], attempts [('mag', 'hit')]; 'Are the running shoes in my size in stock right now?': routed ['mag', 'rag'], attempts [('mag', 'hit'), ('rag', 'hit')] |
| Treatment | ✓ | ✓ | ✓ | qwen3.5, Ollama | 1054 | 8587 | 6692ms / 32383ms | 90% | -6.1% output tokens, -13.3% p50 latency | Baseline = RAG-only AnswerQuestion (top_k=3). Treatment = UnifiedAnswerQuestion routed by each question's labeled route in queries.yaml (an oracle), which separates what the pipeline adds from classifier routing errors, with the CAG/MAG/RAG cascade under generous timeouts, the 128K budget allocator, and a real session budget record. CAVEAT 1: judge and generator are both qwen3.5 (self-grading risk, as in every earlier batch). CAVEAT 2: CAG is a CPU distilgpt2 proxy whose lookup, not generation speed, is what the treatment exercises. CAVEAT 3: question 1 is a static policy question, so Concept 1's rule sends it to CAG, which holds the superseded thirty-day policy in this corpus; when CAG answers it alone, the stale value is what the model sees. That is a Sync Mixer failure routing cannot fix, and it is reported, not relabeled. ROUTES (last repeat): 'What is the return window for unopened items?': routed ['cag'], attempts [('cag', 'hit')]; 'What changed in the return policy today?': routed ['rag'], attempts [('rag', 'hit')]; 'How long does standard shipping take?': routed ['cag'], attempts [('cag', 'hit')]; 'How do I set up an account?': routed ['cag'], attempts [('cag', 'hit')]; 'How long is the product warranty?': routed ['cag'], attempts [('cag', 'hit')]; 'Are the blue running shoes in stock today?': routed ['rag'], attempts [('rag', 'hit')]; 'Is there a sale on backpacks today?': routed ['rag'], attempts [('rag', 'hit')]; 'What shipping speed did I say I prefer?': routed ['mag'], attempts [('mag', 'hit')]; 'What shoe size do I wear?': routed ['mag'], attempts [('mag', 'hit')]; 'Are the running shoes in my size in stock right now?': routed ['mag', 'rag'], attempts [('mag', 'hit'), ('rag', 'hit')] |

## Qualitative (per question)

### Question 1

Task success this question: baseline 100%, treatment 0%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 1 |

- Baseline answer: The return window for unopened items is forty-five days of purchase.
- Treatment answer: The return window for unopened items is thirty days of purchase.

- Treatment unverifiable claims: The return window for unopened items is thirty days of purchase.

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
| Baseline (A) | 5 | 5 | 3 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: To set up an account, provide your email address and choose a password of at least twelve characters.
- Treatment answer: To set up an account, provide your email address and choose a password of at least twelve characters. A verification email is sent immediately after registration.


### Question 5

Task success this question: baseline 100%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The product carries a two-year limited warranty.
- Treatment answer: The product warranty is two years.


### Question 6

Task success this question: baseline 100%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 3 | 5 |
| Treatment (B) | 5 | 5 | 3 | 5 |

- Baseline answer: According to the context, the blue running shoes are out of stock in sizes 9 and 10 as of this morning.
- Treatment answer: As of this morning, the blue running shoes are out of stock in sizes 9 and 10.


### Question 7

Task success this question: baseline 100%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 5 | 5 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: Yes, according to the context, there is a flash sale today that takes 20% off every backpack until midnight.
- Treatment answer: Yes, according to the documents, there is a flash sale today that takes 20% off every backpack until midnight.


### Question 8

Task success this question: baseline 0%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 1 | 2 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain information about what shipping speed you prefer.
- Treatment answer: You prefer expedited two-day shipping on every order.


### Question 9

Task success this question: baseline 20%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 4 | 2 | 3 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer to what shoe size you wear; it only lists the sizes of blue running shoes that are currently out of stock.
- Treatment answer: You wear size 10 running shoes.


### Question 10

Task success this question: baseline 80%, treatment 100%

| Response | Coherence | Relevance | Completeness | Groundedness |
|---|---|---|---|---|
| Baseline (A) | 5 | 5 | 2 | 1 |
| Treatment (B) | 5 | 5 | 5 | 5 |

- Baseline answer: The provided context does not contain the answer because it does not specify your shoe size or the availability of sizes other than 9 and 10.
- Treatment answer: No, as of this morning, the blue running shoes are out of stock in sizes 9 and 10.

- Baseline unverifiable claims: The provided context does not specify your shoe size
