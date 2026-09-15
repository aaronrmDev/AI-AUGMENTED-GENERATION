# CAG Batch H: Multi-Turn Caching — Live Measurement Report

**Scope:** #36 (story), #113 (RocketKV-MT), #114 (KVzip), #116 (ShadowKV), #117 (MemServe), #118 (SGLang's approach). See `docs/superpowers/specs/2026-09-07-cag-multi-turn-caching-design.md` for the design. CAG's ninth and final technique.

## The claim is about a growth rate, so the measurement is too

CAG.md promises "near-O(1) amortized cost per turn instead of cost that grows with every message." A smaller session total would not establish that — halving a quadratic leaves it quadratic. What distinguishes O(1)-per-turn is that per-turn work stops growing while the conversation does not, so that is what both halves measure.

## Live: an eight-turn conversation against a real vLLM server

Each turn sends the full accumulated history, as any chat application does. `vllm:prompt_tokens_total` gives the logical prompt; `vllm:prefix_cache_hits_total` gives what was served from cache; the difference is what the engine actually prefilled.

| Turn | Prompt tokens | Served from cache | **Newly computed** | Latency |
|---|---|---|---|---|
| 1 | 59 | 0 | 59 | 0.168s |
| 2 | 105 | 80 | **25** | 0.162s |
| 3 | 151 | 128 | **23** | 0.113s |
| 4 | 197 | 176 | **21** | 0.102s |
| 5 | 243 | 224 | **19** | 0.089s |
| 6 | 289 | 272 | **17** | 0.092s |
| 7 | 335 | 304 | **31** | 0.100s |
| 8 | 381 | 352 | **29** | 0.106s |

The prompt grows **6.5x** across the session; newly-computed work stays inside a 17-31 token band and does not trend upward. That band is one 16-token cache block wide, which is where the block boundary falls relative to each turn's new content rather than growth. Latency stays flat near 0.1s instead of tracking prompt length, and by turn 8 more than 92% of the prompt is being served from cache.

This is the claim, measured: cost per turn stopped growing while the conversation kept growing. Turn 1 is excluded from the steady-state comparison because nothing is cached yet — including a cold start beside warm turns would compare a miss against hits.

## Simulated: the five methods on a 20-turn conversation

| Strategy | Recomputed | vs naive | Per turn | Flat? | Peak held | Transfer |
|---|---|---|---|---|---|---|
| naive-recompute | 21000 | 1.00x | 1050.0 | **No** | 2000 | 0 |
| incremental-append | 2000 | 0.10x | 100.0 | Yes | 2000 | 0 |
| rocketkv-mt | 800 | 0.04x | 40.0 | Yes | 2000 | 0 |
| kvzip | 2500 | 0.12x | 125.0 | Yes | **1000** | 0 |
| shadowkv | 575 | **0.03x** | 28.8 | Yes | 2000 | 0 |
| memserve | 2000 | 0.10x | 100.0 | Yes | **400** | 13600 |
| sglang | 2000 | 0.10x | 100.0 | Yes | 2000 | 0 |

The naive baseline is the only one whose per-turn cost grows, and it grows exactly as CAG.md's "reprint the whole book every chapter" image predicts: turn 20 costs twenty times turn 1. Every cached strategy is flat.

What separates them is what each pays for flatness. RocketKV-MT is cheapest per turn (40 tokens) but buys that by capping what the current turn attends over — a quality cost this cost model cannot see, and says so rather than scoring it as a straight win. KVzip's total is the *highest* of the cached strategies (2500, above plain incremental append's 2000) because its one-time overhead is real; what it buys is halved residency, and its per-turn cost keeps falling the longer the session runs. MemServe holds only 400 tokens locally against everyone else's 2000, and pays 13600 tokens of network transfer for it — the one strategy whose cost lands somewhere other than compute or memory.

## The branching workload, and a baseline that had to be fixed first

SGLang ties every other cached strategy on a flat conversation, which is correct rather than disappointing: CAG.md says it suits "complex, multi-step agent workflows rather than a single flat conversation." Showing what it does contribute needs a branching program — a 300-token shared trunk with four 50-token branches forking off its tip:

| Strategy | Recomputed | vs naive | Per turn |
|---|---|---|---|
| naive-recompute | 2000 | 1.00x | [100, 200, 300, 350, 350, 350, 350] |
| incremental-append | 1400 | 0.70x | [100, 100, 100, 50, **350, 350, 350**] |
| sglang | **500** | **0.25x** | [100, 100, 100, 50, **50, 50, 50**] |

The session-keyed baseline re-materialises the 300-token trunk for each branch after the first; SGLang materialises it once and each branch costs only its own 50 tokens — a 2.8x difference on the workload the method exists for.

**That comparison initially measured 500 against 500, and the reason is worth recording.** `IncrementalAppendSession` first charged only new tokens for every turn regardless of structure, which silently modelled an oracle that shares trunks across branches for free. Since sharing across branches is precisely SGLang's contribution, the baseline had been handed the very benefit under test, and the two strategies were indistinguishable on the one workload built to tell them apart. Correcting the baseline to be session-keyed — reuse along its own chain, a fork starts a fresh session — produced the 500-against-1400 above. The failure mode is the same family as the Cache-Aware Batching batch's contaminated control: a baseline that is too good makes a real effect vanish, and nothing errors.

## What this batch does not do

- **Does not measure answer quality across turns** — CAG.md also claims "a consistent persona since the agent genuinely remembers the full conversation." That is a quality claim needing a judge harness, not a cache measurement, and only the cost claim is tested here.
- **Does not build the Real-Time Chat combination** — #43/#134 is now unblocked (Multi-Turn Caching, Eviction, and Compression all exist) but composing them is its own batch.
- **Does not run its live half in CI** — the fourth CAG batch of which this is true: GitHub Actions has no ROCm GPU, so the test skips with a stated reason.

## The numbers

Unit tests: 726 (start of batch, after Hybrid Offloading) → 748 (22 new: 10 session metrics, 12 strategies). Integration tests: 1 new against a real vLLM server, outside the CI-run count for the reason above.
