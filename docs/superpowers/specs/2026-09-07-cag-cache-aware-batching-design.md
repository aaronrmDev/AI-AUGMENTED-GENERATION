# CAG Batch F: Cache-Aware Batching — Design Spec

**Scope:** #38 (story), #128 (task). See `docs/architecture/CAG.md`, "How the five pipeline stages fit together" (Scheduling paragraph), and `docs/inputs/concepts/advanced_cag_concepts.md`, Concept 8. Completes the serving-engine trio — Prefill (Prefix Caching), Allocation (PagedAttention), Scheduling — and so makes the High-Throughput Serving combination (#41/#131) buildable.

## Two claims that CAG.md states as one

CAG.md's Scheduling paragraph bundles two distinct claims: that requests sharing a prefix get batched together so that prefix is computed once per group rather than once per request, and that continuous in-flight batching lets new requests join a running batch without waiting for it to fill. They are separable, they have different mechanisms, and — as this batch measures — they differ substantially in how much they contribute. Measuring them together would have reported one number for two effects.

## Half 1 — validated against a real server

vLLM already does continuous batching, so this half tests the real engine rather than reimplementing a scheduler. Two measurements against a live server:

- **Continuous batching**: 16 requests issued sequentially against the same 16 issued concurrently. Measured **6.1x to 8.3x** faster concurrent across runs, which is only possible if the engine admits them into a running batch rather than serialising them.
- **Prefix-aware benefit within a burst**: 16 concurrent requests sharing a long prefix against 16 concurrent requests sharing nothing. The shared burst reuses **3072 tokens** of cached blocks against **0** for the distinct burst, and completes 1.19x to 1.52x faster.

**A test-isolation bug that changed the answer, recorded because it nearly stood.** The first version of the distinct-prefix burst reused prompt strings a previous run had already sent, so the "shares nothing" baseline was itself being served from cache — 1632 hit tokens where zero was expected. Salting the prompts once per run reduced that to 480, because all sixteen still shared an identical leading `Unrelated <salt> prompt` span long enough to fill a whole 16-token block. Only a per-prompt salt in leading position produced a genuinely cold baseline. The correction matters beyond hygiene: with the contaminated baseline the prefix benefit measured about 7% and looked like noise, and with a genuinely cold one it measures 1.19-1.52x. The bug was making the control group fast, not the treatment slow.

Because the latency margin varies run to run while the hit counts do not, the test's load-bearing assertions are the deterministic ones (shared burst reuses a whole number of 16-token blocks; distinct burst reuses exactly zero), with the timing comparison asserted but explicitly secondary.

## Half 2 — implemented, because the scheduler is not swappable

vLLM will not let this project substitute its scheduling policy, so the grouping claim cannot be tested by configuration the way the engine claims can. Both policies are therefore built as real domain code under a shared `BatchScheduler` port:

- `src/cag/infrastructure/fifo_batch_scheduler.py` — the baseline CAG.md implicitly argues against: batch purely in arrival order, so requests sharing a prefix are grouped only when they happen to arrive adjacently.
- `src/cag/infrastructure/prefix_aware_batch_scheduler.py` — groups by block-aligned prefix signature, longest signature first so the tightest groupings form before looser ones absorb the remainder, preserving arrival order within a group (grouping changes who shares a batch, deliberately not who waits longest).

`src/cag/domain/batching_metrics.py` scores a grouping by the prefill work it implies: `prefill_tokens_required` charges each batch once for its shared block-aligned prefix plus each member's remainder, against an `unbatched_prefill_tokens` baseline where nothing is shared. `longest_common_block_prefix` rounds **down** to whole blocks deliberately — a serving engine shares cache at block granularity, so counting a partial block of agreement as saved would overstate the result.

Measured on interleaved arrivals, where requests carrying different system prompts alternate:

| Workload | Baseline | FIFO | Prefix-aware | Saving vs FIFO |
|---|---|---|---|---|
| 2 prefixes x 4 requests, 64-token prefix, batch 2 | 520 | 520 | 264 | **49.2%** |
| 2 prefixes x 8 requests, 128-token prefix, batch 4 | 2064 | 2064 | 528 | **74.4%** |
| 4 prefixes x 8 requests, 128-token prefix, batch 8 | 4128 | 4128 | 544 | **86.8%** |

FIFO equals the no-sharing baseline in all three rows, and that is a property of the arrival pattern rather than a rigged comparison: interleaved arrivals are precisely the case CAG.md's named use cases produce (RAG APIs, chatbot platforms, multi-tenant serving, where requests carrying different system prompts arrive mixed together). If arrivals were already clustered by prefix, FIFO would capture much of the benefit on its own, and the honest scope of the claim is that grouping matters when arrivals are interleaved.

## Testing plan

1. **Unit tests** (`test_batching_metrics.py`, `test_batch_schedulers.py`), fast, deterministic, CI-safe: block-aligned prefix arithmetic, the cost model's charging behavior, each scheduler's grouping mechanics, and the head-to-head on interleaved arrivals.
2. **One integration test** (`tests/integration/test_cag_cache_aware_batching_against_real_vllm.py`) against a live vLLM server. Cannot run in CI for the same reason as the two prior batches; skips itself with a stated reason and runs from inside WSL2.

## What this batch does not do

- **Does not replace vLLM's scheduler** — the two schedulers model the policy question so the grouping claim is measurable; they do not run inside the engine, and the report says so rather than implying the simulated savings were observed end-to-end.
- **Does not build the High-Throughput Serving combination** — #41/#131 becomes buildable now that all three constituent techniques are built, but composing them is its own batch.
- **Does not test in-flight join as a separate scenario** — the 6.1-8.3x concurrent speedup already requires in-flight admission; isolating arrival-time join would need a staggered-arrival harness beyond this batch's Definition of Done.
