# CAG Batch F: Cache-Aware Batching — Live Measurement Report

**Scope:** #38 (story), #128 (task). See `docs/superpowers/specs/2026-09-07-cag-cache-aware-batching-design.md` for the design. This batch completes the serving-engine trio (Prefill, Allocation, Scheduling) and makes the High-Throughput Serving combination (#41/#131) buildable.

## What CAG.md asks for, and what this batch built

CAG.md's Scheduling paragraph makes two claims in one breath: requests sharing a prefix get batched so that prefix is computed once for the group, and continuous in-flight batching keeps utilisation high without waiting for a batch to fill. This batch measures them separately, because they turn out to contribute very differently. vLLM's own scheduler was measured directly for the engine-level claim; the grouping-policy claim was built as real code, since vLLM does not allow its scheduling policy to be substituted.

## Continuous batching, measured against the real engine

Sixteen requests issued one after another against the same sixteen issued at once:

| Run | 16 sequential | 16 concurrent | Speedup |
|---|---|---|---|
| First | 1.282s | 0.154s | **8.3x** |
| Second | 1.121s | 0.182s | **6.1x** |

A speedup of this size is only possible if the engine admits requests into a batch that is already running, rather than serialising them — which is exactly the in-flight admission CAG.md describes.

## Prefix sharing inside a burst, and the bug that nearly hid it

Sixteen concurrent requests sharing a long prefix, against sixteen concurrent requests sharing nothing:

| Burst | Latency | Tokens served from shared blocks |
|---|---|---|
| Shared prefix | 0.166s / 0.179s | **3072** |
| Distinct prefixes | 0.253s / 0.213s | **0** |

The hit counts are stable across runs; the latency margin is not (1.52x then 1.19x), which is why the test's load-bearing assertions are the deterministic hit counts and the timing comparison is asserted but explicitly secondary.

**This result was wrong twice before it was right, and the reason is worth recording.** The first version of the distinct-prefix burst reused prompt strings an earlier probe had already sent to the same long-lived server, so the "shares nothing" control was itself being served from cache — 1632 hit tokens where zero was expected. Adding one salt per run cut that to 480, because all sixteen prompts still began with an identical `Unrelated <salt> prompt` span long enough to fill a whole 16-token block. Only a per-prompt salt in leading position produced a genuinely cold control.

The correction changed the conclusion, not just the hygiene. With the contaminated control the prefix benefit measured about 7% and read as noise, and this report's first draft said so. With a genuinely cold control it measures 1.19-1.52x. The bug was making the control fast rather than the treatment slow, which is the failure mode that produces a confidently understated result rather than an obviously broken one.

## Prefix-aware grouping versus FIFO, simulated

vLLM will not accept a substituted scheduler, so both policies were built as real code and scored by the prefill work each grouping implies. On interleaved arrivals — requests carrying different system prompts alternating, which is what CAG.md's own named use cases produce:

| Workload | No sharing | FIFO | Prefix-aware | Saving vs FIFO |
|---|---|---|---|---|
| 2 prefixes x 4 requests, 64-token prefix, batch 2 | 520 | 520 | 264 | **49.2%** |
| 2 prefixes x 8 requests, 128-token prefix, batch 4 | 2064 | 2064 | 528 | **74.4%** |
| 4 prefixes x 8 requests, 128-token prefix, batch 8 | 4128 | 4128 | 544 | **86.8%** |

The saving grows with both batch size and prefix length, which is the shape CAG.md predicts when it calls one prefill serving hundreds of requests.

Two honest qualifications. FIFO matches the no-sharing baseline exactly in all three rows because the arrivals are interleaved — that is the realistic adversarial case rather than a rigged one, but if arrivals already clustered by prefix, FIFO would capture much of the benefit unaided, and the claim's real scope is "grouping matters when arrivals are interleaved." And these are simulated savings from a cost model, not end-to-end measurements inside a serving engine; the model charges each batch once for its block-aligned shared prefix and rounds that prefix **down** to whole blocks precisely so it cannot overstate what a real engine could share.

## What this batch does not do

- **Does not replace vLLM's scheduler** — the two schedulers make the policy question measurable; they do not run inside the engine.
- **Does not build the High-Throughput Serving combination** — #41/#131 is now unblocked, but composing the three techniques is its own batch.
- **Does not isolate in-flight join as its own scenario** — the concurrent speedup already requires in-flight admission; a staggered-arrival harness would be needed to time the join itself.
- **Does not run its live half in CI** — the third CAG batch of which this is true: GitHub Actions has no ROCm GPU, so the test skips with a stated reason.

## The numbers

Unit tests: 688 (start of batch, after PagedAttention) → 703 (15 new: 8 batching metrics, 7 schedulers). Integration tests: 2 new against a real vLLM server, outside the CI-run count for the reason above.
