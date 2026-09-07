# CAG Batch E: PagedAttention and vAttention — Live Measurement Report

**Scope:** #33 (story), #126 (task, vAttention). See `docs/superpowers/specs/2026-09-07-cag-paged-attention-design.md` for the design and for why this batch splits into a validated half and an implemented half.

## What CAG.md asks for, and what this batch built

CAG.md credits PagedAttention with memory sharing across parallel sequences from one prompt, no fragmentation, efficient beam search, and higher achievable batch sizes, and names vAttention as the alternative that drops paging for contiguous dynamic allocation. PagedAttention is vLLM's own memory manager, so this batch validated it against a live server rather than reimplementing it. vAttention exists nowhere in this stack, so this batch built both allocators as real domain code and measured the trade head-to-head.

## The finding: which layer actually delivers the sharing

The headline claim — parallel sequences from one prompt share physical memory — is true, and the measurement pins down something CAG.md's account leaves out. Both runs below use the same server, same model (Qwen2.5-0.5B-Instruct on this project's AMD 7900 XTX), same ~265-token prompt, same `n`, differing only in whether prefix caching is enabled:

| Configuration | n=1 latency | n=8 latency | n=8 prompt tokens (logical) | n=8 served from shared blocks |
|---|---|---|---|---|
| `--no-enable-prefix-caching` | 0.091s | **4.556s** | 2120 | — |
| `--enable-prefix-caching` | 0.072s | **0.143s** | 2120 | **2048** |

PagedAttention's block table is present in both rows — it is not optional in vLLM and cannot be switched off. Yet with prefix caching disabled, eight parallel branches of one prompt shared nothing and cost **31.8x** more wall-clock than the same request with it enabled. The honest conclusion is therefore narrower than CAG.md's phrasing suggests: **the block table is what makes sharing expressible, and the prefix-cache layer built on top of it is what actually populates and reuses those blocks across a parallel-sampling request.** Attributing the sharing to PagedAttention alone is not wrong so much as incomplete, and the incompleteness is measurable at 31.8x.

## The counter that would have produced a wrong answer

`vllm:prompt_tokens_total` reports 2120 for the `n=8` request in **both** configurations — a clean 8 x 265. It is a logical counter that bills every branch for the full prompt regardless of what was physically computed, so reading it alone would have "shown" that no sharing occurs even in the configuration where 2048 tokens demonstrably came from shared blocks. The load-bearing signal is `vllm:prefix_cache_hits_total`. The integration test asserts the logical counter's uninformative 8x behavior explicitly, so a later reader cannot mistake it for evidence in either direction.

## Sharing is block-granular, confirmed to the token

The 2048 figure is exact arithmetic, not an approximation. A 265-token prompt is 16 whole 16-token blocks (256 tokens) plus a 9-token remainder. Eight branches x 256 block-aligned tokens = 2048 shared; 8 x 9 = 72 tokens fall outside any complete block and are genuinely prefilled per branch; 2120 - 2048 = 72. A second test repeats the measurement on a different prompt and asserts the shared total is always a whole multiple of 16 (measured 2176 = 136 blocks), which is what separates real block-table sharing from a coincidentally large number.

## PagedAttention versus vAttention: the trade, measured

Both strategies built as real allocators under a shared `KVCacheAllocator` port and run through an identical workload — same pool, five 20-slot sequences, two freed from the middle, then one 30-slot request:

| | Paged (blocks) | Contiguous (vAttention) |
|---|---|---|
| Free slots at the final request | 40 | 40 |
| Largest usable run | 40 (fungible blocks) | 20 |
| External fragmentation | **0.0** | **0.5** |
| Final 30-slot request | **served** | **refused** |
| Cost of forking a 160-token sequence 7 times | **0 extra slots** | 7 full copies |
| Internal fragmentation | real (partly-filled tail block) | **none** |

The refusal is the point: the contiguous allocator has enough total free space and still cannot serve the request, because it needs a *run* while the paged allocator needs only a *count*. That is CAG.md's "no fragmentation" claim made falsifiable rather than asserted.

The last row is the correction. CAG.md says paging means "no fragmentation" without qualification; that is true only of the external kind. Paging pays for it in internal fragmentation — a sequence's last block is almost never exactly full — which the contiguous allocator, sized exactly to each sequence, does not have at all. `allocation_metrics.py` measures both halves deliberately, because reporting only the flattering one would repeat the overstatement instead of correcting it.

## What this batch does not do

- **Does not reimplement PagedAttention** — it is vLLM's own memory manager; this batch validates it and sharpens CAG.md's account of which layer delivers the sharing.
- **Does not claim the two allocators are vLLM's or the vAttention paper's real code** — they are faithful models of the documented strategies, built so the stated trade can be measured, with simplifications disclosed rather than implied away.
- **Does not measure higher batch sizes or beam search directly** — both follow from the sharing measured here, but establishing them needs sustained concurrent load, which belongs to `scripts/benchmark.py`, not this batch.
- **Does not run its live half in CI** — the second CAG batch of which this is true, for the same reason as the first: GitHub Actions has no ROCm GPU, so the test skips with a stated reason.

## The numbers

Unit tests: 664 (start of batch, after Prefix Caching) → 688 (24 new: 8 metrics, 8 paged allocator, 8 contiguous allocator). Integration tests: 2 new against a real vLLM server, outside the CI-run count for the reason above.
