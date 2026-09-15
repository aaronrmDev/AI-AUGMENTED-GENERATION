# CAG Batch E: PagedAttention and vAttention — Design Spec

**Scope:** #33 (story), #126 (task, vAttention). See `docs/architecture/CAG.md`, "PagedAttention and its alternative: vAttention", and `docs/inputs/concepts/advanced_cag_concepts.md`, Concept 4.

## Why this batch has two halves of different kinds

PagedAttention and vAttention are described in CAG.md as opposite answers to one allocation-stage question, but they sit in completely different relationships to this project's actual stack. PagedAttention is not optional in vLLM — it *is* vLLM's memory manager, so there is nothing here for this project to implement, and the honest task is validation, the same shape the Prefix Caching batch took. vAttention is the opposite: it exists in the literature as the alternative that drops paging entirely, and no part of this project's stack implements it, so nothing can be validated and the honest task is to build it.

Splitting the batch that way lets each half answer the question its subject actually poses. The live half asks whether vLLM's block table really delivers the memory sharing CAG.md credits it with. The implemented half asks what paging actually buys and costs against the contiguous alternative, which no measurement against vLLM alone could show, since vLLM offers no contiguous mode to compare against.

## Half 1 — PagedAttention, validated against a real server

CAG.md's specific claims are memory sharing across parallel sequences from the same prompt, no fragmentation, efficient beam search, and higher achievable batch sizes. The first is the one that is both central and directly observable from outside the engine, so it is what this half measures: send one long prompt at `n=1` and the same prompt at `n=8` (parallel sampling — literally "parallel sequences that start from the same prompt"), and see whether the eight branches share one physical copy of the prompt or pay for eight.

**The measurement design changed once, for a real reason, and the change is the batch's most useful finding.** The original design ran the server with prefix caching *disabled*, reasoning that this would isolate PagedAttention's own block sharing from the inter-request caching the previous batch already measured. Run that way, `n=8` took **4.556s against `n=1`'s 0.091s** and shared nothing. Re-run against the identical server with prefix caching enabled, the same `n=8` request took **0.143s** with **2048 of its 2120 prompt tokens served from shared blocks**. PagedAttention's block table is present in both configurations — it cannot be turned off — so the conclusion is not that PagedAttention fails to share, but that in vLLM V1 the sharing CAG.md attributes to PagedAttention is *delivered by the prefix-cache layer built on top of the block table*, not by the block table alone. The block table is what makes sharing expressible; the prefix cache is what actually populates and reuses it across the branches of a parallel-sampling request. That distinction is invisible in CAG.md's own account and is this batch's correction to it.

A second trap, disclosed because it nearly produced a wrong conclusion: `vllm:prompt_tokens_total` reads `n x prompt_length` in **both** configurations. It is a logical counter that bills every branch for the full prompt regardless of what was physically computed, so on its own it proves nothing about sharing. The load-bearing signal is `vllm:prefix_cache_hits_total`, and the test asserts the logical counter's uninformative behavior explicitly so a later reader cannot mistake it for evidence.

Sharing is block-granular, which the numbers confirm exactly rather than approximately: the 265-token prompt is 16 whole 16-token blocks (256 tokens) plus a 9-token remainder, and 8 branches x 256 shareable tokens is precisely the 2048 measured, leaving 8 x 9 = 72 tokens genuinely prefilled (2120 - 2048 = 72). A second test asserts the shared total is always a whole multiple of the 16-token block size, which is what distinguishes real block-table sharing from a coincidentally large number.

## Half 2 — vAttention, implemented so the trade can be measured

Two real allocators under this project's hexagonal layering, both implementing a new `KVCacheAllocator` port (`allocate`, `extend`, `fork`, `free`, plus the three observation methods the metrics need), modelled in token slots rather than bytes because the comparison is about layout, not about any model's per-token footprint:

- `src/cag/infrastructure/paged_allocator.py` — fixed-size blocks, a per-sequence block table, blocks drawn from anywhere in the pool, refcounted sharing on `fork`, and real copy-on-write when a forked branch grows into a block it still shares.
- `src/cag/infrastructure/contiguous_allocator.py` — vAttention's shape: one contiguous run per sequence, first-fit placement, in-place growth when the following slots are free and full relocation when they are not, and `fork` as a full copy since there is no block table to share through.

`src/cag/domain/allocation_metrics.py` deliberately measures **both** halves of the trade. `external_fragmentation` is the kind paging genuinely eliminates. `internal_fragmentation` is the kind paging introduces — a partly-filled tail block — and which the contiguous allocator has none of. CAG.md says paging means "no fragmentation" without qualification; reporting only the flattering half would repeat that overstatement rather than correct it.

The decisive unit test runs an identical workload through both allocators — same pool, same five 20-slot sequences, same two frees, same final 30-slot request. The contiguous allocator refuses the request with 40 slots free, because they sit in two separate runs of 20 (external fragmentation 0.5); the paged allocator serves it, because it needs a *count* of blocks and never a *run*. That is CAG.md's fragmentation claim turned into something falsifiable.

## Testing plan

1. **Unit tests** (`test_allocation_metrics.py`, `test_paged_allocator.py`, `test_contiguous_allocator.py`), fast, deterministic, CI-safe: each allocator's own mechanics (block rounding, refcounted fork, copy-on-write, relocation on blocked growth, first-fit failure under fragmentation) plus the head-to-head comparison above.
2. **One integration test** (`tests/integration/test_cag_paged_attention_against_real_vllm.py`) against a real vLLM server with `--enable-prefix-caching`. Cannot run in CI, for the same reason the Prefix Caching batch's test cannot; skips itself with a stated reason and runs from inside WSL2.

## What this batch does not do

- **Does not reimplement PagedAttention** — it is vLLM's own memory manager; this half validates it and corrects CAG.md's account of which layer delivers the sharing.
- **Does not claim the allocators are vLLM's or the vAttention paper's actual code** — they are faithful models of the two documented strategies, built to make the stated trade measurable, with their simplifications disclosed rather than implied away.
- **Does not measure "higher achievable batch sizes" or beam search directly** — both follow from the sharing this batch does measure, but establishing them properly needs sustained concurrent load, which is `scripts/benchmark.py`'s eventual job, not this batch's.
