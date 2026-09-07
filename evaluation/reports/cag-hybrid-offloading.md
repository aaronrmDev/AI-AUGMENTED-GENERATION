# CAG Batch G: Hybrid Memory Offloading — Measurement Report

**Scope:** #35 (story), #106 (InfiniGen), #108 (LayerKV), #109 (INF2), #110 (KVPR), #112 (Oneiros). See `docs/superpowers/specs/2026-09-07-cag-hybrid-offloading-design.md` for the design.

## Simulated, and saying so up front

Unlike the three serving-engine batches before it, this one has no live half, and that is a property of the subject rather than a shortcut. vLLM exposes no tiering knobs to configure, there is no second device to offload between on this machine, and INF2's premise — attention computation running on the SSD itself — needs CSD hardware that does not exist here. Every millisecond quoted below comes from a discrete-event model with stated parameters.

What that model can legitimately establish is the *relative* behavior of five documented mechanisms under identical assumptions, and the structural reason one beats another. What it cannot establish is absolute latency on real hardware, and nothing here claims otherwise.

## The regime where the comparison says nothing

Run with compute (10ms/layer) outlasting fetches (6ms/layer), 32 layers on an 8-layer GPU:

| Strategy | Total | vs GPU-only | Transfer issued | Hidden | Overlap |
|---|---|---|---|---|---|
| sequential | 464.0ms | 1.45x | 144.0 | 0.0 | 0.0% |
| layerkv | 320.0ms | 1.00x | 144.0 | 144.0 | 100% |
| infinigen | 320.0ms | 1.00x | 36.0 | 36.0 | 100% |
| kvpr | 440.0ms | 1.38x | 72.0 | 72.0 | 100% |
| oneiros | 320.5ms | 1.00x | 112.0 | 111.5 | 99.6% |
| inf2 | 320.0ms | 1.00x | 120.0 | 120.0 | 100% |

Three strategies land on exactly 320ms, which is 32 layers x 10ms — the pure-compute floor. When every fetch fits inside the compute it overlaps with, any strategy that prefetches at all hides everything, and the differences between them stop being visible. This regime is reported rather than dropped precisely because it is the one that would make a flattering-but-empty comparison: pick these parameters and five mechanisms look identical.

KVPR is the exception, at 1.38x, and for a defensible reason: it converts transfer into GPU recompute, so when the GPU is already the bottleneck it is trading into the scarce resource rather than out of it.

## The regime the technique family actually exists for

Run with fetches (40ms/layer) outlasting compute (10ms/layer) — the long-context situation where GPU memory has run out and data is coming from far away:

| Strategy | Total | vs GPU-only | Transfer issued | Hidden | Overlap | Resident |
|---|---|---|---|---|---|---|
| sequential | 1280.0ms | 4.00x | 960.0 | 0.0 | 0.0% | 8 |
| layerkv | 1040.0ms | 3.25x | 960.0 | 240.0 | **25.0%** | 8 |
| kvpr | 800.0ms | 2.50x | 480.0 | 120.0 | 25.0% | 8 |
| oneiros | 808.5ms | 2.53x | 656.0 | 167.5 | 25.5% | **16** |
| inf2 | 392.0ms | 1.23x | 312.0 | 240.0 | 76.9% | 8 |
| infinigen | **320.0ms** | **1.00x** | 240.0 | 240.0 | **100%** | 8 |

## The structural finding

LayerKV's overlap efficiency is exactly 25.0%, and that is not a coincidence of the parameters: compute is 10ms against 40ms fetches, and 10/40 is 25%. **Pure overlap is bounded by the compute-to-transfer ratio and cannot exceed it.** A strategy that only prefetches can hide, at most, as much transfer as it has compute to hide it behind.

That bound is what separates the field. The two strategies that reach or approach the compute floor are the two that also reduce how much has to move: InfiniGen fetches only the quarter of entries it predicts are critical, so its 240ms of transfer fits entirely under 320ms of compute and it lands exactly on the floor at 1.00x; INF2 shrinks what crosses PCIe tenfold and reaches 1.23x, paying only for its slower in-storage accelerator. KVPR and Oneiros both improve on LayerKV by reducing transfer another way — recomputing half of it, or holding twice as much resident — and both land near 2.5x, better than pure overlap but short of the volume-reducers.

CAG.md frames the shared art of this family as "hiding transfer latency behind computation that's already happening anyway." The measurement sharpens that: hiding is necessary but bounded, and the strategies that win are the ones that also shrink what must be hidden. A reader taking CAG.md's sentence at face value would expect LayerKV — the purest expression of hiding — to be representative of the family, when it is in fact the weakest member of it in the regime that matters.

## The trade CAG.md states, quantified

"GPU memory saved rises together with latency" holds in both regimes, but the exchange rate varies enormously by mechanism. All six strategies serve 32 layers on a GPU holding 8 — a 4.0x effective context multiplier — and pay for it with anywhere from 0% extra latency (InfiniGen) to 300% (sequential tiering). The technique's value is real; which strategy implements it is worth 4x.

## What this batch does not do

- **Does not measure real hardware** — every number above is simulated under the parameters named with it.
- **Does not implement a real three-tier store** — the cold tier is a latency parameter, not SSD-backed storage.
- **Does not build the Long Context Champion combination** — #42/#133 is now unblocked (Eviction, Compression, and Offloading all exist) but composing them is its own batch.

## The numbers

Unit tests: 703 (start of batch, after Cache-Aware Batching) → 726 (23 new: 11 metrics, 12 strategies). No integration tests, for the reason given at the top.
