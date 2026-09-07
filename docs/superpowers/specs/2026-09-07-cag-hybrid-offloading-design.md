# CAG Batch G: Hybrid Memory Offloading — Design Spec

**Scope:** #35 (story), #106 (InfiniGen), #108 (LayerKV), #109 (INF2), #110 (KVPR), #112 (Oneiros). See `docs/architecture/CAG.md`, "Hybrid Memory Offloading", and `docs/inputs/concepts/advanced_cag_concepts.md`, Concept 5. Unblocks the Long Context Champion combination (#42/#133), whose other two constituents (Eviction, Compression) are already built.

## What is being modelled, and what that does and does not license

Every strategy in this family is about where KV data lives across a GPU/CPU/SSD hierarchy and how much of the resulting movement can be made to happen underneath compute that was going to run anyway. None of it is measurable against this project's hardware the way the serving-engine batches were: vLLM exposes no tiering knobs to configure, there is no second GPU to offload between, and no CSD-equipped SSD exists here to run INF2 on at all. So this batch does what the allocator half of the PagedAttention batch did — builds the strategies as real code and runs them through a shared discrete-event model, with the parameters stated as parameters.

**The numbers this batch produces are simulated, and the report says so wherever it quotes one.** What the simulation can legitimately establish is the *relative* behavior of five documented mechanisms under identical assumptions, and the structural reasons one beats another. What it cannot establish is absolute latency on real hardware, and no claim of that kind is made.

## The model

A layer-by-layer forward pass over `num_layers`, of which only `gpu_layer_capacity` fit resident. Each layer costs `compute_ms_per_layer`; a non-resident layer must be fetched at `warm_fetch_ms_per_layer` (CPU) or `cold_fetch_ms_per_layer` (SSD).

The shared overlap primitive, `pipeline_total`, models prefetching honestly: while layer *i* computes, layer *i+1*'s fetch is already in flight, so the pipeline stalls only for however much of that fetch outlasts the compute. The first layer's fetch is charged in full, because nothing has started yet for it to hide behind — modelling it as free would invent an overlap no real pipeline gets.

`OffloadRun` records `transfer_ms_issued` and `transfer_ms_hidden` separately rather than only a total, which is what makes `overlap_efficiency` a measurement rather than an inference.

## The five strategies, each modelling its own distinguishing mechanism

- **LayerKV** (#108) — whole-layer granularity: a subset stays resident, the rest stream in, overlapped against resident-layer compute. Pure overlap, no volume reduction.
- **InfiniGen** (#106) — predicts which entries the *next* layer will need and prefetches only those. Two mechanisms compounding: it overlaps *and* moves less.
- **KVPR** (#110) — partially recomputes on the GPU while transferring the remainder, so a layer's stall is the longer of the two halves rather than their sum. Recompute is charged at real GPU cost, because rebuilding KV is real work on the same device.
- **Oneiros** (#112) — structurally unlike the rest: it offloads model *parameters* rather than cache, so its effect appears as larger residency rather than cleverer movement. The displaced parameters are charged a per-layer streaming cost; omitting that would model a free lunch rather than a trade.
- **INF2** (#109) — pushes attention computation into the storage device so only results cross PCIe, shrinking transfer by `pcie_reduction` in exchange for a slower in-storage accelerator charged as `storage_compute_ms`.

A `SequentialOffloader` baseline tiers without overlapping anything, giving every other strategy something to be measured against and pinning the honest worst case.

## The finding the comparison produced

Run in a **compute-bound** regime (10ms compute against 6ms fetches), LayerKV, InfiniGen and INF2 all land on exactly 320ms — the pure-compute floor, 100% overlap — and the comparison says nothing, because any strategy that overlaps at all hides everything. That regime is reported but explicitly labelled uninformative rather than quietly dropped.

Run in a **transfer-bound** regime (10ms compute against 40ms fetches), which is the long-context situation the family exists for, they separate sharply — and LayerKV's overlap efficiency lands on exactly 25.0%, which is precisely the compute-to-transfer ratio. That is the structural result: **pure overlap is bounded by that ratio and cannot exceed it**, so the strategies that reach the compute floor are the ones that also reduce how much has to move (InfiniGen at 1.00x by fetching a quarter of the entries; INF2 at 1.23x by shrinking bus traffic tenfold). CAG.md frames the art of this family as "hiding transfer latency behind computation that's already happening anyway"; the measurement's correction is that hiding alone caps out, and volume reduction is what lets hiding finish the job.

## Testing plan

Unit tests only — there is no live half, and inventing one would mean asserting simulated numbers against hardware that cannot produce them. `test_offloading_metrics.py` covers the three metrics including their boundary behavior (a run that moved nothing scores 1.0 overlap rather than dividing by zero); `test_offloading_strategies.py` pins each strategy's distinguishing mechanism — that the baseline hides nothing by construction, that LayerKV hides everything when compute outlasts fetch and only partially when it does not, that InfiniGen moves strictly less than LayerKV, that KVPR stalls for the longer of its two halves rather than their sum, that Oneiros raises residency where nothing else does and still pays for the parameters it displaced, and that INF2 shrinks bus traffic while being charged for the slower accelerator.

## What this batch does not do

- **Does not measure real hardware** — stated above and repeated in the report; every quoted millisecond is simulated under named parameters.
- **Does not implement a real three-tier store** — the cold tier is a latency parameter, not an SSD-backed implementation; building that belongs with a serving integration, not with comparing scheduling mechanisms.
- **Does not build the Long Context Champion combination** — #42/#133 becomes buildable now that Eviction, Compression, and Offloading all exist, but composing them is its own batch.
