# CAG Batch I: The Three Combinations — Design Spec

**Scope:** #41/#131 (High-Throughput Serving), #42/#133 (Long Context Champion), #43/#134 (Real-Time Chat). See `docs/architecture/CAG.md`, "High-synergy combinations". This is the last CAG work; all nine constituent techniques were built in batches C through H.

## One question, asked of all three

CAG.md makes a specific structural claim about these archetypes, and it is stronger than "these work together": they "aren't three independently good ideas bolted together, but a genuine pipeline where each technique's output is smaller or better-organized input for the next one — which is exactly the property that separates an A+ combination from techniques that are merely compatible without reinforcing each other."

That is falsifiable, so the batch is built around falsifying it. Two hypotheses, both expressible as numbers: if the techniques genuinely compound, a composed pipeline approaches the **product** of its stages' individual reductions; if they are merely compatible, it lands near the **best single stage**. `synergy_score` places a measured result between those two poles.

## Why the obvious measurement is not sufficient, and what replaced it

The synergy score alone turned out to be nearly tautological for a pipeline of this shape, which is worth stating plainly rather than reporting 1.000 as a triumph. A pipeline that reduces toward a **fixed** GPU capacity telescopes: measure eviction, compression and offloading as ratios into a fixed target and their product is `original / capacity` regardless of how the stages divide the labour. Running Archetype A at three compression group sizes made this visible — the stage attributions shifted substantially (compression 1.6x to 5.33x, offloading 10.0x down to 3.0x) while the measured total sat at exactly 128.0x in all three, because the total was pinned by the target, not discovered.

`interference_drift` is what the batch actually rests on. It compares each stage's ratio **inside** the pipeline against that same stage run **alone**, so it cannot telescope: 0.0 means the stages compose cleanly, and a negative value means an earlier stage handed the next one something harder to work with than it started with — precisely the case the naive product overstates.

## The three archetypes

**Archetype A, Long Context Champion (#42/#133)** — `LongContextPipeline` composes the real `H2OEvictor`, a real compressor, and a real offloader in CAG.md's stated order, over a real `distilgpt2` KV tensor and real accumulated attention scores. It composes components the earlier batches built rather than reimplementing any of them: if the pipeline compounds, that has to be because those components genuinely feed each other, not because a combination-specific reimplementation was tuned to make it look so. This is the one combination test that runs in CI, since it needs only CPU.

**Archetype B, High-Throughput Serving (#41/#131)** — live against vLLM, all three techniques active at once on the workload CAG.md names for it. The workload deliberately **interleaves** four tenants rather than grouping them, because interleaved arrival is what a real multi-tenant endpoint sees and is where scheduling has to do actual work; grouping by tenant first would hand the result to prefix caching alone. Prompts are salted per run for the reason the Cache-Aware Batching batch established the hard way.

**Archetype C, Real-Time Chat (#43/#134)** — `RealTimeChatPipeline` runs a session strategy, bounds its growth with eviction, and shrinks the survivors with compression, testing CAG.md's literal claim that this is "what let the fiftieth turn of a conversation run as fast as the first."

## A measurement bug the tests caught, worth recording

The first version of both pipelines measured compression by counting payload **elements**, and scored KIVI at 0.66x — i.e. as inflation. Compression's entire mechanism is fewer **bits** per element, not fewer elements: a quantizer stores the same count of values at a quarter of the width, and once per-channel scales and residuals are added the element count genuinely rises. `payload_footprint.py` replaced it, charging ints at the quantized width and floats at 32, which is faithful for a quantizer, for a low-rank method whose payload is entirely float, and for a hybrid carrying both. A test pins the failure mode directly so it cannot return.

## Testing plan

1. **Unit tests** — the metrics (including that `synergy_score` can report a combination that actively *hurt*, and that `interference_drift` distinguishes the clean case from the interfering one), the bit accounting, and both composable pipelines' stage ordering.
2. **Two integration tests** — Archetype A against a real `distilgpt2` cache, which **runs in CI**; Archetype B against a live vLLM server, which cannot, and skips with a stated reason.

## What this batch does not do

- **Does not resolve CAG.md's pair-level A-versus-A+ grading for Archetype C.** The source flags Eviction + Multi-Turn Caching as scoring A rather than A+. This batch's cost model measures footprint and recompute; whatever that grade reflects is not visible in those terms, and claiming it confirmed or refuted would overstate what was measured.
- **Does not measure output quality for any archetype** — every combination here is scored on footprint, reuse and latency, not on whether answers stayed good.
