# CAG Batch I: The Three Combinations — Measurement Report

**Scope:** #41/#131 (High-Throughput Serving), #42/#133 (Long Context Champion), #43/#134 (Real-Time Chat). See `docs/superpowers/specs/2026-09-07-cag-combinations-design.md` for the design. Final CAG batch; all nine constituent techniques were built in batches C through H.

## The claim, and the measurement that nearly rubber-stamped it

CAG.md says these archetypes "aren't three independently good ideas bolted together, but a genuine pipeline where each technique's output is smaller or better-organized input for the next one." The natural test is whether a composed pipeline achieves the **product** of its stages' reductions (compounding) or merely the **best single stage** (compatible but not reinforcing).

Measured that way, Archetype A scores a synergy of **1.000** — the full product, apparently perfect compounding. That number should not be trusted, and this report says so before quoting it as a finding.

Running the same pipeline at three compression group sizes shows why:

| Compression group size | Stage reductions (evict, compress, offload) | Measured total | Synergy |
|---|---|---|---|
| 4 | 8.0x, 1.60x, 10.0x | **128.0x** | 1.000 |
| 16 | 8.0x, 4.00x, 4.0x | **128.0x** | 1.000 |
| 32 | 8.0x, 5.33x, 3.0x | **128.0x** | 1.000 |

The stage attributions move substantially — compression more than triples in effectiveness across these rows — and the total does not move at all. A pipeline reducing toward a **fixed** GPU capacity telescopes: the product is `original / capacity` however the stages divide the labour. The 1.000 was pinned by the target, not discovered in the data.

## What actually tests the claim: interference

`interference_drift` compares each stage's ratio **inside** the pipeline against the same stage run **alone**. It cannot telescope, and on a real `distilgpt2` cache it separates the two compressors sharply:

| Compressor | Standalone | After eviction | Drift |
|---|---|---|---|
| KIVI (quantization) | 2.54x | 2.67x | **+5.0%** |
| PALU (low-rank) | 7.91x | 2.71x | **−65.7%** |

**Quantization composes cleanly with eviction; low-rank compression does not.** Quantization is per-value, so removing tokens barely changes how well the survivors quantize — the +5.0% is a boundary effect of KIVI's trailing full-precision residual group, not a trend. PALU's ratio depends on sequence length relative to its rank, so the shorter the sequence eviction leaves, the less its fixed reconstruction-matrix overhead is amortised: on synthetic data the drift ran −29.2% at aggressive eviction, and on the real cache **−65.7%**.

This is a real correction to CAG.md. The source grades Archetype A's every pair A+ without qualification, but **the archetype's synergy depends on which compressor implements the middle stage** — pair a low-rank method with aggressive eviction and the naive product overstates the pipeline by roughly a factor of three. That distinction does not appear anywhere in the source's compatibility matrix.

## Archetype A: Long Context Champion, on a real cache

Composing the real `H2OEvictor`, `KIVICompressor` and `LayerKVOffloader` over a real `distilgpt2` KV tensor and real accumulated attention scores, in CAG.md's stated order:

**33 tokens → 16 after eviction → 6.00 slots after compression → 4 GPU-resident**, stage reductions 2.06x / 2.67x / 1.5x.

Each stage genuinely hands the next one less to handle, which is the ordering claim, asserted as an ordering rather than inferred from a total. The compressor is handed only the survivors and never sees the evicted rows — the relationship the archetype depends on.

## Archetype B: High-Throughput Serving, live

All three serving-tier techniques active at once against a real vLLM server, on the workload CAG.md names — 4 tenants, 6 requests each, **arriving interleaved** rather than grouped, because interleaved arrival is what a real multi-tenant endpoint sees and grouping by tenant would hand the result to prefix caching alone:

| | Latency | Prefix-cache hit rate |
|---|---|---|
| 24 requests, sequential | 1.372s | **79.4%** |
| 24 requests, concurrent | 0.169s | 76.9% |
| | **8.1x** | |

Roughly four fifths of all prompt tokens are served from shared blocks despite the tenants arriving mixed together — each tenant's system prompt is prefilled once and reused across that tenant's remaining requests, which is CAG.md's "one prefill ends up serving hundreds of requests instead of one" at this batch's scale. The hit rate holds at ~77-80% whether the requests are issued sequentially or concurrently, which means the 8.1x is the scheduling tier contributing throughput **on top of** what caching already delivered, rather than the two being different views of one effect.

## Archetype C: Real-Time Chat

Over a 50-turn conversation with a 64-token residency budget:

- **Turn 50 costs exactly what turn 1 cost** (40 tokens of recompute each) — CAG.md's "what let the fiftieth turn of a conversation run as fast as the first," measured literally.
- Unbounded, the session would hold **2000 tokens** by turn 50; bounded by eviction and shrunk by compression it holds **40** — memory plateaus while the conversation does not.

## What this batch does not establish

- **CAG.md's pair-level A-versus-A+ grading for Archetype C is not resolved.** The source flags Eviction + Multi-Turn Caching as scoring A rather than A+, unlike every pair in the other two archetypes. This batch's cost model measures footprint and recompute; whatever that lower grade reflects is not visible in those terms. Reporting it as confirmed or refuted would overstate the measurement, so it is reported as untested.
- **No archetype was scored on output quality.** All three are measured on footprint, reuse and latency. Whether answers stayed good under aggressive eviction and compression is a judge-harness question this batch did not ask.
- **Archetype B is the only one measured end-to-end inside a real engine.** A and C compose real, tested components but score them through cost models, not by observing a serving engine execute the pipeline.

## A measurement bug the tests caught

Both pipelines initially measured compression by counting payload **elements** and scored KIVI at **0.66x** — as inflation. Compression's mechanism is fewer *bits* per element, not fewer elements; a quantizer stores the same count of values at a quarter of the width, and adding per-channel scales and residuals genuinely raises the count. `payload_footprint.py` replaced the accounting, charging ints at the quantized width and floats at 32 — faithful for a quantizer, for a low-rank method whose payload is entirely float, and for a hybrid carrying both. A test pins the failure mode so it cannot return.

## The numbers

Unit tests: 748 (start of batch, after Multi-Turn Caching) → 777 (29 new: 13 combination metrics, 8 payload footprint, 8 pipelines). Integration tests: 3 new — 1 against a real `distilgpt2` cache which **does run in CI**, and 2 against a live vLLM server which cannot.
