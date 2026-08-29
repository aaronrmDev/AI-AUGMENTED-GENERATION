# CAG Batch B: KV Cache Compression — Design Spec

**Scope:** #31 (story), #97 (KIVI), #99 (KVQuant), #100 (PALU), #101 (MiniCache), #103 (ShadowKV) — five distinct compression algorithms sharing one story. See `docs/architecture/CAG.md`'s KV Cache Compression section and `docs/inputs/concepts/advanced_cag_concepts.md`, Concept 2, for the source material each implementation below is grounded in.

## Why this batch, and why not Prefix Caching next

CAG Batch A (Alternative Attention) established this track's own real code and review cadence. The next candidate in sequence per the "CPU/Ollama-feasible techniques first, GPU setup later" plan was Prefix Caching (#32/#127) — but `docs/testing/TESTING.md`'s own stated testing strategy commits that specific technique to being validated "against vLLM itself rather than a stand-in," because the claim actually under test is the Latency-Adaptive Fallback Cascade's unconditional trust in a cache hit — invalidation-on-prefix-change and eviction-under-memory-pressure correctness, which only a real serving engine's block manager and eviction policy can demonstrate. A hand-rolled `transformers`-based prefix cache (spiked and confirmed technically working — real KV continuation via `past_key_values`, ~10x measured speedup, identical logits between cold and warm paths) would produce real numbers, but not the numbers this project already committed to needing. Prefix Caching moves to the GPU-dependent queue alongside Eviction, PagedAttention, Cache-Aware Batching, Hybrid Offloading, Multi-Turn Caching, and the three Combinations.

KV Cache Compression doesn't carry that same commitment. Its correctness question — does compressing a KV tensor's precision, dimensionality, or cross-layer redundancy preserve enough signal for correct generation — is self-contained: it can be honestly measured against a real KV tensor pulled from a real small model's real forward pass, with no serving-engine invalidation or cascade-trust behavior involved. `docs/testing/TESTING.md` names only prefix-caching specifically as needing vLLM; nothing in it or `docs/decisions/adr/0001-vllm-over-sglang.md` extends that requirement to compression.

## What CAG.md and the source concept doc actually specify

Five methods, all reducing an already-evicted KV cache's remaining footprint by precision, dimensionality, or redundancy rather than by discarding tokens outright (that's Eviction's job, a separate, deferred technique):

| Method | Category | Mechanism | Stated compression |
|---|---|---|---|
| KIVI | Quantization | Per-channel quantization for keys, per-token for values, dynamic grouping (32 tokens/group), recent tokens kept full-precision as residuals until a threshold merges them in | ~4x |
| KVQuant | Quantization | Quantizes before the RoPE positional transform, non-uniform quantization levels, explicitly preserves outlier values a uniform scheme would clip | Up to 8x |
| PALU | Low-rank | SVD-decomposes each KV projection matrix W into A × B, caches only the compact latent H = X × A, reconstructs as H × B when needed | ~4x |
| MiniCache | Cross-layer | SLERP (spherical linear interpolation) between two adjacent layers' similar KV states, storing one interpolated representation instead of both | ~2-3x |
| ShadowKV | Hybrid | Offloads value tensors, keeps a low-rank representation of keys, layers sparsity and quantization on top of both | ~6x |

Four of the five (KIVI, KVQuant, PALU, ShadowKV) compress a single layer's KV tensor in isolation. MiniCache is structurally different — it takes **two** adjacent layers' tensors and produces one shared representation, which is a different-shaped operation than "compress this one tensor," not a variant of it.

## Approach

**Domain** (`src/cag/domain/`), framework-free per this project's hexagonal convention, matching how RAG's `ChatModel`/`EmbeddingModel` ports use plain `list[float]` rather than a framework tensor type at the port boundary:

- `entities.py` — `CompressedKV` (method: str, payload: dict[str, Any], original_shape: tuple[int, int]): a uniform result envelope; `payload`'s actual contents are method-specific (quantized ints + scales for KIVI/KVQuant, a latent matrix + reconstruction matrix for PALU, an interpolated representation + blend metadata for MiniCache, a composite of the above for ShadowKV), which is why it stays a dict rather than a shared frozen shape — forcing five genuinely different compressed representations into one fixed dataclass would be the same mistake as forcing MiniCache behind the single-tensor port.
- `ports.py` — two ports, not one, because MiniCache's input shape genuinely differs:
  - `KVCacheCompressor(ABC)`: `compress(kv: list[list[float]]) -> CompressedKV`, `decompress(compressed: CompressedKV) -> list[list[float]]`. Implemented by KIVI, KVQuant, PALU, ShadowKV.
  - `CrossLayerKVCompressor(ABC)`: `compress(layer_a: list[list[float]], layer_b: list[list[float]]) -> CompressedKV`, `decompress(compressed: CompressedKV) -> tuple[list[list[float]], list[list[float]]]`. Implemented by MiniCache alone — the same reasoning this project already used for keeping `Archive` outside `GateMemories`'s default composition in MAG Batch E: a technique that doesn't share the common shape stays honestly outside the shared interface rather than being forced in.
- `compression_metrics.py` — pure functions: `reconstruction_error(original, reconstructed) -> float` (mean absolute error) and `compression_ratio(original_element_count, compressed_bit_width, ...) -> float`, so each compressor's own claimed ratio is computed the same way rather than five different ad hoc calculations.

**Infrastructure** (`src/cag/infrastructure/`), one file per method, each a real, working implementation of the mechanism described above — actual quantization arithmetic (scale/zero-point, not a placeholder), actual SVD via `torch.linalg.svd`, actual SLERP interpolation, not simplified stand-ins that only look like the algorithm:

- `kivi_compressor.py`, `kvquant_compressor.py`, `palu_compressor.py`, `shadowkv_compressor.py` — each implementing `KVCacheCompressor`.
- `minicache_compressor.py` — implementing `CrossLayerKVCompressor`.

No application layer yet, matching Batch A's own disclosed reasoning: nothing composes these five yet (that would be Cache-Aware Batching or a future serving-configuration use case's job, once serving infrastructure exists to actually route requests through a compression choice).

## Testing plan

Two tiers, matching this project's own test-pyramid reasoning for why some CAG correctness questions need real dependencies:

1. **Unit tests, one file per compressor** (`tests/unit/test_kivi_compressor.py`, etc.), fast and deterministic: round-trip a small synthetic tensor with known values through compress/decompress, assert the reconstruction error stays within a tolerance appropriate to the method's own stated precision loss, and assert the measured compression ratio via `compression_metrics.py` is in the neighborhood of CAG.md's stated figure for that method (not asserting an exact match — these are architectural techniques with tunable parameters, not fixed constants, and the source's own figures are themselves approximate, several stated as "~"). `compression_metrics.py` itself gets its own direct unit tests.
2. **One integration test** (`tests/integration/test_kv_cache_compression_against_real_model.py`) using the same `past_key_values` continuation mechanism the Prefix Caching spike already proved works in this environment, but against `distilgpt2` rather than the spike's `sshleifer/tiny-gpt2` — the spike's model has `n_embd=2` (a 1-dimensional head), too small for any of these five methods to meaningfully compress; `distilgpt2` (768 hidden, 12 heads, 64 dims per head, 6 layers, CPU-fast, no GPU needed) gives each method a real dimensional space to reduce. Extract a real KV tensor from a real forward pass, compress and decompress it through each of the five methods, splice the reconstructed tensor back into the model's cache, and confirm continued generation produces next-token logits close to the uncompressed baseline — the real "does this preserve enough to not corrupt inference" check CAG.md's own "Key Takeaway" is about, not just a synthetic round-trip.

"Real result documented" for each of the five task issues means the actual measured compression ratio and reconstruction error from these tests, pasted into that issue's closing comment — including honest disclosure if a measured ratio falls short of or exceeds CAG.md's stated figure for that method, the same disclosure standard every MAG batch's evaluation report already holds itself to.

## What this batch does not do

- **Does not implement Eviction** — a separate, deferred (GPU-dependent) technique; this batch compresses what eviction would have left behind, not what eviction removes.
- **Does not implement a serving-time compression *selection* policy** (choosing which method to apply, when, under what memory pressure) — that's Cache-Aware Batching's or a future orchestration concern, not this batch's.
- **Does not attempt bit-exact reproduction of any paper's published benchmark numbers** — CAG.md's own figures are themselves architecture-level approximations ("~4x", "up to 8x"), and this batch's tensors, model, and parameter choices (group sizes, quantization bit-width, SVD rank) are its own, disclosed choices, not a reproduction of any specific paper's exact experimental setup.
- **Does not require GPU or vLLM** — real CPU tensors from a real small model are sufficient to honestly measure compression ratio and reconstruction fidelity for all five methods.
