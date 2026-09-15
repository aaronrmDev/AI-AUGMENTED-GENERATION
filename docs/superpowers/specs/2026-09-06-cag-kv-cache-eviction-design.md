# CAG Batch C: KV Cache Eviction — Design Spec

**Scope:** #29 (story), #89 (H2O), #91 (SnapKV), #92 (NACL), #93 (MorphKV), #95 (HASHEVICT), #96 (InfiniPot) — six distinct eviction algorithms sharing one story. See `docs/architecture/CAG.md`'s "KV Cache Eviction" section and `docs/inputs/concepts/advanced_cag_concepts.md`, Concept 1, for the source material each implementation below is grounded in.

## The correction this batch makes to CAG.md

CAG.md and `docs/architecture/OVERVIEW.md` both stated that Eviction, along with five other CAG techniques, "need real GPU/vLLM serving to measure honestly." That claim was already false for two sibling techniques before this batch: KV Cache Compression and Speculative Decoding both shipped as real, live-measured code against a real `distilgpt2` forward pass on CPU, with no GPU or vLLM involved, because their own correctness questions are self-contained. Eviction's correctness question is the same shape: does discarding low-attention tokens preserve enough signal for the model to keep producing a coherent continuation. H2O's own "accumulated attention score across the whole generation so far" is, over one real causal forward pass with `output_attentions=True`, exactly the column sum of the resulting attention matrix — real accumulated mass, not a synthetic stand-in, read off one teacher-forced forward pass instead of a token-by-token decode loop (the same category of honest simplification Speculative Decoding's own single-shot verification already relies on). This batch builds and live-measures all six algorithms on CPU and corrects CAG.md, CONTEXT_GRAPH.md, and OVERVIEW.md accordingly: five techniques remain genuinely GPU/vLLM-blocked (Prefix Caching, PagedAttention, Hybrid Offloading, Multi-Turn Caching, Cache-Aware Batching), not six.

## What CAG.md and the source concept doc actually specify

Six algorithms, all deciding which tokens' KV pairs stay resident during decoding rather than reducing the footprint of what survives (that's Compression's job, already built):

| Method | Mechanism | Distinguishing idea |
|---|---|---|
| H2O | Greedy dynamic eviction by accumulated attention score, plus a fixed recent window | The baseline: "heavy hitters" by running total |
| SnapKV | Observation window plus 1D pooling instead of a running accumulated score | Spreads importance to a token's immediate neighbors |
| NACL | Single-shot scoring via end-of-input-token proxies, plus some random eviction | One pass, not continuous; randomness as a deliberate perturbation |
| MorphKV | Sum/Max Fusion over recent attention patterns, continuously adapting | Recency-based, not full-history |
| HASHEVICT | Locality-sensitive hashing (SimHash) over raw KV vectors, before attention runs | Lighter-weight: no attention computation needed at all |
| InfiniPot | Distills the whole cache via CaP/NuC metrics once it overflows | Selective compression of the whole cache, not per-token selection |

Four of the six (H2O, SnapKV, NACL, MorphKV) fundamentally select a subset of existing token indices to keep. HASHEVICT selects indices too, but from vector similarity rather than any attention-derived score. InfiniPot is structurally different from all five — it doesn't select surviving tokens at all, it blends groups of tokens into representative centroids.

## Approach

**Domain** (`src/cag/domain/`), framework-free per this project's hexagonal convention:

- `entities.py` — `EvictionDecision` (method: str, keep_indices: list[int], evicted_count: int): a uniform result envelope for every algorithm that selects indices (five of six — InfiniPot returns a reduced KV tensor directly instead, since a centroid isn't any single original token's real vector).
- `ports.py` — four ports, not one, because the six algorithms genuinely differ in what they operate on, the same reasoning that split Compression into two ports for MiniCache's sake:
  - `KVCacheEvictor(ABC)`: `select_keep_indices(attention_scores: list[float], budget: int) -> EvictionDecision`. Implemented by H2O, SnapKV, NACL — all three consume a single per-token importance score vector, differing only in how that vector was computed upstream and what each does with it.
  - `RecentPatternEvictor(ABC)`: `select_keep_indices(recent_attention_windows: list[list[float]], budget: int) -> EvictionDecision`. Implemented by MorphKV alone — it needs several recent per-step attention vectors, not one already-accumulated vector.
  - `HashBasedEvictor(ABC)`: `select_keep_indices(kv_vectors: list[list[float]], budget: int) -> EvictionDecision`. Implemented by HASHEVICT alone — it operates on raw KV vectors, with no attention score involved at all.
  - `CacheDistiller(ABC)`: `distill(kv: list[list[float]], budget: int) -> list[list[float]]`. Implemented by InfiniPot alone — it returns a reduced tensor, not an index list.
- `eviction_metrics.py` — pure functions: `accumulate_attention_scores(attention_matrix) -> list[float]` (the real, disclosed simplification described above), `memory_reduction_ratio(original_token_count, kept_token_count) -> float` (Compression's `compression_ratio`, mirrored for token counts), and `retained_attention_mass(scores, keep_indices) -> float` (the real accuracy-risk proxy, Compression's `reconstruction_error` mirrored for eviction: what fraction of total accumulated attention survives in the kept set).

**Infrastructure** (`src/cag/infrastructure/`), one file per method, each a real, working implementation of the mechanism described above rather than a simplified stand-in:

- `h2o_evictor.py` — reserves a fixed recent window unconditionally, fills the remaining budget with the highest-scoring tokens outside that window.
- `snapkv_evictor.py` — applies real 1D max-pooling over the score vector before the same recent-window-plus-top-up selection H2O uses, so a token beside an important one inherits some of that importance.
- `nacl_evictor.py` — deterministic top-k by score for most of the budget, with a `random_fraction` slice instead filled by a seeded random draw from the remaining pool, disclosed as an interpretation of "plus some random eviction" (random_fraction=0.0 degrades to plain top-k, tested directly).
- `morphkv_evictor.py` — computes `fused[i] = sum(window[i] for window in recent_windows) + max(window[i] for window in recent_windows)` and keeps the top-`budget` fused scores; a direct, disclosed reading of "Sum/Max Fusion" with no further reweighting invented on top.
- `hashevict_evictor.py` — real SimHash: a fixed set of random hyperplanes (seeded) turns each raw KV vector into a bucket id; only the most recent occurrence per bucket survives as non-redundant, with a recency-based further trim if deduplication alone doesn't reach the budget.
- `infinipot_distiller.py` — partitions an overflowing cache into `budget` contiguous, near-even groups and replaces each with its centroid. The source's own CaP/NuC metrics aren't reimplemented bit-for-bit (no published formula to match against, the same disclosed-choice position ShadowKV and MiniCache already took in the Compression batch); what's built is a real instance of the same structural idea — whole-cache compaction, not per-token selection.

No application layer yet, matching every prior CAG batch's own disclosed reasoning: nothing composes these six yet (that's Cache-Aware Batching's or a future orchestration concern, once serving infrastructure exists to route requests through an eviction policy choice).

## Testing plan

Two tiers, matching this project's own test-pyramid reasoning:

1. **Unit tests, one file per algorithm plus `eviction_metrics.py` itself** (`tests/unit/test_eviction_metrics.py`, `test_h2o_evictor.py`, `test_snapkv_evictor.py`, `test_nacl_evictor.py`, `test_morphkv_evictor.py`, `test_hashevict_evictor.py`, `test_infinipot_distiller.py`), fast and deterministic, exercising each algorithm's own distinguishing mechanism against hand-constructed scores/vectors with known correct answers (e.g. HASHEVICT's dedup and further-trim tests use exact-duplicate and exact-antipodal vectors specifically because SimHash's bucket assignment is then guaranteed deterministic regardless of the random hyperplanes drawn — a property proven, not merely likely, and confirmed by an actual run rather than assumed).
2. **One integration test** (`tests/integration/test_kv_cache_eviction_against_real_model.py`) against a real `distilgpt2` forward pass with `output_attentions=True` and `attn_implementation="eager"` (the transformers default, SDPA, returns no attention weights at all): extracts the real causal attention matrix, computes real accumulated scores, runs H2O/SnapKV/NACL, and splices the reduced cache back — reducing every layer's keys and values to the identical kept positions, not just one layer, since eviction changes sequence length and a per-layer mismatch would break attention. Confirms H2O's real value proposition directly: retained attention mass measurably beats naive random eviction at the same budget. InfiniPot gets its own splice path (replacing the full layer's tensor with distilled centroids, since a centroid isn't any original token's real vector) and is checked for a finite, non-garbage continuation rather than top-token preservation, the same lighter bar Eviction as a whole is held to below.

"Real result documented" for each of the six task issues means the actual measured numbers from these tests — retained attention mass, memory reduction ratio, and whether the reduced cache still predicted the same top token as the uncompressed baseline — pasted into that issue's closing comment, including honest disclosure where an algorithm's real behavior falls short of or exceeds intuition, the same disclosure standard every prior CAG batch already holds itself to.

## What this batch does not do

- **Does not assert top-token preservation as a pass/fail bar** — eviction is a lossier operation than compression by design (real tokens are discarded outright, not reconstructed from a compressed representation); the integration test measures and reports real numbers rather than forcing an unrealistic bar the way Compression's own near-lossless target could.
- **Does not implement a serving-time eviction *selection* policy** (choosing which algorithm to apply, when, under what memory pressure) — that's Cache-Aware Batching's or a future orchestration concern, not this batch's.
- **Does not attempt bit-exact reproduction of any paper's published benchmark numbers** — CAG.md's own descriptions are architecture-level, and this batch's specific interpretations (NACL's random-fraction model, MorphKV's Sum/Max Fusion formula, InfiniPot's centroid distillation in place of CaP/NuC) are its own disclosed choices.
- **Does not require GPU or vLLM** — a real CPU forward pass against a real small model is sufficient to honestly measure retained attention mass and coherent-continuation behavior for all six methods, correcting CAG.md's prior claim that Eviction needed GPU/vLLM serving to measure honestly.
