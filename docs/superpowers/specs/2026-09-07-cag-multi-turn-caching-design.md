# CAG Batch H: Multi-Turn Caching — Design Spec

**Scope:** #36 (story), #113 (RocketKV-MT), #114 (KVzip), #116 (ShadowKV), #117 (MemServe), #118 (SGLang's approach). See `docs/architecture/CAG.md`, "Multi-turn conversation caching: turning linear cost into constant cost", and `docs/inputs/concepts/advanced_cag_concepts.md`, Concept 6. This is CAG's ninth and final technique, and it unblocks the last combination, Real-Time Chat (#43/#134).

## The claim, and what would actually establish it

CAG.md's headline for this technique is "near-O(1) amortized cost per turn instead of cost that grows with every message." That is a claim about a *growth rate*, and it is worth being precise about what evidence bears on it, because the obvious measurement does not. A smaller session total proves nothing on its own: a strategy can halve a quadratic and still be quadratic. What distinguishes O(1)-per-turn is that per-turn work **stops growing** while the conversation keeps growing.

Both halves of this batch are built around that distinction. The live half measures per-turn newly-computed work against per-turn prompt length and compares their growth rates. The domain half keeps `per_turn_recompute` on every `SessionRun` alongside the total, and `session_metrics.is_flat_per_turn` asks the structural question directly — a test that would pass on `[25, 50, 75, 100]` if it only looked at totals fails here, and there is a test asserting exactly that.

## Half 1 — validated against a real server

A real growing conversation, eight turns, each sending the full accumulated history as any chat application does. Per turn, `vllm:prompt_tokens_total` gives the logical prompt and `vllm:prefix_cache_hits_total` gives what was served from cache; their difference is what the engine genuinely had to prefill.

Measured: the prompt grew 6.5x (59 to 381 tokens) while newly-computed work stayed in a 17-31 token band — a spread of exactly one 16-token cache block, which is where the block boundary happens to fall rather than growth. Latency stayed flat at roughly 0.1s throughout instead of tracking prompt length. Turn 1 is excluded from the steady-state comparison because nothing is cached yet, and including a cold start alongside warm turns would compare a miss against hits.

The prompts are salted per run, for the reason the Cache-Aware Batching batch established the hard way: an identical conversation from a previous run against a long-lived server pre-warms this one's cache and quietly flatters the result.

## Half 2 — the five methods, and an honest baseline

All five implemented under a shared `MultiTurnCacheStrategy` port over a workload model whose turns carry a `parent_turn_id`, so a session can be a **branching agent program** and not only a flat conversation. That is not generality for its own sake: SGLang's entire distinguishing claim is about reuse across a program's control flow, and a linear-only model would have been structurally unable to show it.

- **RocketKV-MT** (#113) — retains everything for future turns while capping what the *current* turn attends over. Models a memory-for-quality trade, not memory-for-compute: nothing is discarded, but the present turn sees less than it could.
- **KVzip** (#114) — query-agnostic compression whose overhead is paid once, up front, and never again. Spreading that cost evenly across turns would hide precisely the amortisation the method claims, so it lands entirely on turn one.
- **ShadowKV** (#116) — composes the real `ShadowKVCompressor` the Compression batch already built rather than reimplementing its arithmetic, exactly as that compressor composes `PALUCompressor`. CAG.md calls this "the same method" doing double duty; a second implementation would be free to drift from the first, and the claim that both roles share a mechanism would become a statement about the prose rather than the code.
- **MemServe** (#117) — disaggregated serving, and the only strategy here reporting non-zero `transfer_tokens`. The elastic pool means nothing is evicted for lack of local room; the cost is that history crosses a network every turn. Modelling the elasticity without the transfer would describe unlimited free memory rather than a serving architecture.
- **SGLang** (#118) — tracks which prefixes it has materialised, so a forked program builds its shared trunk once. On a flat conversation it degrades exactly to incremental append, which CAG.md predicts outright ("rather than a single flat conversation"), and the test asserts that equality as the correct result rather than treating it as a shortfall.

**The baseline needed correcting mid-batch, and the correction is the point.** `IncrementalAppendSession` initially charged only new tokens for every turn regardless of structure — which silently modelled an oracle that shares trunks across branches for free. Since sharing across branches is exactly what SGLang contributes, that version handed SGLang's benefit to the baseline and the two measured identically (500 against 500) on the very workload built to distinguish them. Corrected to be session-keyed — a turn continuing the session in front of it reuses everything, a fork begins a new session whose cache starts empty — the same comparison reads 500 against 1400.

## Testing plan

1. **Unit tests** (`test_session_metrics.py`, `test_multi_turn_sessions.py`), CI-safe: the metrics including the flatness discriminator, and each method's distinguishing mechanism — that RocketKV-MT caps the current turn while retaining full history, that KVzip's per-turn cost falls as the session lengthens, that ShadowKV's continuations cost less than its first turn, that MemServe is alone in moving history across a network and moves none when everything fits locally, and that SGLang beats the session-keyed baseline on a branching program while tying it on a flat one.
2. **One integration test** against a real vLLM server, which cannot run in CI for the same reason as the three batches before it.

## What this batch does not do

- **Does not measure answer quality across turns** — CAG.md also claims "a consistent persona since the agent genuinely remembers the full conversation," which is a quality claim needing a judge harness, not a cache measurement. Only the cost claim is tested here.
- **Does not build the Real-Time Chat combination** — #43/#134 becomes buildable now that Multi-Turn Caching, Eviction, and Compression all exist, but composing them is its own batch.
