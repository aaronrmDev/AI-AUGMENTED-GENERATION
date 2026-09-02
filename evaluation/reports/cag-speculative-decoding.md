# CAG Batch C: Speculative Decoding — Live Measurement Report

**Scope:** #37 (story), #120 (Medusa), #121 (Lookahead Decoding), #122 (Prompt Lookup Decoding). See `docs/superpowers/specs/2026-08-29-cag-speculative-decoding-design.md` for the design.

## What CAG.md asks for

Autoregressive generation costs one full forward pass per token — O(N²) compute for N tokens. Speculative decoding breaks that: a cheap source proposes K candidate tokens, the real target model verifies all K in a single forward pass, and the loop accepts every token up through the first mismatch, plus one bonus token the same forward pass gives for free. Three named variants change only *where* the candidates come from — Medusa (extra prediction heads on the target model), Lookahead Decoding (an n-gram cache from prior generated context), Prompt Lookup Decoding (tokens reused from the prompt) — the propose-verify-accept loop underneath is identical across all three. This is genuinely CPU-feasible: a real forward pass over a real small model (`distilgpt2`) needs no GPU or vLLM, the same setup Batch B's own integration test already proved out.

## The shared loop and one real target model

`SpeculativeDecode` (`src/cag/application/speculative_decode.py`) is the one loop all three variants share, differing only in which `CandidateGenerator` gets plugged in. `HFTargetModel` (`src/cag/infrastructure/hf_target_model.py`) is the real verification: one forward pass over `tokens + candidates`, comparing each candidate position's greedy prediction against the target model's own — spike-verified before writing it that the exact indexing (`logits[len(tokens) - 1 + i]` predicts `candidates[i]`) reproduces a real model's own greedy continuation exactly, including the free bonus token.

## Prompt Lookup Decoding — reuses tokens from the prompt

Searches the prompt for the current context's trailing n-gram, proposing whatever followed that match. On a prompt built to make echo likely ("Repeat the phrase three times..."), measured **3.00x fewer forward passes, 81.8% of proposed candidates accepted** — genuinely landing in CAG.md's stated "2-3x typical" range, at the high end.

A real algorithm bug surfaced and was fixed along the way: the self-match guard (excluding the trivial case where the search tail matches the prompt's own literal trailing position) was only applied when nothing had been generated yet. But `find_ngram_continuation`'s fallback to shorter n-grams can produce a search key that coincidentally equals the prompt's own trailing token even with generation already underway — confirmed empirically (`propose([5,6,7,5,6,7], [7], 3)` returned `[]` instead of the real earlier match `[5,6,7]`). Fixing this (applying the guard unconditionally) also genuinely improved the measured real-model result: 2.40x → 3.00x, a real correctness fix producing a real speedup improvement, not a coincidence.

## Lookahead Decoding — reuses tokens from generated text

The same n-gram-match mechanism, but searching the generated tail itself rather than the prompt — confirmed to genuinely ignore the prompt even when the prompt alone would have matched. On a prompt hoped to produce repetitive output ("List numbers from one to five..."), measured **1.09x fewer forward passes, 12.5% of proposed candidates accepted** — the model's actual greedy continuation ("The first line is the first line, and the second") repeated less than hoped, an honest result below CAG.md's stated range, reported as measured rather than adjusted to look better. Disclosed simplification: this reproduces the variant's distinguishing candidate *source* (an n-gram cache from prior context, not a trained head or the prompt) through the same shared loop every variant uses, not the original paper's own parallel Jacobi-iteration decoding algorithm.

## Medusa — extra prediction heads on the target model

`num_heads` extra linear layers attached to the target model's own final hidden state, all applied in the same forward pass that produces the ordinary logits — no second model, no autoregressive drafting. Disclosed simplification: real Medusa heads are fine-tuned; these are warm-started as exact copies of the target model's own output-embedding layer. That warm start makes head 0 mathematically identical to a real next-token prediction — but since every head is an *identical* copy applied to the *identical* hidden state, every head produces the identical top prediction. Heads 1+ have no way to know they're supposed to predict a different, later position; only real training with a future-offset-aware loss teaches that.

Measured **2.00x fewer forward passes, 25.0% of proposed candidates accepted** — mechanistically exactly what the disclosed limitation predicts: every round accepts candidate[0] (guaranteed, since head 0 duplicates the target's own prediction) plus one free bonus token, while candidates 1-3 are rejected every time. 6 rounds × (1 accepted + 1 bonus) = 12 tokens in 6 forward passes. Confirmed deterministically in a dedicated unit test (not just observed once against the real model): every head produces the identical prediction given an identical hidden state, by construction.

## What two full review rounds caught

**Round one** (four dimensions) confirmed 7 of 7 raw findings, two of them genuine correctness bugs, not just test gaps:

- **`tokens_accepted_from_candidates` overcounted whenever a verification batch's accepted tokens overshot the remaining budget.** The counter was incremented by the full accepted-token count *before* truncation to `max_new_tokens` was applied, so tokens that were later discarded still counted as "accepted" — silently inflating the reported acceptance rate on exactly the runs (a late batch fully accepted, or `num_candidates` not evenly dividing `max_new_tokens`) this project's own evaluation methodology depends on being accurate. Fixed by capping accepted tokens at the remaining budget *before* counting, which also made the old post-hoc truncation branch provably unreachable — removed rather than left as dead defensive code.
- **`PromptLookupCandidateGenerator`'s self-match guard bug**, described above.
- **The Prompt Lookup and Lookahead integration tests asserted only the loop's own trivially-guaranteed correctness floor** (`forward_passes <= max_new_tokens`), which a completely non-functional candidate generator would still pass — confirmed by substituting a `propose()` that always returns `[]` into the real test and watching both assertions pass unchanged (12 forward passes for 12 tokens, pure naive fallback). Fixed by asserting real, non-trivial acceptance occurred.
- **`FakeTargetModel`'s scripted responses never depended on the `candidates` argument it received**, and no unit test asserted on it — confirmed by mutating `SpeculativeDecode` to discard the real candidates entirely and watching all 5 unit tests still pass.
- **No dedicated unit test existed for `HFTargetModel` or `MedusaCandidateGenerator`** — the two classes that run a real torch forward pass — unlike every one of Batch B's five compressors, each covered by a fast, deterministic, no-download unit test. Closed with `test_hf_target_model.py` and `test_medusa_candidate_generator.py`, both using tiny synthetic fake models instead of a real download.
- A defensive forward-progress guard (not a live bug, since the one shipped `TargetModel` always returns a real bonus token) and the recurring `OVERVIEW.md` build-status staleness (already caught and fixed once each for Batches A and B).

**Round two** (a scoped re-review of round one's own fix wave) confirmed 2 of 2 raw findings, both in the fix wave's own new code:

- A stale comment in the shared `ngram_search.py` still described the self-match exclusion as conditional, after the caller's own fix made it unconditional.
- **The new `HFTargetModel`/`MedusaCandidateGenerator` unit tests' fake model was itself too weak to catch a real indexing bug** — it broadcast the identical hidden vector across every position and exposed only one hidden-state layer, so a deliberately reintroduced version of exactly the bug the tests exist to catch (reading `hidden_states[0][0, 0]` instead of `hidden_states[-1][0, -1]`) passed all 4 Medusa tests unchanged. Fixed with a fake giving every `(layer, position)` pair its own individually identifiable predicted token; the same reintroduced mutation now correctly fails 2 of 4 tests.

## What this batch does not do

- **Does not implement draft-model-based speculative decoding** (a wholly separate small model) — CAG.md's three named variants specifically avoid hosting a second model, which is this batch's whole scope.
- **Does not train real Medusa heads** — disclosed above; the warm-start is a real, defensible simplification whose limitation is honestly measured and reported, not hidden.
- **Does not implement Lookahead Decoding's original parallel Jacobi-trajectory algorithm** — disclosed above.
- **Does not require GPU or vLLM** — a real CPU forward pass over a real small model was sufficient to honestly measure all three variants.

## The numbers

Unit tests: 494 (start of batch, after CAG Batch B) → 504 (initial implementation, unchanged through both review rounds — round two strengthened existing tests rather than adding new ones). Integration tests: 183 → 186 (three new tests, one per variant). Every real-model measurement re-verified against `distilgpt2` after each fix wave — three full live runs across this batch, greedy/deterministic throughout (no sampling anywhere in this batch), so results are exact facts about these specific prompts and model, not run-to-run noise. Full integration suite (186 tests, the entire repository) re-verified green on `develop` after merge.
