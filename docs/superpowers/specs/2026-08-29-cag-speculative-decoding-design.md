# CAG Batch C: Speculative Decoding — Design Spec

**Scope:** #37 (story), #120 (Medusa), #121 (Lookahead Decoding), #122 (Prompt Lookup Decoding). See `docs/architecture/CAG.md`'s "Speculative decoding: drafting ahead and verifying in parallel" section and `docs/inputs/concepts/advanced_cag_concepts.md`, Concept 7, for the source material.

## What CAG.md asks for

Autoregressive generation costs one full forward pass per token — O(N²) compute for N tokens. Speculative decoding breaks that: a cheap source proposes K candidate tokens, the real target model verifies all K in a single forward pass, and the pipeline accepts every token up through the first mismatch (plus one "bonus" token the target model's own distribution at the mismatch point gives for free, since it came from the same forward pass) before retrying from there. Three variants change only *where* the candidates come from — Medusa (extra prediction heads on the target model itself), Lookahead Decoding (an n-gram cache built from prior context), Prompt Lookup Decoding (tokens reused straight from the prompt) — the propose-verify-accept loop underneath all three is identical.

This is genuinely CPU-feasible today: `transformers` (already a declared dependency after Batch B) is real and instantiable against a real small causal LM with no GPU needed — the same `distilgpt2`/CPU setup Batch B's integration test already proved out.

## Approach

**Domain** (`src/cag/domain/`), framework-free:

- `entities.py` additions: `VerificationResult` (accepted_tokens: list[int], bonus_token: int | None) — what one verification forward pass produces; `SpeculativeDecodingRun` (generated_tokens: list[int], forward_passes: int, tokens_accepted_from_candidates: int, tokens_proposed: int) — a real, measured record of one full generation run, from which acceptance rate and forward-pass reduction are both directly computable.
- `ports.py` additions, two ports (mirroring Batch B's precedent of splitting a port only when the underlying operation genuinely differs, not by default):
  - `CandidateGenerator(ABC)`: `propose(self, prompt_tokens: list[int], generated_tokens: list[int], num_candidates: int) -> list[int]`. Takes both the original prompt and the tokens generated so far, since the three variants draw from different slices of that same context (Prompt Lookup: the prompt; Lookahead: the generated tail; Medusa: neither — it doesn't search text at all, its candidates come from the model's own extra heads, but the uniform signature costs it nothing to ignore what it doesn't need).
  - `TargetModel(ABC)`: `verify_candidates(self, tokens: list[int], candidates: list[int]) -> VerificationResult`. One real forward pass over `tokens + candidates`; compares each candidate position's greedy prediction against what the target model itself would have generated there, accepts the prefix up to first mismatch, returns it plus the bonus token.

**Application** (`src/cag/application/`):

- `SpeculativeDecode` — `execute(target_model: TargetModel, candidate_generator: CandidateGenerator, prompt_tokens: list[int], max_new_tokens: int, num_candidates: int) -> SpeculativeDecodingRun`. The one shared loop: propose, verify, accept + append the bonus token, repeat until `max_new_tokens` is reached, tracking forward-pass count against tokens produced — the real, honest "how much did this actually reduce forward passes vs. naive autoregressive" measurement CAG.md's own O(N²) → O(N²/K) claim is about.

**Infrastructure** (`src/cag/infrastructure/`):

- `hf_target_model.py` — `HFTargetModel(TargetModel)`, wrapping a real `transformers` causal LM + tokenizer. `verify_candidates` runs one real forward pass over the concatenated tokens, argmaxes each candidate position's logits, and compares.
- `prompt_lookup_candidate_generator.py` — `PromptLookupCandidateGenerator(CandidateGenerator)`: takes the last few generated tokens as an n-gram key, searches the *prompt* for a matching occurrence, proposes whatever tokens followed that match in the prompt. Matches `transformers`' own built-in `PromptLookupCandidateGenerator` mechanism structurally (confirmed present in this project's installed `transformers` 5.15.1), reimplemented here as real CAG-module code rather than called through, consistent with how Batch B built its own compressors instead of wrapping an existing library.
- `lookahead_candidate_generator.py` — `LookaheadCandidateGenerator(CandidateGenerator)`: the same n-gram-match mechanism as Prompt Lookup, but searching the *generated tail* (an n-gram cache built from prior context, growing as generation proceeds) instead of the prompt — CAG.md's own distinguishing description for this variant. **Disclosed simplification**: the original Lookahead Decoding paper's full mechanism is a parallel Jacobi-iteration decoding scheme that speculates and verifies multiple n-gram trajectories simultaneously; this implementation reproduces its distinguishing *candidate source* (an n-gram cache from prior context, not the prompt and not a trained head) through the same shared propose-verify-accept loop the other two variants use, not the original paper's own parallel-trajectory decoding algorithm.
- `medusa_candidate_generator.py` — `MedusaCandidateGenerator(CandidateGenerator)`: K extra linear prediction heads (`hidden_size → vocab_size`, matching the target model's own `lm_head` shape) attached to the target model's final hidden state, each head predicting one further position (head 0 predicts the next token, head 1 the one after, etc.) in the *same* forward pass that produces the ordinary next-token logits — real architecture, one real extra forward computation, not a second model. **Disclosed simplification**: real Medusa heads are fine-tuned on a training run entirely out of this batch's scope; these heads are initialized by copying the target model's own `lm_head` weights (a real, defensible warm-start — position i+1's most likely token is often a reasonable prior for position i+2 too, especially on repetitive or structured text) rather than left randomly initialized, which would make acceptance-rate measurement meaningless. The report will honestly show whatever acceptance rate this untrained warm-start actually achieves, not a number tuned to look good.

## Testing plan

Unit tests, fast and deterministic, no model needed:
- `PromptLookupCandidateGenerator` and `LookaheadCandidateGenerator` against synthetic token-id sequences with known, constructed n-gram repeats — exact match, no match, multiple candidate matches (longest/most-recent-preferred), edge cases (empty generated tokens, n-gram longer than available context).
- `SpeculativeDecode`'s propose-verify-accept loop against a fake `TargetModel`/`CandidateGenerator` pair (matching this project's own established unit-test convention of fakes for ports, real infrastructure only in integration tests) — full acceptance, partial acceptance, zero acceptance (falls back to one token per forward pass, the correctness floor speculative decoding must never fall below), and the forward-pass/token-count bookkeeping itself.

One integration test, using the real `distilgpt2` CPU setup Batch B already proved out:
- Run all three variants' full pipeline (via `HFTargetModel` + each real `CandidateGenerator`, through the real `SpeculativeDecode` loop) against a real prompt, and report the real, measured forward-pass count and acceptance rate for each — compared honestly against a naive-autoregressive baseline's forward-pass count for the same number of generated tokens. Prompt Lookup gets a prompt specifically containing text likely to be echoed (matching CAG.md's own "works well when the expected output is likely to repeat... material already in the prompt"); Lookahead and Medusa get a more generic prompt, since their candidate source doesn't depend on prompt-echoing the same way.

"Real result documented" for each of the three task issues means the actual measured forward-pass reduction and acceptance rate from this integration test, pasted into that issue's closing comment — including honest disclosure where a variant's real measured speedup falls short of CAG.md's stated "2-3x typical, up to 5x," the same disclosure standard every batch in this project already holds itself to.

## What this batch does not do

- **Does not implement draft-model-based speculative decoding** (a wholly separate small model, `AssistedCandidateGenerator`'s real-world shape) — CAG.md's own three named variants (Medusa, Lookahead, Prompt Lookup) are specifically the ones that avoid hosting a second model, which is the whole point of this batch's scope.
- **Does not train real Medusa heads** — disclosed above; the warm-start initialization is a real, defensible simplification, not a claim of matching a trained model's acceptance rate.
- **Does not implement Lookahead Decoding's original parallel Jacobi-trajectory algorithm** — disclosed above; this batch reproduces the variant's distinguishing candidate source through the shared loop, not the source paper's own decoding algorithm.
- **Does not require GPU or vLLM** — a real CPU forward pass over a real small model is sufficient to honestly measure forward-pass reduction and acceptance rate for all three variants.
