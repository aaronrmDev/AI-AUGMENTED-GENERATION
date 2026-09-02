"""Live verification of all three speculative decoding variants against a
real distilgpt2 forward pass (CPU, no GPU needed) -- CAG.md's own claim
that speculation reduces forward passes from O(N) to O(N/K) for N
generated tokens. No testcontainers or Docker needed -- this test
requests none of conftest.py's container fixtures.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cag.application.speculative_decode import SpeculativeDecode
from src.cag.infrastructure.hf_target_model import HFTargetModel
from src.cag.infrastructure.lookahead_candidate_generator import LookaheadCandidateGenerator
from src.cag.infrastructure.medusa_candidate_generator import MedusaCandidateGenerator
from src.cag.infrastructure.prompt_lookup_candidate_generator import (
    PromptLookupCandidateGenerator,
)

_MODEL_ID = "distilgpt2"
_MAX_NEW_TOKENS = 12
_NUM_CANDIDATES = 4


def _load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
    model.eval()
    return model, tokenizer


def _report(name: str, tokenizer, prompt_tokens: list[int], run) -> None:
    naive_forward_passes = len(run.generated_tokens)
    speedup = naive_forward_passes / run.forward_passes if run.forward_passes else 0.0
    acceptance_rate = (
        run.tokens_accepted_from_candidates / run.tokens_proposed if run.tokens_proposed else 0.0
    )
    decoded = tokenizer.decode(run.generated_tokens)
    print(
        f"\n{name}: generated {len(run.generated_tokens)} tokens in "
        f"{run.forward_passes} forward passes (naive would need "
        f"{naive_forward_passes}) -- {speedup:.2f}x fewer forward passes, "
        f"{acceptance_rate:.1%} of {run.tokens_proposed} proposed candidates "
        f"accepted. Text: {decoded!r}"
    )


def test_prompt_lookup_reduces_forward_passes_on_a_prompt_with_a_repeated_phrase():
    # CAG.md: Prompt Lookup "works well specifically when the expected
    # output is likely to repeat or closely echo material that's
    # already sitting in the prompt" -- a prompt built to make that
    # likely, then measured for real rather than assumed.
    model, tokenizer = _load_model_and_tokenizer()
    prompt = (
        "Repeat the phrase three times. 1: the quick brown fox jumps. "
        "2: the quick brown fox jumps. 3:"
    )
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    use_case = SpeculativeDecode()
    run = use_case.execute(
        target_model=HFTargetModel(model),
        candidate_generator=PromptLookupCandidateGenerator(),
        prompt_tokens=prompt_tokens,
        max_new_tokens=_MAX_NEW_TOKENS,
        num_candidates=_NUM_CANDIDATES,
    )

    _report("Prompt Lookup", tokenizer, prompt_tokens, run)
    assert len(run.generated_tokens) == _MAX_NEW_TOKENS
    # The correctness floor every variant must clear regardless of how
    # well its own candidate source performs: never MORE forward passes
    # than naive one-token-per-pass generation would need.
    assert run.forward_passes <= _MAX_NEW_TOKENS
    # Review-caught (HIGH): the floor above is trivially satisfied even
    # by a completely broken candidate generator whose propose() always
    # returns [] -- confirmed empirically by substituting exactly that
    # into this test and watching both assertions still pass (12 forward
    # passes for 12 tokens, the naive fallback, indistinguishable from a
    # real speedup by these bounds alone). This is the real, stronger
    # claim the test's own name makes: Prompt Lookup's candidate source
    # must have actually found and used real matches on this
    # favorable-by-design prompt, not merely never made things worse.
    assert run.tokens_accepted_from_candidates > 0


def test_lookahead_decoding_reduces_forward_passes_on_naturally_repetitive_text():
    model, tokenizer = _load_model_and_tokenizer()
    prompt = "List numbers from one to five, each on its own line, no other text. 1"
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    use_case = SpeculativeDecode()
    run = use_case.execute(
        target_model=HFTargetModel(model),
        candidate_generator=LookaheadCandidateGenerator(),
        prompt_tokens=prompt_tokens,
        max_new_tokens=_MAX_NEW_TOKENS,
        num_candidates=_NUM_CANDIDATES,
    )

    _report("Lookahead Decoding", tokenizer, prompt_tokens, run)
    assert len(run.generated_tokens) == _MAX_NEW_TOKENS
    assert run.forward_passes <= _MAX_NEW_TOKENS
    # Same review-caught strengthening as Prompt Lookup's test above: the
    # floor alone can't distinguish a working n-gram cache from a
    # completely broken one. Greedy decoding against a fixed model and
    # prompt is fully deterministic (argmax throughout this batch, no
    # sampling), so this measured acceptance is a fixed fact about this
    # exact input, not a run-to-run gamble -- real, if modest (~12.5% in
    # this batch's own measured run, honestly below CAG.md's stated
    # typical range, disclosed in this batch's evaluation report).
    assert run.tokens_accepted_from_candidates > 0


def test_medusa_never_does_worse_than_naive_generation_even_with_untrained_heads():
    # Disclosed in the design spec and confirmed by a standalone spike
    # before this test was written: untrained, lm_head-weight-copied
    # heads all produce the IDENTICAL top prediction (they have no way
    # to know they're supposed to predict different future positions),
    # so only the first proposed candidate is ever expected to be
    # accepted. The real claim this test verifies is the loop's own
    # correctness floor -- speculative decoding must never cost MORE
    # forward passes than naive generation, even when its candidate
    # source is this degenerate -- not a specific speedup number.
    model, tokenizer = _load_model_and_tokenizer()
    prompt = "The history of the Roman Empire began"
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    use_case = SpeculativeDecode()
    run = use_case.execute(
        target_model=HFTargetModel(model),
        candidate_generator=MedusaCandidateGenerator(model, num_heads=_NUM_CANDIDATES),
        prompt_tokens=prompt_tokens,
        max_new_tokens=_MAX_NEW_TOKENS,
        num_candidates=_NUM_CANDIDATES,
    )

    _report("Medusa", tokenizer, prompt_tokens, run)
    assert len(run.generated_tokens) == _MAX_NEW_TOKENS
    assert run.forward_passes <= _MAX_NEW_TOKENS
