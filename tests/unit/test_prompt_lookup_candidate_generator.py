from src.cag.infrastructure.prompt_lookup_candidate_generator import (
    PromptLookupCandidateGenerator,
)


def test_proposes_the_tokens_that_followed_the_matching_prompt_occurrence():
    # CAG.md: "reuses tokens straight from the input prompt as its
    # candidates, which works well specifically when the expected output
    # is likely to repeat or closely echo material that's already
    # sitting in the prompt."
    prompt_tokens = [1, 2, 3, 4, 5, 6, 7]
    generated_tokens = [10, 3, 4]  # tail [3, 4] echoes prompt_tokens[2:4]
    generator = PromptLookupCandidateGenerator(max_ngram_size=2)

    candidates = generator.propose(prompt_tokens, generated_tokens, num_candidates=3)

    assert candidates == [5, 6, 7]


def test_returns_empty_when_the_tail_never_occurred_in_the_prompt():
    prompt_tokens = [1, 2, 3]
    generated_tokens = [99, 98]
    generator = PromptLookupCandidateGenerator(max_ngram_size=2)

    assert generator.propose(prompt_tokens, generated_tokens, num_candidates=2) == []


def test_a_short_generated_tail_coinciding_with_the_prompts_own_end_still_finds_the_real_match():
    # Review-caught (MEDIUM): a SHORT (but non-empty) generated_tokens
    # can still make find_ngram_continuation's fallback n-gram key
    # coincidentally equal the prompt's own literal trailing token(s) --
    # generation having "started" doesn't guarantee the tail is free of
    # that overlap once the search falls back to a shorter n-gram. The
    # prompt's own trailing 7 here is a genuine echo of an earlier 7 at
    # index 2, whose real continuation is [5, 6, 7] -- an unguarded
    # search would trivially "match" the tail against the prompt's own
    # last token (index 5, nothing after it) and return [] instead,
    # confirmed empirically before this fix.
    prompt_tokens = [5, 6, 7, 5, 6, 7]
    generated_tokens = [7]
    generator = PromptLookupCandidateGenerator(max_ngram_size=3)

    candidates = generator.propose(prompt_tokens, generated_tokens, num_candidates=3)

    assert candidates == [5, 6, 7]


def test_falls_back_to_the_prompt_itself_before_any_generation_has_happened():
    # generated_tokens is empty on the very first call -- the search key
    # comes from the tail of the prompt itself in that case.
    prompt_tokens = [1, 2, 3, 4, 2, 3, 9, 8]
    generator = PromptLookupCandidateGenerator(max_ngram_size=2)

    candidates = generator.propose(prompt_tokens, [], num_candidates=2)

    # tail of (prompt+generated) = tail of prompt = [4, 2, 3, 9, 8]'s last
    # 2 tokens [9, 8] -- no earlier occurrence of [9, 8] in the prompt, so
    # this falls back to the 1-gram [8], which also never recurs earlier
    # -- correctly empty, not a spurious self-match on the tail itself.
    assert candidates == []
