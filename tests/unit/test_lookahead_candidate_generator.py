from src.cag.infrastructure.lookahead_candidate_generator import LookaheadCandidateGenerator


def test_proposes_tokens_that_followed_an_earlier_occurrence_in_generated_text():
    # CAG.md: "generates its candidates from an n-gram cache built out
    # of the prior context" -- prior GENERATED context, not the prompt
    # (that's Prompt Lookup's job). The pattern [5, 6] repeated earlier
    # in generated_tokens, followed by [7, 8] -- that's what should be
    # proposed for the current, second occurrence of [5, 6].
    generator = LookaheadCandidateGenerator(max_ngram_size=2)
    generated_tokens = [1, 5, 6, 7, 8, 2, 5, 6]

    candidates = generator.propose(
        prompt_tokens=[99], generated_tokens=generated_tokens, num_candidates=2
    )

    assert candidates == [7, 8]


def test_never_matches_the_tails_own_trailing_position_against_itself():
    # The tail IS generated_tokens' own last 2 tokens -- without
    # excluding the trivial self-match, this would "match" against
    # itself and propose nothing (there's nothing after the very end).
    # With no OTHER earlier occurrence anywhere in generated_tokens,
    # this must return empty, not silently match itself.
    generator = LookaheadCandidateGenerator(max_ngram_size=2)
    generated_tokens = [1, 2, 3, 9, 8]

    candidates = generator.propose(
        prompt_tokens=[], generated_tokens=generated_tokens, num_candidates=2
    )

    assert candidates == []


def test_ignores_the_prompt_entirely_even_when_it_would_have_matched():
    # A tail that DOES occur in the prompt but never (other than
    # trivially) in generated_tokens must still return empty --
    # Lookahead's whole distinguishing property vs. Prompt Lookup is
    # that it never looks at the prompt at all.
    generator = LookaheadCandidateGenerator(max_ngram_size=2)

    candidates = generator.propose(
        prompt_tokens=[9, 8, 100, 101], generated_tokens=[1, 2, 9, 8], num_candidates=2
    )

    assert candidates == []


def test_returns_empty_before_enough_tokens_have_been_generated():
    generator = LookaheadCandidateGenerator(max_ngram_size=3)
    assert generator.propose(prompt_tokens=[1, 2, 3], generated_tokens=[], num_candidates=2) == []
