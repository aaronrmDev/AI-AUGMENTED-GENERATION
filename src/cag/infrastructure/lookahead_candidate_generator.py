from src.cag.domain.ngram_search import find_ngram_continuation
from src.cag.domain.ports import CandidateGenerator


class LookaheadCandidateGenerator(CandidateGenerator):
    # CAG.md: "generates its candidates from an n-gram cache built out
    # of the prior context, rather than from a trained draft model or
    # extra heads at all." The prompt is deliberately never consulted --
    # that's Prompt Lookup's own distinguishing behavior, not this one's.
    # Searches generated_tokens for an EARLIER occurrence of its own
    # trailing n-gram, always excluding the trailing position itself
    # (unlike Prompt Lookup, self-match risk is unconditional here,
    # since the tail is always drawn from the same buffer being
    # searched).
    #
    # Disclosed simplification (per the design spec): the original
    # Lookahead Decoding paper's full mechanism is a parallel Jacobi-
    # iteration decoding scheme speculating multiple n-gram trajectories
    # at once. This reproduces the variant's distinguishing candidate
    # SOURCE (an n-gram cache from prior context, not the prompt and not
    # a trained head) through the same shared propose-verify-accept loop
    # every variant in this batch uses, not the source paper's own
    # parallel-trajectory algorithm.
    def __init__(self, max_ngram_size: int = 3, min_ngram_size: int = 1) -> None:
        self._max_ngram_size = max_ngram_size
        self._min_ngram_size = min_ngram_size

    def propose(
        self, prompt_tokens: list[int], generated_tokens: list[int], num_candidates: int
    ) -> list[int]:
        tail = generated_tokens[-self._max_ngram_size :]
        return find_ngram_continuation(
            tail,
            generated_tokens,
            num_candidates,
            max_ngram_size=self._max_ngram_size,
            min_ngram_size=self._min_ngram_size,
            max_start=len(generated_tokens) - 1,
        )
