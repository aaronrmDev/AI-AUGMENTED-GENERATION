from src.cag.domain.ngram_search import find_ngram_continuation
from src.cag.domain.ports import CandidateGenerator


class PromptLookupCandidateGenerator(CandidateGenerator):
    # CAG.md: "reuses tokens straight from the input prompt as its
    # candidates, which works well specifically when the expected
    # output is likely to repeat or closely echo material that's
    # already sitting in the prompt." The search key is the tail of the
    # FULL context (prompt + everything generated so far), searched
    # against the prompt alone.
    def __init__(self, max_ngram_size: int = 3, min_ngram_size: int = 1) -> None:
        self._max_ngram_size = max_ngram_size
        self._min_ngram_size = min_ngram_size

    def propose(
        self, prompt_tokens: list[int], generated_tokens: list[int], num_candidates: int
    ) -> list[int]:
        context = prompt_tokens + generated_tokens
        if not context:
            return []
        tail = context[-self._max_ngram_size :]
        # Self-match guard: only relevant when nothing has been
        # generated yet, since the tail then comes purely from the
        # prompt's own end -- searching the prompt for its own literal
        # tail would trivially "find" the tail itself with nothing after
        # it to propose. Once generation has started, the tail always
        # includes at least one non-prompt token, so no exclusion is
        # needed -- a real coincidental match at the prompt's own end is
        # legitimate content, not a structural self-match.
        max_start = len(prompt_tokens) - 1 if not generated_tokens else None
        return find_ngram_continuation(
            tail,
            prompt_tokens,
            num_candidates,
            max_ngram_size=self._max_ngram_size,
            min_ngram_size=self._min_ngram_size,
            max_start=max_start,
        )
