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
        # Self-match guard, applied unconditionally -- review-caught: an
        # earlier version only excluded the prompt's own trailing
        # position when generated_tokens was completely empty, reasoning
        # that once generation starts the tail always includes at least
        # one non-prompt token. That's true of the LONGEST n-gram tried,
        # but find_ngram_continuation falls back to shorter n-grams when
        # the longest finds nothing, and a short fallback key can still
        # coincidentally equal the prompt's own trailing token(s) even
        # with a non-empty generated_tokens (e.g. the model's first
        # generated token happens to repeat the prompt's last one --
        # exactly the kind of echo Prompt Lookup is meant to catch).
        # Because find_ngram_continuation returns on the first (most
        # recent) match, that coincidental trailing self-match would
        # shadow a genuine, earlier, non-empty match -- confirmed
        # empirically: propose([5,6,7,5,6,7], [7], num_candidates=3)
        # returned [] instead of the real earlier match's [5, 6, 7].
        # Always excluding the prompt's own last valid start position
        # costs at most one rare, legitimate match ending exactly there;
        # it closes the bug for every fallback n-gram size, not just the
        # longest.
        max_start = len(prompt_tokens) - 1
        return find_ngram_continuation(
            tail,
            prompt_tokens,
            num_candidates,
            max_ngram_size=self._max_ngram_size,
            min_ngram_size=self._min_ngram_size,
            max_start=max_start,
        )
