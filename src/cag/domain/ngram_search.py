def find_ngram_continuation(
    tail: list[int],
    haystack: list[int],
    num_candidates: int,
    max_ngram_size: int = 3,
    min_ngram_size: int = 1,
    max_start: int | None = None,
) -> list[int]:
    # Shared by Prompt Lookup (haystack = the prompt) and Lookahead
    # Decoding (haystack = everything generated so far) -- both are
    # "search this buffer for the current tail, propose whatever
    # followed it," differing only in which buffer. Tries the longest
    # ngram first, falling back to shorter ones only if nothing matched
    # -- a longer match is a stronger, more specific signal than a
    # shorter one.
    #
    # max_start caps which match START positions are considered, without
    # shrinking haystack itself -- callers whose tail could coincide with
    # haystack's OWN trailing end pass len(haystack) - 1, so the position
    # that would trivially "match" the tail against itself is excluded
    # from matching, while every token of haystack -- including its very
    # last one -- stays available as CONTINUATION content for a genuine,
    # non-trivial earlier match. Both Lookahead and Prompt Lookup apply
    # this UNCONDITIONALLY (not just when generated_tokens is empty): a
    # review finding caught that find_ngram_continuation's own fallback
    # to shorter ngrams can produce a search key that coincidentally
    # equals haystack's own trailing token(s) even when the tail's
    # LONGEST form includes non-haystack content, so a conditional
    # exclusion isn't enough -- see PromptLookupCandidateGenerator's own
    # comment for the concrete failure case this caused.
    if not tail or not haystack:
        return []
    largest = min(max_ngram_size, len(tail))
    search_ceiling = len(haystack) if max_start is None else min(max_start, len(haystack))
    for ngram_size in range(largest, min_ngram_size - 1, -1):
        ngram = tail[-ngram_size:]
        for start in range(search_ceiling - ngram_size, -1, -1):
            if haystack[start : start + ngram_size] == ngram:
                return haystack[start + ngram_size : start + ngram_size + num_candidates]
    return []
