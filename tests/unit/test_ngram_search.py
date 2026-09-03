from src.cag.domain.ngram_search import find_ngram_continuation


def test_finds_the_continuation_after_an_exact_match():
    haystack = [1, 2, 3, 4, 5, 6]
    tail = [3, 4]
    assert find_ngram_continuation(tail, haystack, num_candidates=2) == [5, 6]


def test_prefers_the_most_recent_match_when_several_exist():
    haystack = [3, 4, 99, 3, 4, 100, 101]
    tail = [3, 4]
    assert find_ngram_continuation(tail, haystack, num_candidates=2) == [100, 101]


def test_truncates_the_continuation_at_the_end_of_the_haystack():
    haystack = [1, 2, 3, 4]
    tail = [3, 4]
    assert find_ngram_continuation(tail, haystack, num_candidates=5) == []


def test_falls_back_to_a_shorter_ngram_when_the_longest_has_no_match():
    # tail's own most recent 2 tokens ([9, 4]) never occur together in the
    # haystack, but the single-token tail [4] does, at index 2 -- falling
    # back to a 1-gram match still finds a usable continuation instead of
    # giving up outright.
    haystack = [7, 8, 4, 10, 11]
    tail = [9, 4]
    result = find_ngram_continuation(
        tail, haystack, num_candidates=2, max_ngram_size=2, min_ngram_size=1
    )
    assert result == [10, 11]


def test_returns_empty_when_no_ngram_size_matches_anything():
    haystack = [1, 2, 3]
    tail = [99, 98]
    assert find_ngram_continuation(tail, haystack, num_candidates=2) == []


def test_returns_empty_for_an_empty_tail_or_empty_haystack():
    assert find_ngram_continuation([], [1, 2, 3], num_candidates=2) == []
    assert find_ngram_continuation([1, 2], [], num_candidates=2) == []


def test_max_start_excludes_a_trivial_self_match_without_losing_continuation_content():
    # tail [9, 8] is ALSO haystack's own literal trailing 2-gram (at
    # index 4-5) -- without max_start, the search would trivially
    # "match" the tail against itself first and return [] (nothing
    # follows the very end). max_start excludes that one trivial start
    # position (search_ceiling caps at len(haystack)-2, i.e. start=4 is
    # never tried) while still finding the genuine EARLIER occurrence of
    # [9, 8] at index 0, and the continuation it returns can still reach
    # haystack's own last two tokens (3, 4) despite them sitting past
    # the excluded trivial-match position.
    haystack = [9, 8, 3, 4, 9, 8]
    tail = [9, 8]
    result = find_ngram_continuation(
        tail, haystack, num_candidates=3, max_ngram_size=2, max_start=len(haystack) - 2
    )
    assert result == [3, 4, 9]


def test_max_ngram_size_caps_how_much_of_the_tail_is_used_as_the_key():
    # tail's last 3 tokens [1, 2, 3] never occur together, but the last 2
    # ([2, 3]) do -- max_ngram_size=2 should never even try the 3-gram,
    # going straight to the 2-gram match.
    haystack = [9, 2, 3, 42]
    tail = [1, 2, 3]
    result = find_ngram_continuation(
        tail, haystack, num_candidates=1, max_ngram_size=2, min_ngram_size=1
    )
    assert result == [42]
