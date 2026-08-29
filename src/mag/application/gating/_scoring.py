import math


def safe_score(score: float) -> float:
    # NaN fails every comparison (nan < x and x < nan are both False),
    # which violates the strict weak ordering sorted() requires -- once one
    # NaN score reaches a sort key, Timsort doesn't just misplace that one
    # candidate, it can silently scramble the order of OTHER, well-defined
    # scores around it too. Substituting -inf treats "unknown/corrupted
    # relevance" as "assume irrelevant": it sorts last under every sort
    # shape this package uses, whether descending on the raw score
    # (reverse=True) or ascending on its negation (HierarchicalAssembly),
    # since -(-inf) is +inf either way.
    return score if not math.isnan(score) else float("-inf")
