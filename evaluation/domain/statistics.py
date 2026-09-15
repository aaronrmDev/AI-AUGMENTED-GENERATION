def nearest_rank_percentile(sorted_values: list[float], pct: float) -> float:
    # The same nearest-rank formula RunComparison uses, so latency figures
    # across this project's reports mean the same thing.
    if not sorted_values:
        raise ValueError("at least one value is required")
    index = min(len(sorted_values) - 1, round(pct * (len(sorted_values) - 1)))
    return sorted_values[index]
