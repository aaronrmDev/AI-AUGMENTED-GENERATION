def amortized_tokens_per_turn(tokens_recomputed: int, num_turns: int) -> float:
    # CAG.md's headline claim for this technique is about the AMORTIZED
    # per-turn cost -- "near-O(1) amortized cost per turn instead of cost
    # that grows with every message" -- so the per-turn average is the
    # number that claim is actually about, not the session total.
    if num_turns < 1:
        raise ValueError("num_turns must be at least 1")
    if tokens_recomputed < 0:
        raise ValueError("tokens_recomputed must be non-negative")
    return tokens_recomputed / num_turns


def recompute_ratio(strategy_tokens: int, naive_tokens: int) -> float:
    # How much of the naive "reprint the whole book every chapter" cost a
    # strategy actually pays. 1.0 means it saved nothing.
    if naive_tokens <= 0:
        raise ValueError("naive_tokens must be positive")
    if strategy_tokens < 0:
        raise ValueError("strategy_tokens must be non-negative")
    return strategy_tokens / naive_tokens


def is_flat_per_turn(per_turn_recompute: list[int], tolerance: float = 0.0) -> bool:
    # The difference between "cheaper" and "O(1) per turn" is whether the
    # per-turn cost STOPS GROWING, not whether the total is smaller --
    # a strategy can halve a quadratic and still be quadratic. This asks
    # the structural question directly: does the last turn cost more than
    # the first, beyond the allowed tolerance.
    if not per_turn_recompute:
        raise ValueError("per_turn_recompute must be non-empty")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    first = per_turn_recompute[0]
    if first == 0:
        return all(cost == 0 for cost in per_turn_recompute)
    return max(per_turn_recompute) <= first * (1.0 + tolerance)
