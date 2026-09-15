def accumulate_attention_scores(attention_matrix: list[list[float]]) -> list[float]:
    # H2O's own "accumulated attention score across the whole generation
    # so far" is, over one real causal forward pass, exactly the column
    # sum of a lower-triangular attention matrix: row i is query position
    # i's attention distribution over key positions 0..i, so key position
    # j's accumulated score is the sum, over every query row i >= j, of
    # that row's weight at column j. This reads real accumulated mass off
    # one teacher-forced forward pass instead of a token-by-token decode
    # loop -- the same category of honest simplification the project's
    # speculative-decoding batch already uses (one real forward pass
    # verifies candidates, rather than simulating true autoregressive
    # generation step by step).
    if not attention_matrix:
        raise ValueError("attention_matrix must be non-empty")
    seq_len = len(attention_matrix)
    scores = [0.0] * seq_len
    for query_row in attention_matrix:
        if len(query_row) != seq_len:
            raise ValueError("attention_matrix must be square (seq_len x seq_len)")
        for key_index, weight in enumerate(query_row):
            scores[key_index] += weight
    return scores


def memory_reduction_ratio(original_token_count: int, kept_token_count: int) -> float:
    if kept_token_count <= 0:
        raise ValueError("kept_token_count must be a positive number of tokens")
    if original_token_count < kept_token_count:
        raise ValueError("original_token_count must be at least kept_token_count")
    return original_token_count / kept_token_count


def retained_attention_mass(scores: list[float], keep_indices: list[int]) -> float:
    # The real accuracy-risk proxy for eviction, the analogue of
    # compression's reconstruction_error: what fraction of the total
    # accumulated attention mass survives in the kept set. 1.0 means
    # nothing that ever mattered was dropped; a value near
    # len(keep_indices) / len(scores) means eviction did no better than
    # picking indices at random.
    if not scores:
        raise ValueError("scores must be non-empty")
    total = sum(scores)
    if total <= 0:
        raise ValueError("scores must sum to a positive total")
    return sum(scores[i] for i in keep_indices) / total
