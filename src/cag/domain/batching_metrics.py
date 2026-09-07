def longest_common_block_prefix(
    sequences: list[tuple[int, ...]], block_size: int
) -> int:
    # How many leading tokens every sequence in a batch agrees on,
    # rounded DOWN to a whole block. Rounding down is the honest part:
    # a serving engine shares cache at block granularity, so a partial
    # block of agreement past the last whole one buys nothing and must
    # not be counted as saved.
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if not sequences:
        return 0
    shortest = min(len(sequence) for sequence in sequences)
    matched = 0
    while matched < shortest and len({sequence[matched] for sequence in sequences}) == 1:
        matched += 1
    return (matched // block_size) * block_size


def prefill_tokens_required(
    batches: list[list[tuple[int, ...]]], block_size: int
) -> int:
    # What a prefix-sharing engine actually has to compute for a given
    # grouping: each batch pays for its shared block-aligned prefix once,
    # then for whatever each member has beyond it. Grouping requests that
    # share a prefix is exactly what shrinks this number, which is the
    # claim CAG.md makes for the scheduling stage.
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    total = 0
    for batch in batches:
        if not batch:
            continue
        shared = longest_common_block_prefix(batch, block_size)
        total += shared
        total += sum(len(sequence) - shared for sequence in batch)
    return total


def unbatched_prefill_tokens(sequences: list[tuple[int, ...]]) -> int:
    # The baseline every grouping is measured against: no sharing at
    # all, every request prefilled independently.
    return sum(len(sequence) for sequence in sequences)
