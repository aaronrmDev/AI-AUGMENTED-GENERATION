def internal_fragmentation(physical_slots_used: int, logical_tokens_held: int) -> float:
    # The waste PagedAttention pays for its own design: a sequence's last
    # block is almost never exactly full, so some slots inside allocated
    # blocks hold nothing. CAG.md's claim that paging means "no
    # fragmentation" is true only of the EXTERNAL kind measured below --
    # paging trades external fragmentation for this, and reporting only
    # the half that flatters the technique would be exactly the kind of
    # selective measurement this project's reports refuse elsewhere.
    if physical_slots_used < 0 or logical_tokens_held < 0:
        raise ValueError("slot counts must be non-negative")
    if logical_tokens_held > physical_slots_used:
        raise ValueError("logical tokens held cannot exceed physical slots used")
    if physical_slots_used == 0:
        return 0.0
    return (physical_slots_used - logical_tokens_held) / physical_slots_used


def external_fragmentation(free_runs: list[int]) -> float:
    # The classic measure, and the one PagedAttention genuinely does
    # eliminate: given the sizes of every separate run of free space,
    # how much of the total free space is unusable for a single
    # largest-possible request. 0.0 means all free space is in one
    # usable run; approaching 1.0 means free space exists but is
    # scattered into slivers too small to serve anything.
    if any(run < 0 for run in free_runs):
        raise ValueError("free run sizes must be non-negative")
    total_free = sum(free_runs)
    if total_free == 0:
        return 0.0
    return 1.0 - (max(free_runs) / total_free)
