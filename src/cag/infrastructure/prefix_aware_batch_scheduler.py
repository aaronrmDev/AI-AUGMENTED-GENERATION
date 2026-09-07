from src.cag.domain.entities import BatchRequest
from src.cag.domain.ports import BatchScheduler


class PrefixAwareBatchScheduler(BatchScheduler):
    # CAG.md's scheduling claim, implemented: "requests that share a
    # prefix get batched together so that shared prefix is computed once
    # for the whole group instead of once per request."
    #
    # Grouping is done by block-aligned prefix signature rather than by
    # comparing every pair: two requests belong together when their
    # leading whole blocks are identical, which is exactly the
    # granularity a real engine can share cache at. Requests are keyed
    # by as many whole leading blocks as they have, longest key first,
    # so the tightest groupings form before looser ones absorb what is
    # left. Within a group, arrival order is preserved -- grouping
    # changes WHO shares a batch, and deliberately not who waits longest.
    def __init__(self, block_size: int, grouping_blocks: int = 1) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if grouping_blocks < 1:
            raise ValueError("grouping_blocks must be at least 1")
        self._block_size = block_size
        self._grouping_blocks = grouping_blocks

    def _signature(self, request: BatchRequest) -> tuple[int, ...]:
        # As many whole leading blocks as this request can offer, capped
        # at grouping_blocks. A request shorter than one whole block has
        # an empty signature and groups with other such requests, which
        # is correct: it has no shareable prefix to group on.
        usable_blocks = min(len(request.tokens) // self._block_size, self._grouping_blocks)
        return request.tokens[: usable_blocks * self._block_size]

    def form_batches(
        self, pending: list[BatchRequest], max_batch_size: int
    ) -> list[list[BatchRequest]]:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")

        groups: dict[tuple[int, ...], list[BatchRequest]] = {}
        for request in pending:
            groups.setdefault(self._signature(request), []).append(request)

        batches: list[list[BatchRequest]] = []
        # Longest shared signature first: those groups have the most to
        # gain from being kept intact when a group exceeds max_batch_size.
        for signature in sorted(groups, key=len, reverse=True):
            members = groups[signature]
            for start in range(0, len(members), max_batch_size):
                batches.append(members[start : start + max_batch_size])
        return batches
