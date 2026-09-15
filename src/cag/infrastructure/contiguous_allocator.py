from src.cag.domain.ports import KVCacheAllocator


class ContiguousKVAllocator(KVCacheAllocator):
    # vAttention's own strategy (CAG.md): no paging at all -- memory is
    # managed dynamically in a flat address space where each sequence
    # occupies one contiguous run. That buys a plainer layout with no
    # block table to consult on every access, and gives up exactly two
    # things paging provides: free space stops being fungible (a request
    # needs a RUN, not a count, so it can fail with plenty of total
    # space left), and a fork has to copy, since there are no blocks to
    # share by reference.
    def __init__(self, total_slots: int) -> None:
        if total_slots < 1:
            raise ValueError("total_slots must be at least 1")
        self._total_slots = total_slots
        # sequence_id -> (start, length), kept sorted by start on read.
        self._regions: dict[str, tuple[int, int]] = {}

    def _sorted_regions(self) -> list[tuple[int, int]]:
        return sorted(self._regions.values())

    def _find_run(self, size: int, ignoring: str | None = None) -> int | None:
        # First-fit over the gaps between occupied regions.
        if size == 0:
            return 0
        occupied = sorted(
            region for key, region in self._regions.items() if key != ignoring
        )
        cursor = 0
        for start, length in occupied:
            if start - cursor >= size:
                return cursor
            cursor = max(cursor, start + length)
        if self._total_slots - cursor >= size:
            return cursor
        return None

    def allocate(self, sequence_id: str, num_tokens: int) -> bool:
        if sequence_id in self._regions:
            raise ValueError(f"sequence {sequence_id!r} is already allocated")
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        start = self._find_run(num_tokens)
        if start is None:
            return False
        self._regions[sequence_id] = (start, num_tokens)
        return True

    def extend(self, sequence_id: str, additional_tokens: int) -> bool:
        if sequence_id not in self._regions:
            raise KeyError(f"unknown sequence {sequence_id!r}")
        if additional_tokens < 0:
            raise ValueError("additional_tokens must be non-negative")
        start, length = self._regions[sequence_id]
        target = length + additional_tokens

        # Grow in place only if the slots immediately after this region
        # are still free; otherwise the whole sequence has to be
        # relocated to a run big enough for the new length -- a real
        # copy, and the cost paging avoids entirely by just appending
        # another block from anywhere.
        neighbours = [
            (other_start, other_length)
            for key, (other_start, other_length) in self._regions.items()
            if key != sequence_id
        ]
        blocked = any(
            other_start < start + target and other_start + other_length > start
            for other_start, other_length in neighbours
        )
        if not blocked and start + target <= self._total_slots:
            self._regions[sequence_id] = (start, target)
            return True

        relocated = self._find_run(target, ignoring=sequence_id)
        if relocated is None:
            return False
        self._regions[sequence_id] = (relocated, target)
        return True

    def fork(self, parent_id: str, child_id: str) -> bool:
        # No block table, so nothing to share -- the child needs its own
        # full copy of the parent's tokens. This is the cost CAG.md
        # points at when it calls block sharing the thing that makes
        # parallel sampling and beam search efficient.
        if parent_id not in self._regions:
            raise KeyError(f"unknown sequence {parent_id!r}")
        if child_id in self._regions:
            raise ValueError(f"sequence {child_id!r} is already allocated")
        _, length = self._regions[parent_id]
        return self.allocate(child_id, length)

    def free(self, sequence_id: str) -> None:
        if sequence_id not in self._regions:
            raise KeyError(f"unknown sequence {sequence_id!r}")
        del self._regions[sequence_id]

    def physical_slots_used(self) -> int:
        return sum(length for _, length in self._regions.values())

    def logical_tokens_held(self) -> int:
        # Every allocated slot holds a real token here: a contiguous run
        # is sized exactly to the sequence, so this allocator has no
        # internal fragmentation at all -- the other half of the trade.
        return self.physical_slots_used()

    def free_runs(self) -> list[int]:
        runs: list[int] = []
        cursor = 0
        for start, length in self._sorted_regions():
            if start > cursor:
                runs.append(start - cursor)
            cursor = max(cursor, start + length)
        if self._total_slots > cursor:
            runs.append(self._total_slots - cursor)
        return runs
