import math

from src.cag.domain.ports import KVCacheAllocator


class PagedKVAllocator(KVCacheAllocator):
    # PagedAttention's own strategy (CAG.md): the pool is divided into
    # fixed-size blocks, each sequence keeps a block table mapping its
    # logical token positions to physical blocks, and those blocks may
    # live anywhere -- contiguity is never required. Forking shares
    # blocks by reference count rather than copying, and a shared block
    # is copied only when a branch actually writes into it
    # (copy-on-write).
    def __init__(self, total_blocks: int, block_size: int) -> None:
        if total_blocks < 1:
            raise ValueError("total_blocks must be at least 1")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        self._block_size = block_size
        self._free_blocks: list[int] = list(range(total_blocks))
        self._block_tables: dict[str, list[int]] = {}
        self._token_counts: dict[str, int] = {}
        self._refcounts: dict[int, int] = {}

    def _blocks_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self._block_size)

    def _take(self, count: int) -> list[int] | None:
        # Any free block will do -- this is exactly why external
        # fragmentation cannot arise here: the allocator never needs a
        # RUN of adjacent blocks, only a count of them.
        if count > len(self._free_blocks):
            return None
        taken = [self._free_blocks.pop() for _ in range(count)]
        for block in taken:
            self._refcounts[block] = 1
        return taken

    def allocate(self, sequence_id: str, num_tokens: int) -> bool:
        if sequence_id in self._block_tables:
            raise ValueError(f"sequence {sequence_id!r} is already allocated")
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        taken = self._take(self._blocks_needed(num_tokens))
        if taken is None:
            return False
        self._block_tables[sequence_id] = taken
        self._token_counts[sequence_id] = num_tokens
        return True

    def extend(self, sequence_id: str, additional_tokens: int) -> bool:
        if sequence_id not in self._block_tables:
            raise KeyError(f"unknown sequence {sequence_id!r}")
        if additional_tokens < 0:
            raise ValueError("additional_tokens must be non-negative")
        table = self._block_tables[sequence_id]
        current = self._token_counts[sequence_id]

        # Copy-on-write: growing writes into this sequence's last block,
        # so if that block is still shared with a fork sibling it has to
        # become private first -- the one moment paging actually pays
        # for the sharing it got for free at fork time.
        if table and self._refcounts[table[-1]] > 1 and current % self._block_size != 0:
            replacement = self._take(1)
            if replacement is None:
                return False
            self._refcounts[table[-1]] -= 1
            table[-1] = replacement[0]

        target = current + additional_tokens
        extra_blocks = self._blocks_needed(target) - len(table)
        if extra_blocks > 0:
            taken = self._take(extra_blocks)
            if taken is None:
                return False
            table.extend(taken)
        self._token_counts[sequence_id] = target
        return True

    def fork(self, parent_id: str, child_id: str) -> bool:
        # The claim CAG.md makes about parallel sampling and beam search:
        # a fork costs no new blocks at all, only reference counts, so
        # N branches of one prompt hold one physical copy of it.
        if parent_id not in self._block_tables:
            raise KeyError(f"unknown sequence {parent_id!r}")
        if child_id in self._block_tables:
            raise ValueError(f"sequence {child_id!r} is already allocated")
        table = list(self._block_tables[parent_id])
        for block in table:
            self._refcounts[block] += 1
        self._block_tables[child_id] = table
        self._token_counts[child_id] = self._token_counts[parent_id]
        return True

    def free(self, sequence_id: str) -> None:
        if sequence_id not in self._block_tables:
            raise KeyError(f"unknown sequence {sequence_id!r}")
        for block in self._block_tables.pop(sequence_id):
            self._refcounts[block] -= 1
            if self._refcounts[block] == 0:
                del self._refcounts[block]
                self._free_blocks.append(block)
        del self._token_counts[sequence_id]

    def physical_slots_used(self) -> int:
        # Distinct blocks actually held, counted once no matter how many
        # sequences share them -- the whole point of the block table.
        return len(self._refcounts) * self._block_size

    def logical_tokens_held(self) -> int:
        return sum(self._token_counts.values())

    def free_runs(self) -> list[int]:
        # Every free block is independently usable, so free space is
        # never "scattered" in any sense that can fail an allocation --
        # reported as one run, which is what makes external_fragmentation
        # measure 0.0 for this allocator by construction rather than by
        # luck of a particular workload.
        return [len(self._free_blocks) * self._block_size]
