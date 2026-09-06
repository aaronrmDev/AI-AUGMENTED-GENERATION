from src.cag.domain.ports import CacheDistiller


class InfiniPotDistiller(CacheDistiller):
    # InfiniPot's own departure from every other algorithm in this batch
    # (CAG.md): not a per-token keep/evict decision at all -- once the
    # cache overflows its budget, it's distilled, "closer to selective
    # compression of the whole cache than to per-token selection." The
    # source's own CaP/NuC metrics aren't reimplemented bit-for-bit here
    # (no published formula to match against, the same disclosed-choice
    # position ShadowKV and MiniCache already took in the Compression
    # batch); what's built is a real, honest instance of the same
    # structural idea -- partition the overflowing cache into `budget`
    # contiguous groups and replace each with its centroid, so every kept
    # row is a genuine blend of several original tokens rather than a
    # survivor of a per-token cut.
    def distill(self, kv: list[list[float]], budget: int) -> list[list[float]]:
        if not kv:
            raise ValueError("kv must be non-empty")
        if budget < 1:
            raise ValueError("budget must be at least 1")
        seq_len = len(kv)
        dim = len(kv[0])

        if budget >= seq_len:
            return [list(row) for row in kv]

        # Contiguous, near-even partition into `budget` groups -- earlier
        # groups may carry one extra row when seq_len doesn't divide
        # evenly, rather than dropping the remainder.
        base_size, remainder = divmod(seq_len, budget)
        distilled: list[list[float]] = []
        start = 0
        for group_index in range(budget):
            group_size = base_size + (1 if group_index < remainder else 0)
            group = kv[start : start + group_size]
            centroid = [sum(row[c] for row in group) / len(group) for c in range(dim)]
            distilled.append(centroid)
            start += group_size
        return distilled
