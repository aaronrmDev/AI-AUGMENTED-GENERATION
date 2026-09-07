from src.cag.domain.entities import OffloadRun, OffloadWorkload
from src.cag.domain.ports import OffloadingStrategy
from src.cag.infrastructure.sequential_offloader import resident_and_offloaded


class KVPROffloader(OffloadingStrategy):
    # KVPR (CAG.md): "overlaps two operations that would otherwise run
    # sequentially -- it partially recomputes KV values on the GPU while
    # simultaneously transferring the remaining values from CPU,
    # synchronizing the two so neither has to wait fully on the other."
    #
    # So the thing hidden here is not transfer behind unrelated compute,
    # as in LayerKV and InfiniGen, but transfer behind RECOMPUTE of the
    # part that was not transferred. A layer's stall is therefore the
    # longer of the two halves rather than their sum. recompute_fraction
    # is what share of the layer the GPU rebuilds itself; recompute is
    # charged at compute_ms_per_layer, since rebuilding KV is real work
    # on the same device.
    def __init__(self, recompute_fraction: float) -> None:
        if not (0.0 <= recompute_fraction <= 1.0):
            raise ValueError("recompute_fraction must be between 0.0 and 1.0")
        self._recompute_fraction = recompute_fraction

    def run(self, workload: OffloadWorkload) -> OffloadRun:
        resident, offloaded = resident_and_offloaded(workload)
        recompute_ms = workload.compute_ms_per_layer * self._recompute_fraction
        transfer_ms = workload.warm_fetch_ms_per_layer * (1.0 - self._recompute_fraction)
        stall_per_layer = max(recompute_ms, transfer_ms)
        hidden_per_layer = min(recompute_ms, transfer_ms)
        return OffloadRun(
            strategy="kvpr",
            total_ms=(workload.num_layers * workload.compute_ms_per_layer)
            + (offloaded * stall_per_layer),
            transfer_ms_issued=offloaded * transfer_ms,
            transfer_ms_hidden=offloaded * hidden_per_layer,
            gpu_layers_resident=resident,
        )
