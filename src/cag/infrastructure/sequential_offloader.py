from src.cag.domain.entities import OffloadRun, OffloadWorkload
from src.cag.domain.ports import OffloadingStrategy


def resident_and_offloaded(workload: OffloadWorkload) -> tuple[int, int]:
    # Shared by every strategy below: how many layers fit on the GPU and
    # how many have to live somewhere slower.
    resident = min(workload.gpu_layer_capacity, workload.num_layers)
    return resident, workload.num_layers - resident


class SequentialOffloader(OffloadingStrategy):
    # The baseline the other five are measured against, and the shape
    # CAG.md implicitly argues past: tiering with no attempt to overlap
    # anything. Every non-resident layer is fetched, the pipeline waits
    # for it, and only then computes. Nothing is hidden, so this run's
    # overlap efficiency is 0.0 by construction and its total is the
    # honest worst case tiering can produce.
    def run(self, workload: OffloadWorkload) -> OffloadRun:
        resident, offloaded = resident_and_offloaded(workload)
        compute = workload.num_layers * workload.compute_ms_per_layer
        transfer = offloaded * workload.warm_fetch_ms_per_layer
        return OffloadRun(
            strategy="sequential",
            total_ms=compute + transfer,
            transfer_ms_issued=transfer,
            transfer_ms_hidden=0.0,
            gpu_layers_resident=resident,
        )
