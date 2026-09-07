from src.cag.domain.entities import OffloadRun, OffloadWorkload
from src.cag.domain.ports import OffloadingStrategy
from src.cag.infrastructure.layerkv_offloader import pipeline_total
from src.cag.infrastructure.sequential_offloader import resident_and_offloaded


class INF2Offloader(OffloadingStrategy):
    # INF2 (CAG.md): "pushes computation itself toward storage rather
    # than just moving data across the bus: it offloads attention
    # computation to accelerators that live on the SSDs themselves --
    # what the source calls CSDs -- which reduces how much data has to
    # cross PCIe in the first place, rather than merely compressing what
    # does cross it."
    #
    # So the lever is neither overlap nor prediction: it is that only
    # the RESULT crosses the bus instead of the raw KV, shrinking the
    # transfer by pcie_reduction. In exchange the in-storage accelerator
    # is slower than the GPU would have been, charged per offloaded
    # layer as storage_compute_ms -- without which this would model a
    # free lunch rather than a trade.
    def __init__(self, pcie_reduction: float, storage_compute_ms: float) -> None:
        if not (0.0 < pcie_reduction <= 1.0):
            raise ValueError("pcie_reduction must be in (0.0, 1.0]")
        if storage_compute_ms < 0:
            raise ValueError("storage_compute_ms must be non-negative")
        self._pcie_reduction = pcie_reduction
        self._storage_compute_ms = storage_compute_ms

    def run(self, workload: OffloadWorkload) -> OffloadRun:
        resident, _ = resident_and_offloaded(workload)
        per_layer = (
            workload.cold_fetch_ms_per_layer * self._pcie_reduction
        ) + self._storage_compute_ms
        fetches = [
            0.0 if layer < resident else per_layer for layer in range(workload.num_layers)
        ]
        total, hidden = pipeline_total(
            workload.num_layers, workload.compute_ms_per_layer, fetches
        )
        return OffloadRun(
            strategy="inf2",
            total_ms=total,
            transfer_ms_issued=sum(fetches),
            transfer_ms_hidden=hidden,
            gpu_layers_resident=resident,
        )
