from src.cag.domain.entities import OffloadRun, OffloadWorkload
from src.cag.domain.ports import OffloadingStrategy
from src.cag.infrastructure.layerkv_offloader import pipeline_total
from src.cag.infrastructure.sequential_offloader import resident_and_offloaded


class InfiniGenOffloader(OffloadingStrategy):
    # InfiniGen (CAG.md): CPU storage paired with predictive prefetching
    # "specifically at the layer level -- it predicts which KV entries
    # will be critical for the NEXT layer's computation and prefetches
    # them while the GPU is still busy processing the CURRENT layer."
    #
    # Two mechanisms, not one, and they compound. Like LayerKV it
    # overlaps; unlike LayerKV it moves only the entries it predicts are
    # critical, so each transfer is smaller before any overlap applies.
    # critical_fraction is the share of a layer's entries the predictor
    # keeps -- a parameter of this model, since CAG.md names no figure.
    def __init__(self, critical_fraction: float) -> None:
        if not (0.0 < critical_fraction <= 1.0):
            raise ValueError("critical_fraction must be in (0.0, 1.0]")
        self._critical_fraction = critical_fraction

    def run(self, workload: OffloadWorkload) -> OffloadRun:
        resident, _ = resident_and_offloaded(workload)
        per_layer = workload.warm_fetch_ms_per_layer * self._critical_fraction
        fetches = [
            0.0 if layer < resident else per_layer for layer in range(workload.num_layers)
        ]
        total, hidden = pipeline_total(
            workload.num_layers, workload.compute_ms_per_layer, fetches
        )
        return OffloadRun(
            strategy="infinigen",
            total_ms=total,
            transfer_ms_issued=sum(fetches),
            transfer_ms_hidden=hidden,
            gpu_layers_resident=resident,
        )
