from src.cag.domain.entities import OffloadRun, OffloadWorkload
from src.cag.domain.ports import OffloadingStrategy
from src.cag.infrastructure.sequential_offloader import resident_and_offloaded


def pipeline_total(
    num_layers: int, compute_ms: float, fetch_ms_by_layer: list[float]
) -> tuple[float, float]:
    # The overlap model every prefetching strategy here shares: while
    # layer i computes, layer i+1's fetch is already in flight, so the
    # pipeline only stalls for however much of that fetch outlasts the
    # compute. Returns (total_ms, transfer_ms_hidden).
    #
    # The first layer's fetch has no compute to hide behind -- nothing
    # has started yet -- which is why it is charged in full. Modelling it
    # as free would quietly invent an overlap no real pipeline gets.
    total = fetch_ms_by_layer[0] if fetch_ms_by_layer else 0.0
    hidden = 0.0
    for layer in range(num_layers):
        next_fetch = fetch_ms_by_layer[layer + 1] if layer + 1 < len(fetch_ms_by_layer) else 0.0
        total += max(compute_ms, next_fetch)
        hidden += min(compute_ms, next_fetch)
    return total, hidden


class LayerKVOffloader(OffloadingStrategy):
    # LayerKV (CAG.md): offloads at the granularity of whole layers
    # rather than individual tokens -- a subset of layers stays resident
    # on GPU and the rest live on CPU, "overlapping the transfer of
    # offloaded layers with computation on the layers that stayed."
    def run(self, workload: OffloadWorkload) -> OffloadRun:
        resident, _ = resident_and_offloaded(workload)
        fetches = [
            0.0 if layer < resident else workload.warm_fetch_ms_per_layer
            for layer in range(workload.num_layers)
        ]
        total, hidden = pipeline_total(
            workload.num_layers, workload.compute_ms_per_layer, fetches
        )
        return OffloadRun(
            strategy="layerkv",
            total_ms=total,
            transfer_ms_issued=sum(fetches),
            transfer_ms_hidden=hidden,
            gpu_layers_resident=resident,
        )
