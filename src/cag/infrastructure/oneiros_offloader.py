from src.cag.domain.entities import OffloadRun, OffloadWorkload
from src.cag.domain.ports import OffloadingStrategy
from src.cag.infrastructure.layerkv_offloader import pipeline_total


class OneirosOffloader(OffloadingStrategy):
    # Oneiros (CAG.md): "takes a different resource off the GPU entirely
    # -- instead of moving KV cache data around, it remaps the model's
    # own parameters off GPU memory, freeing up room for the KV cache
    # itself to expand."
    #
    # Structurally unlike every other strategy here: it does not offload
    # the cache at all, so its effect appears as a LARGER resident
    # capacity rather than as cleverer movement of what did not fit.
    # freed_layer_capacity is how many additional layers' worth of KV
    # the reclaimed parameter space holds. The cost it does pay -- the
    # parameters themselves now streaming from elsewhere -- is charged
    # per layer as parameter_stream_ms, because pretending that space
    # came for free would be the flattering half of the trade only.
    def __init__(self, freed_layer_capacity: int, parameter_stream_ms: float) -> None:
        if freed_layer_capacity < 0:
            raise ValueError("freed_layer_capacity must be non-negative")
        if parameter_stream_ms < 0:
            raise ValueError("parameter_stream_ms must be non-negative")
        self._freed_layer_capacity = freed_layer_capacity
        self._parameter_stream_ms = parameter_stream_ms

    def run(self, workload: OffloadWorkload) -> OffloadRun:
        resident = min(
            workload.gpu_layer_capacity + self._freed_layer_capacity, workload.num_layers
        )
        fetches = [
            self._parameter_stream_ms
            if layer < resident
            else workload.warm_fetch_ms_per_layer + self._parameter_stream_ms
            for layer in range(workload.num_layers)
        ]
        total, hidden = pipeline_total(
            workload.num_layers, workload.compute_ms_per_layer, fetches
        )
        return OffloadRun(
            strategy="oneiros",
            total_ms=total,
            transfer_ms_issued=sum(fetches),
            transfer_ms_hidden=hidden,
            gpu_layers_resident=resident,
        )
