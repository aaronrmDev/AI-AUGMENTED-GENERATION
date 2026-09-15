def overlap_efficiency(transfer_ms_issued: float, transfer_ms_hidden: float) -> float:
    # The single claim CAG.md makes about every strategy in this family:
    # "the art across every one of these strategies is hiding transfer
    # latency behind computation that's already happening anyway." This
    # is that art, scored -- the fraction of issued transfer time that
    # never appeared on the critical path because compute was running
    # underneath it. 1.0 means every byte moved for free; 0.0 means the
    # pipeline stalled for all of it.
    if transfer_ms_issued < 0 or transfer_ms_hidden < 0:
        raise ValueError("transfer times must be non-negative")
    if transfer_ms_hidden > transfer_ms_issued:
        raise ValueError("hidden transfer time cannot exceed issued transfer time")
    if transfer_ms_issued == 0:
        return 1.0
    return transfer_ms_hidden / transfer_ms_issued


def effective_context_multiplier(
    gpu_layer_capacity: int, layers_served: int
) -> float:
    # How far past the GPU's own capacity a tiering strategy stretched
    # the context, which is the reason CAG.md gives for the technique
    # existing at all: "extend the effective context window beyond
    # whatever a single GPU's memory allows."
    if gpu_layer_capacity < 1:
        raise ValueError("gpu_layer_capacity must be at least 1")
    if layers_served < 0:
        raise ValueError("layers_served must be non-negative")
    return layers_served / gpu_layer_capacity


def latency_cost_ratio(baseline_ms: float, offloaded_ms: float) -> float:
    # The other half of CAG.md's stated trade: "GPU memory saved rises
    # together with latency." A ratio of 1.0 means offloading cost
    # nothing in time; 2.0 means it doubled the run. Reported alongside
    # the memory saved rather than separately, since neither number
    # means anything without the other.
    if baseline_ms <= 0:
        raise ValueError("baseline_ms must be positive")
    if offloaded_ms < 0:
        raise ValueError("offloaded_ms must be non-negative")
    return offloaded_ms / baseline_ms
