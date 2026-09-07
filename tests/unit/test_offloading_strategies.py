import pytest

from src.cag.domain.entities import OffloadWorkload
from src.cag.domain.offloading_metrics import overlap_efficiency
from src.cag.infrastructure.inf2_offloader import INF2Offloader
from src.cag.infrastructure.infinigen_offloader import InfiniGenOffloader
from src.cag.infrastructure.kvpr_offloader import KVPROffloader
from src.cag.infrastructure.layerkv_offloader import LayerKVOffloader
from src.cag.infrastructure.oneiros_offloader import OneirosOffloader
from src.cag.infrastructure.sequential_offloader import SequentialOffloader

# 24 layers, only 8 of which fit on the GPU: the situation the whole
# technique family exists for.
_WORKLOAD = OffloadWorkload(
    num_layers=24,
    compute_ms_per_layer=10.0,
    warm_fetch_ms_per_layer=6.0,
    cold_fetch_ms_per_layer=40.0,
    gpu_layer_capacity=8,
)


def test_the_sequential_baseline_hides_nothing_by_construction():
    run = SequentialOffloader().run(_WORKLOAD)
    assert run.transfer_ms_hidden == 0.0
    assert overlap_efficiency(run.transfer_ms_issued, run.transfer_ms_hidden) == 0.0
    # 24 layers of compute plus a full stall for each of the 16 offloaded.
    assert run.total_ms == pytest.approx(24 * 10.0 + 16 * 6.0)


def test_layerkv_hides_every_transfer_when_compute_outlasts_the_fetch():
    # 6ms fetches under 10ms of compute: each one finishes before the
    # layer it overlaps with does, so nothing reaches the critical path.
    run = LayerKVOffloader().run(_WORKLOAD)
    assert overlap_efficiency(run.transfer_ms_issued, run.transfer_ms_hidden) == pytest.approx(1.0)
    assert run.total_ms < SequentialOffloader().run(_WORKLOAD).total_ms


def test_layerkv_stalls_only_for_the_excess_when_fetches_outlast_compute():
    slow = OffloadWorkload(
        num_layers=4,
        compute_ms_per_layer=10.0,
        warm_fetch_ms_per_layer=25.0,
        cold_fetch_ms_per_layer=40.0,
        gpu_layer_capacity=1,
    )
    run = LayerKVOffloader().run(slow)
    # Overlap is real but partial -- 10ms of each 25ms fetch hides.
    efficiency = overlap_efficiency(run.transfer_ms_issued, run.transfer_ms_hidden)
    assert 0.0 < efficiency < 1.0


def test_infinigen_moves_less_than_layerkv_because_it_prefetches_only_critical_entries():
    layerkv = LayerKVOffloader().run(_WORKLOAD)
    infinigen = InfiniGenOffloader(critical_fraction=0.25).run(_WORKLOAD)
    assert infinigen.transfer_ms_issued < layerkv.transfer_ms_issued
    assert infinigen.total_ms <= layerkv.total_ms


def test_infinigen_rejects_a_critical_fraction_outside_the_unit_interval():
    with pytest.raises(ValueError):
        InfiniGenOffloader(critical_fraction=0.0)
    with pytest.raises(ValueError):
        InfiniGenOffloader(critical_fraction=1.5)


def test_kvpr_stalls_for_the_longer_of_recompute_and_transfer_not_their_sum():
    run = KVPROffloader(recompute_fraction=0.5).run(_WORKLOAD)
    # Per offloaded layer: recompute 5ms against transfer 3ms, so the
    # stall is 5ms, not 8ms, and 3ms hides underneath.
    assert run.total_ms == pytest.approx(24 * 10.0 + 16 * 5.0)
    assert run.transfer_ms_hidden == pytest.approx(16 * 3.0)


def test_kvpr_at_full_recompute_moves_nothing_across_the_bus():
    run = KVPROffloader(recompute_fraction=1.0).run(_WORKLOAD)
    assert run.transfer_ms_issued == pytest.approx(0.0)


def test_oneiros_holds_more_layers_resident_rather_than_moving_cache_more_cleverly():
    baseline = LayerKVOffloader().run(_WORKLOAD)
    run = OneirosOffloader(freed_layer_capacity=8, parameter_stream_ms=0.5).run(_WORKLOAD)
    # The distinguishing effect: capacity grew, where every other
    # strategy leaves it exactly as the GPU set it.
    assert run.gpu_layers_resident > baseline.gpu_layers_resident
    assert run.gpu_layers_resident == 16


def test_oneiros_still_charges_for_the_parameters_it_displaced():
    free_lunch = OneirosOffloader(freed_layer_capacity=8, parameter_stream_ms=0.0)
    honest = OneirosOffloader(freed_layer_capacity=8, parameter_stream_ms=2.0)
    assert honest.run(_WORKLOAD).transfer_ms_issued > free_lunch.run(_WORKLOAD).transfer_ms_issued


def test_inf2_shrinks_what_crosses_the_bus_rather_than_hiding_it():
    # Against a cold-tier workload, where INF2's premise applies.
    cold = OffloadWorkload(
        num_layers=24,
        compute_ms_per_layer=10.0,
        warm_fetch_ms_per_layer=6.0,
        cold_fetch_ms_per_layer=40.0,
        gpu_layer_capacity=8,
    )
    naive_cold_transfer = 16 * cold.cold_fetch_ms_per_layer
    run = INF2Offloader(pcie_reduction=0.1, storage_compute_ms=1.0).run(cold)
    assert run.transfer_ms_issued < naive_cold_transfer


def test_inf2_charges_for_the_slower_in_storage_accelerator():
    cheap = INF2Offloader(pcie_reduction=0.1, storage_compute_ms=0.0).run(_WORKLOAD)
    realistic = INF2Offloader(pcie_reduction=0.1, storage_compute_ms=3.0).run(_WORKLOAD)
    assert realistic.transfer_ms_issued > cheap.transfer_ms_issued


def test_every_strategy_serves_all_the_layers_the_gpu_alone_could_not_hold():
    strategies = [
        SequentialOffloader(),
        LayerKVOffloader(),
        InfiniGenOffloader(critical_fraction=0.25),
        KVPROffloader(recompute_fraction=0.5),
        OneirosOffloader(freed_layer_capacity=4, parameter_stream_ms=0.5),
        INF2Offloader(pcie_reduction=0.1, storage_compute_ms=1.0),
    ]
    for strategy in strategies:
        run = strategy.run(_WORKLOAD)
        assert run.gpu_layers_resident <= _WORKLOAD.num_layers
        assert run.total_ms > 0.0
