import uuid

from src.mag.application.queries.retrieve_by_spreading_activation import (
    SpreadingActivationRetrieval,
)
from src.mag.domain.entities import ActivatedNode
from tests.unit.mag_fakes import FakeMemoryGraphRepository


def _node(node_id: str, activation: float, hops: int) -> ActivatedNode:
    return ActivatedNode(
        node_id=node_id, node_type="Entity", properties={}, activation=activation, hops=hops
    )


async def test_execute_delegates_to_the_graph_repository_and_returns_its_results():
    results = [_node("a", 1.0, 0), _node("b", 0.5, 1)]
    graph = FakeMemoryGraphRepository()
    graph.set_spread_activation_results(results)
    tenant_id = uuid.uuid4()

    result = await SpreadingActivationRetrieval(graph).execute(
        tenant_id=tenant_id, start_entity_names=["Paris"]
    )

    assert result == results


async def test_execute_uses_documented_default_parameters_when_not_overridden():
    graph = FakeMemoryGraphRepository()
    captured: dict[str, object] = {}

    async def _spy(**kwargs: object) -> list[ActivatedNode]:
        captured.update(kwargs)
        return []

    graph.spread_activation = _spy  # type: ignore[method-assign]

    await SpreadingActivationRetrieval(graph).execute(
        tenant_id=uuid.uuid4(), start_entity_names=["Bob"]
    )

    assert captured["max_hops"] == 3
    assert captured["decay_factor"] == 0.5
    assert captured["activation_threshold"] == 0.05


async def test_execute_passes_through_caller_supplied_overrides():
    graph = FakeMemoryGraphRepository()
    captured: dict[str, object] = {}

    async def _spy(**kwargs: object) -> list[ActivatedNode]:
        captured.update(kwargs)
        return []

    graph.spread_activation = _spy  # type: ignore[method-assign]

    await SpreadingActivationRetrieval(graph).execute(
        tenant_id=uuid.uuid4(),
        start_entity_names=["Bob"],
        max_hops=5,
        decay_factor=0.7,
        activation_threshold=0.1,
    )

    assert captured["max_hops"] == 5
    assert captured["decay_factor"] == 0.7
    assert captured["activation_threshold"] == 0.1


async def test_execute_returns_an_empty_list_when_nothing_is_activated():
    graph = FakeMemoryGraphRepository()

    result = await SpreadingActivationRetrieval(graph).execute(
        tenant_id=uuid.uuid4(), start_entity_names=["Nonexistent"]
    )

    assert result == []
