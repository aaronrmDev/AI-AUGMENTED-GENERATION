import uuid

from src.mag.domain.entities import ActivatedNode
from src.mag.domain.ports import MemoryGraphRepository

# Matches MAG.md's own example values closely enough to be a reasonable
# default (a handful of hops, activation halving per hop, a small floor
# below which a node isn't worth returning as context) without claiming
# these are tuned against any real workload -- a caller with a better-
# informed choice overrides them.
_DEFAULT_MAX_HOPS = 3
_DEFAULT_DECAY_FACTOR = 0.5
_DEFAULT_ACTIVATION_THRESHOLD = 0.05


class SpreadingActivationRetrieval:
    # Thin wrapper: the actual traversal algorithm lives in
    # MemoryGraphRepository.spread_activation (Neo4j is where the
    # traversal itself should run natively -- see the port docstring and
    # the design spec for why this isn't hand-rolled BFS in Python).
    def __init__(self, memory_graph_repository: MemoryGraphRepository) -> None:
        self._graph = memory_graph_repository

    async def execute(
        self,
        tenant_id: uuid.UUID,
        start_entity_names: list[str],
        max_hops: int = _DEFAULT_MAX_HOPS,
        decay_factor: float = _DEFAULT_DECAY_FACTOR,
        activation_threshold: float = _DEFAULT_ACTIVATION_THRESHOLD,
    ) -> list[ActivatedNode]:
        return await self._graph.spread_activation(
            tenant_id=tenant_id,
            start_entity_names=start_entity_names,
            max_hops=max_hops,
            decay_factor=decay_factor,
            activation_threshold=activation_threshold,
        )
