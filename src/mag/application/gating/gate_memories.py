from datetime import datetime

from src.mag.application.gating._candidates import (
    from_activated_node,
    from_scored_episode,
    from_scored_fact,
)
from src.mag.application.gating.dynamic_reranking import DynamicReranking
from src.mag.application.gating.hierarchical_assembly import HierarchicalAssembly
from src.mag.application.gating.recency_weighted_sampling import RecencyWeightedSampling
from src.mag.application.gating.task_specific_filtering import TaskSpecificFiltering
from src.mag.application.gating.token_budget_allocation import TokenBudgetAllocation
from src.mag.domain.entities import ActivatedNode, GatingCandidate, ScoredEpisode, ScoredFact


class GateMemories:
    # #15's own pipeline description: "retrieve candidates across every
    # tier and strategy, score them by a composite of similarity, recency,
    # salience, and task fit, filter by hard constraints (max tokens...),
    # assemble the prompt with the resulting ordering." This composes five
    # of the six sibling gating strategies into that pipeline; TopKSelection
    # is deliberately NOT one of this pipeline's own stages (see the design
    # spec's "The pipeline: GateMemories" section for why TokenBudget
    # Allocation is the better default for a mixed episode/fact/graph-node
    # pool) -- it remains fully implemented and independently usable on its
    # own, just not wired into this particular default assembly.
    def __init__(self) -> None:
        self._dynamic_reranking = DynamicReranking()
        self._task_specific_filtering = TaskSpecificFiltering()
        self._recency_weighted_sampling = RecencyWeightedSampling()
        self._token_budget_allocation = TokenBudgetAllocation()
        self._hierarchical_assembly = HierarchicalAssembly()

    async def execute(
        self,
        episodes: list[ScoredEpisode],
        facts: list[ScoredFact],
        graph_nodes: list[ActivatedNode],
        token_budget: int,
        query_embedding: list[float] | None = None,
        allowed_source_types: set[str] | None = None,
        recency_half_life_hours: float = 24.0,
        now: datetime | None = None,
    ) -> list[GatingCandidate]:
        candidates = (
            [from_scored_episode(e) for e in episodes]
            + [from_scored_fact(f) for f in facts]
            + [from_activated_node(n) for n in graph_nodes]
        )

        # Dynamic re-ranking and task filtering are explicit-parameter,
        # skip-if-not-given stages -- same convention Batch C's
        # RecencyDecayFusionRetrieval established for its own optional
        # legs: a caller supplies what it has, this pipeline runs whatever
        # it was given enough to run rather than guessing.
        if query_embedding is not None:
            candidates = await self._dynamic_reranking.execute(candidates, query_embedding)
        if allowed_source_types is not None:
            candidates = await self._task_specific_filtering.execute(
                candidates, allowed_source_types
            )

        candidates = await self._recency_weighted_sampling.execute(
            candidates, half_life_hours=recency_half_life_hours, now=now
        )
        candidates = await self._token_budget_allocation.execute(candidates, token_budget)
        return await self._hierarchical_assembly.execute(candidates)
