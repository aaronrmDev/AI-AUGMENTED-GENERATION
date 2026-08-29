import uuid

from src.mag.application.commands.invalidate_memory import InvalidateMemory
from src.mag.application.commands.refine_memory import RefineMemory
from src.mag.application.commands.update_memory import UpdateMemory
from src.mag.application.queries.classify_fact_evolution import ClassifyFactEvolution
from src.mag.domain.entities import FactEvolutionClassification, SemanticMemory


class EvolveMemory:
    # The end-to-end pipeline MAG.md's own Memory Evolution description
    # narrates (#16): detect (the caller already knows which fact a new
    # piece of information is about -- see the design spec for why this
    # batch doesn't build automatic similarity-based detection), compare
    # (ClassifyFactEvolution), decide, and dispatch. Composes three of
    # Batch F's four operations the same way Batch E's GateMemories
    # composed five of its own six siblings into one pipeline.
    #
    # Archive is deliberately NOT one of the three: its trigger (#64,
    # access frequency) has nothing to do with comparing new information
    # against old, so it doesn't fit this orchestrator's shape -- it stays
    # independently invocable, the same way TopKSelection stayed fully
    # built but outside GateMemories's own default composition.
    def __init__(
        self,
        classify_fact_evolution: ClassifyFactEvolution,
        update_memory: UpdateMemory,
        invalidate_memory: InvalidateMemory,
        refine_memory: RefineMemory,
    ) -> None:
        self._classify = classify_fact_evolution
        self._update = update_memory
        self._invalidate = invalidate_memory
        self._refine = refine_memory

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, fact_key: str, new_information: str
    ) -> tuple[FactEvolutionClassification, SemanticMemory | None]:
        classification = await self._classify.execute(
            tenant_id=tenant_id, user_id=user_id, fact_key=fact_key, new_information=new_information
        )
        if classification.operation == "update":
            result: SemanticMemory | None = await self._update.execute(
                tenant_id=tenant_id,
                user_id=user_id,
                fact_key=fact_key,
                new_fact_value=new_information,
            )
        elif classification.operation == "refine":
            result = await self._refine.execute(
                tenant_id=tenant_id,
                user_id=user_id,
                fact_key=fact_key,
                new_information=new_information,
            )
        elif classification.operation == "invalidate":
            # No replacement value -- matches #63's own "without
            # necessarily replacing it with anything."
            result = await self._invalidate.execute(
                tenant_id=tenant_id, user_id=user_id, fact_key=fact_key
            )
        else:  # "no_conflict" -- unrelated context, nothing to do to this fact
            result = None
        return classification, result
