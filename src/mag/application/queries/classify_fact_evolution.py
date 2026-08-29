import json
import uuid

from src.mag.domain.entities import FactEvolutionClassification
from src.mag.domain.ports import SemanticMemoryRepository
from src.mag.infrastructure._fact_evolution_prompt import (
    FACT_EVOLUTION_SYSTEM_PROMPT,
    build_fact_evolution_user_message,
)
from src.mag.infrastructure._llm_json import strip_markdown_fence
from src.rag.domain.ports import ChatModel

_MAX_CLASSIFICATION_ATTEMPTS = 3
_VALID_OPERATIONS = {"update", "invalidate", "refine", "no_conflict"}


class ClassifyFactEvolution:
    # Answers MAG.md's own "comparison" step (issue #16): given an
    # existing fact and a new piece of information, is this a correction
    # (update), a sign the old fact isn't true anymore with nothing to
    # replace it (invalidate), added nuance (refine), or unrelated to this
    # fact entirely (no_conflict)? A judgment, not a write -- lives in
    # queries/ alongside this project's other retrieval-strategy classes,
    # not commands/, since nothing here mutates storage.
    def __init__(
        self, semantic_memory_repository: SemanticMemoryRepository, chat_model: ChatModel
    ) -> None:
        self._repository = semantic_memory_repository
        self._chat_model = chat_model

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, fact_key: str, new_information: str
    ) -> FactEvolutionClassification:
        existing = await self._repository.find_by_key(user_id, fact_key, tenant_id)
        if existing is None:
            raise ValueError(
                f"no existing fact for user_id={user_id} fact_key={fact_key!r} to classify against"
            )
        prompt = (
            f"{FACT_EVOLUTION_SYSTEM_PROMPT}\n\n"
            f"{build_fact_evolution_user_message(existing.fact_value, new_information)}"
        )
        for _ in range(_MAX_CLASSIFICATION_ATTEMPTS):
            response = await self._chat_model.complete(prompt)
            try:
                parsed = json.loads(strip_markdown_fence(response))
                operation = parsed["operation"]
                reasoning = parsed.get("reasoning", "")
                if not isinstance(operation, str) or operation not in _VALID_OPERATIONS:
                    raise TypeError(
                        "operation must be one of update/invalidate/refine/no_conflict"
                    )
                if not isinstance(reasoning, str):
                    raise TypeError("reasoning must be a string")
                return FactEvolutionClassification(operation=operation, reasoning=reasoning)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        # Exhausted retries: no_conflict, the least destructive outcome --
        # a caller dispatching on this takes no action, which is always
        # safe, unlike guessing "update" and overwriting a fact on a
        # response this method couldn't actually make sense of.
        return FactEvolutionClassification(
            operation="no_conflict",
            reasoning="classification failed after retries; defaulting to no-op",
        )
