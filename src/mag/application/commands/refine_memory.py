import json
import uuid
from datetime import UTC, datetime

from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.domain.entities import SemanticMemory, SemanticMemoryHistoryEntry
from src.mag.domain.ports import SemanticMemoryRepository
from src.mag.infrastructure._llm_json import strip_markdown_fence
from src.mag.infrastructure._refine_prompt import REFINE_SYSTEM_PROMPT, build_refine_user_message
from src.rag.domain.ports import ChatModel

_MAX_REFINE_ATTEMPTS = 3


class RefineMemory:
    # Refine (#66): new information adds nuance without contradicting the
    # existing fact -- merges old and new into one richer fact rather than
    # discarding either, the gap between Update ("this fact is false,
    # replace it") and doing nothing ("this fact is fine, leave it alone").
    def __init__(
        self,
        semantic_memory_repository: SemanticMemoryRepository,
        record_semantic_fact: RecordSemanticFact,
        chat_model: ChatModel,
    ) -> None:
        self._repository = semantic_memory_repository
        self._record_semantic_fact = record_semantic_fact
        self._chat_model = chat_model

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, fact_key: str, new_information: str
    ) -> SemanticMemory:
        existing = await self._repository.find_by_key(user_id, fact_key, tenant_id)
        if existing is None:
            raise ValueError(
                f"no existing fact for user_id={user_id} fact_key={fact_key!r} to refine"
            )
        merged_fact_value = await self._merge(existing.fact_value, new_information)
        await self._repository.save_history_entry(
            SemanticMemoryHistoryEntry(
                id=uuid.uuid4(),
                original_fact_id=existing.id,
                user_id=user_id,
                fact_key=fact_key,
                fact_value=existing.fact_value,
                confidence=existing.confidence,
                source=existing.source,
                operation="refine",
                superseded_at=datetime.now(UTC),
            ),
            tenant_id,
        )
        # Same status-preservation reasoning as UpdateMemory -- Refine
        # merges content, it must not silently un-invalidate or
        # un-archive a fact as a side effect.
        return await self._record_semantic_fact.execute(
            tenant_id=tenant_id,
            user_id=user_id,
            fact_key=fact_key,
            fact_value=merged_fact_value,
            confidence=existing.confidence,
            source=existing.source,
            valid_until=existing.valid_until,
            archived_at=existing.archived_at,
        )

    async def _merge(self, existing_fact_value: str, new_information: str) -> str:
        prompt = (
            f"{REFINE_SYSTEM_PROMPT}\n\n"
            f"{build_refine_user_message(existing_fact_value, new_information)}"
        )
        for _ in range(_MAX_REFINE_ATTEMPTS):
            response = await self._chat_model.complete(prompt)
            try:
                parsed = json.loads(strip_markdown_fence(response))
                merged = parsed["merged_fact_value"]
                if not isinstance(merged, str) or not merged.strip():
                    raise TypeError("merged_fact_value must be a non-empty string")
                return merged
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        # Exhausted retries: a plain concatenation, not a crash and not
        # silently dropping the new information -- merge is a hot,
        # synchronous path (unlike Consolidation's batch reflection), and
        # an inelegantly-merged fact is still more useful than none.
        return f"{existing_fact_value}; {new_information}"
