import uuid
from datetime import UTC, datetime

from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.domain.entities import SemanticMemory, SemanticMemoryHistoryEntry
from src.mag.domain.ports import SemanticMemoryRepository


class UpdateMemory:
    # Update (#62): new information directly contradicts old information,
    # action is a straightforward overwrite -- but "archived with its
    # timestamp rather than simply deleted" (#62's own worked example)
    # means the old value gets snapshotted first, not just discarded.
    def __init__(
        self,
        semantic_memory_repository: SemanticMemoryRepository,
        record_semantic_fact: RecordSemanticFact,
    ) -> None:
        self._repository = semantic_memory_repository
        self._record_semantic_fact = record_semantic_fact

    async def execute(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        fact_key: str,
        new_fact_value: str,
        confidence: float = 1.0,
        source: str = "",
    ) -> SemanticMemory:
        existing = await self._repository.find_by_key(user_id, fact_key, tenant_id)
        if existing is None:
            # Updating a fact that was never recorded is a caller bug, not
            # a valid degenerate case -- there's nothing to overwrite and
            # no old value to preserve, so RecordSemanticFact (a genuinely
            # new fact) is the right call instead, not this one.
            raise ValueError(
                f"no existing fact for user_id={user_id} fact_key={fact_key!r} to update"
            )
        await self._repository.save_history_entry(
            SemanticMemoryHistoryEntry(
                id=uuid.uuid4(),
                original_fact_id=existing.id,
                user_id=user_id,
                fact_key=fact_key,
                fact_value=existing.fact_value,
                confidence=existing.confidence,
                source=existing.source,
                operation="update",
                superseded_at=datetime.now(UTC),
            ),
            tenant_id,
        )
        # The actual overwrite (Postgres upsert, Qdrant upsert, best-effort
        # Neo4j sync) is exactly what RecordSemanticFact already does --
        # composing it here rather than duplicating that three-store dance.
        # valid_until/archived_at are explicitly carried over from the
        # existing fact -- RecordSemanticFact defaults both to None, and
        # without this an Update on a previously Invalidated or Archived
        # fact would silently reset its status as an unrelated side effect
        # of a content correction (confirmed as a real bug by review).
        # Content and status are independent; updating one must not
        # silently clear the other.
        return await self._record_semantic_fact.execute(
            tenant_id=tenant_id,
            user_id=user_id,
            fact_key=fact_key,
            fact_value=new_fact_value,
            confidence=confidence,
            source=source,
            valid_until=existing.valid_until,
            archived_at=existing.archived_at,
        )
