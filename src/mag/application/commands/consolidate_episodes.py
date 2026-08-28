import json
import uuid
from typing import Any

from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.domain.entities import EpisodicMemory, SemanticMemory
from src.mag.domain.ports import (
    EpisodicMemoryRepository,
    SemanticMemoryIndex,
    SemanticMemoryRepository,
)
from src.mag.infrastructure._consolidation_prompt import (
    CONSOLIDATION_SYSTEM_PROMPT,
    build_consolidation_user_message,
)
from src.rag.domain.ports import ChatModel, EmbeddingModel

# Same reasoning as OllamaJudge.score() (#149): complete() has no forced
# JSON mode (unlike the judge's raw ollama client call with format="json"),
# so a malformed or fenced response is a real, observed possibility, not a
# theoretical one -- retry a bounded number of times before giving up.
_MAX_REFLECTION_ATTEMPTS = 3


class ConsolidateEpisodes:
    def __init__(
        self,
        episodic_memory_repository: EpisodicMemoryRepository,
        semantic_memory_repository: SemanticMemoryRepository,
        semantic_memory_index: SemanticMemoryIndex,
        embedding_model: EmbeddingModel,
        chat_model: ChatModel,
    ) -> None:
        self._episodes = episodic_memory_repository
        self._record_fact = RecordSemanticFact(
            semantic_memory_repository=semantic_memory_repository,
            semantic_memory_index=semantic_memory_index,
            embedding_model=embedding_model,
        )
        self._chat_model = chat_model

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, batch_size: int = 10
    ) -> list[SemanticMemory]:
        episodes = await self._episodes.get_unconsolidated_by_session(
            session_id, tenant_id, limit=batch_size
        )
        if not episodes:
            return []

        facts_raw = await self._reflect(episodes)

        written: list[SemanticMemory] = []
        for fact in facts_raw:
            written.append(
                await self._record_fact.execute(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    fact_key=fact["fact_key"],
                    fact_value=fact["fact_value"],
                    confidence=float(fact.get("confidence", 1.0)),
                    source="consolidation",
                )
            )

        # Marked regardless of whether reflection produced any facts --
        # see the design spec's Consolidation section and
        # EpisodicMemoryRepository.mark_consolidated's docstring for why a
        # fact-free episode still shouldn't be re-sent to the LLM forever.
        await self._episodes.mark_consolidated([e.id for e in episodes], tenant_id)
        return written

    async def _reflect(self, episodes: list[EpisodicMemory]) -> list[dict[str, Any]]:
        episode_dicts = [
            {"content": e.content, "timestamp": e.timestamp.isoformat()} for e in episodes
        ]
        prompt = (
            f"{CONSOLIDATION_SYSTEM_PROMPT}\n\n"
            f"{build_consolidation_user_message(episode_dicts)}"
        )
        for _ in range(_MAX_REFLECTION_ATTEMPTS):
            response = await self._chat_model.complete(prompt)
            try:
                parsed = json.loads(_strip_markdown_fence(response))
                facts = parsed["facts"]
                if not isinstance(facts, list):
                    raise TypeError("'facts' must be a list")
                return _validate_and_dedupe_facts(facts)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        # Exhausted retries: treated the same as "nothing durable found,"
        # not raised -- the episodes stay marked consolidated (this batch
        # was reflected on) but produce no facts. A failed reflection pass
        # shouldn't crash the caller or hold the batch hostage; the honest
        # alternative (leaving episodes unconsolidated forever after a
        # parse failure) risks the same episodes failing the same way on
        # every future run if the failure mode is content-specific, not
        # transient.
        return []


def _validate_and_dedupe_facts(facts: list[Any]) -> list[dict[str, Any]]:
    # The outer envelope ({"facts": [...]}) being valid JSON says nothing
    # about what's INSIDE each element -- an LLM extracting facts from an
    # open-ended reflection prompt can return a bare string instead of an
    # object, rename fields, or hand back confidence as a word ("high")
    # instead of a number. Any of those used to reach RecordSemanticFact
    # directly and crash outside this method's retry loop -- which also
    # meant the episodes never got marked consolidated (they'd fail the
    # same way again on the next run) and, worse, could leave a fact
    # already written to Qdrant by an EARLIER iteration of this same loop
    # orphaned when a LATER iteration's crash rolled back the caller's
    # Postgres transaction. Validating here folds a malformed element into
    # the same retry path as malformed JSON, rather than a second,
    # unguarded failure mode one call-frame up.
    validated: list[dict[str, Any]] = []
    seen_keys: dict[str, int] = {}
    for item in facts:
        if not isinstance(item, dict):
            raise TypeError("each fact must be a JSON object")
        fact_key = item["fact_key"]
        fact_value = item["fact_value"]
        if not isinstance(fact_key, str) or not fact_key.strip():
            raise TypeError("fact_key must be a non-empty string")
        if not isinstance(fact_value, str) or not fact_value.strip():
            raise TypeError("fact_value must be a non-empty string")
        confidence = float(item.get("confidence", 1.0))  # ValueError on e.g. "high"
        record = {"fact_key": fact_key, "fact_value": fact_value, "confidence": confidence}
        # Last-wins on a duplicate fact_key within one reflection response:
        # RecordSemanticFact's deterministic id means both would resolve to
        # the same row anyway (the second write overwrites the first), so
        # returning both as if they were two independently-written facts
        # would report one that was never actually the final, persisted
        # value.
        if fact_key in seen_keys:
            validated[seen_keys[fact_key]] = record
        else:
            seen_keys[fact_key] = len(validated)
            validated.append(record)
    return validated


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening fence (optionally "```json") and a trailing
        # closing fence if present -- a model ignoring "no markdown
        # fencing" is exactly the kind of non-compliance #149 established
        # this project can't assume away.
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        stripped = "\n".join(lines)
    return stripped
