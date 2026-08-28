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
                return list(facts)
            except (json.JSONDecodeError, KeyError, TypeError):
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
