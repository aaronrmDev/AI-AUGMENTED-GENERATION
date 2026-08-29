import json
import uuid
from datetime import UTC, datetime
from typing import Any

from src.mag.domain.entities import EpisodicMemory
from src.mag.domain.ports import EpisodicMemoryIndex, EpisodicMemoryRepository
from src.mag.infrastructure._llm_json import strip_markdown_fence
from src.mag.infrastructure._salience_prompt import (
    SALIENCE_SYSTEM_PROMPT,
    build_salience_user_message,
)
from src.rag.domain.ports import ChatModel, EmbeddingModel

# Same reasoning as ConsolidateEpisodes's _MAX_REFLECTION_ATTEMPTS (#149's
# retry pattern): complete() has no forced JSON mode, so a malformed or
# fenced response is real, not theoretical.
_MAX_SALIENCE_ATTEMPTS = 3


class CaptureEpisode:
    def __init__(
        self,
        episodic_memory_repository: EpisodicMemoryRepository,
        episodic_memory_index: EpisodicMemoryIndex,
        embedding_model: EmbeddingModel,
        chat_model: ChatModel,
    ) -> None:
        self._episodes = episodic_memory_repository
        self._index = episodic_memory_index
        self._embedder = embedding_model
        self._chat_model = chat_model

    async def execute(
        self, tenant_id: uuid.UUID, session_id: uuid.UUID, content: dict[str, Any]
    ) -> EpisodicMemory:
        # Embeds the whole event (json.dumps, keys sorted for a stable
        # embedding regardless of dict insertion order) rather than one
        # designated field -- content's shape (input/reasoning/tool_calls/
        # output/actors/entities) has no single field guaranteed to carry the
        # episode's meaning, and search_by_similarity has to be able to match
        # on any of them.
        embedding = self._embedder.embed(json.dumps(content, sort_keys=True))
        episode = EpisodicMemory(
            id=uuid.uuid4(),
            session_id=session_id,
            content=content,
            embedding=embedding,
            timestamp=datetime.now(UTC),
            salience_score=await self._score_salience(content),
        )
        await self._episodes.save(episode, tenant_id)
        await self._index.upsert(episode, tenant_id)
        return episode

    async def _score_salience(self, content: dict[str, Any]) -> float:
        prompt = f"{SALIENCE_SYSTEM_PROMPT}\n\n{build_salience_user_message(content)}"
        for _ in range(_MAX_SALIENCE_ATTEMPTS):
            response = await self._chat_model.complete(prompt)
            try:
                parsed = json.loads(strip_markdown_fence(response))
                score = float(parsed["salience_score"])
                if not (0.0 <= score <= 1.0):
                    raise ValueError("salience_score out of [0.0, 1.0] range")
                return score
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        # Exhausted retries: 0.0, not a crash -- capture is a hot,
        # synchronous path (unlike Consolidation's batch reflection), and an
        # unscored episode is still a valid episode. Matches
        # ConsolidateEpisodes's identical "fail safe, not fail loud" choice
        # for a retrieval/write-time LLM call.
        return 0.0
