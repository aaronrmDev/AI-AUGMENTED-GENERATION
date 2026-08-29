import json
import uuid
from datetime import UTC, datetime
from typing import Any

from src.mag.domain.entities import EpisodicMemory
from src.mag.domain.ports import (
    EpisodicMemoryIndex,
    EpisodicMemoryRepository,
    MemoryGraphRepository,
)
from src.mag.infrastructure._graph_write_safety import best_effort_graph_write
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
        memory_graph_repository: MemoryGraphRepository,
    ) -> None:
        self._episodes = episodic_memory_repository
        self._index = episodic_memory_index
        self._embedder = embedding_model
        self._chat_model = chat_model
        self._graph = memory_graph_repository

    async def execute(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        content: dict[str, Any],
    ) -> EpisodicMemory:
        # Fetched before saving the new episode -- this is the episode
        # TEMPORALLY_FOLLOWS will link from, so it has to be "the most
        # recent episode as of before this one," not after. get_recent_by_
        # session(limit=1) (MAG Batch C) is the cheap way to ask for just
        # that, rather than get_by_session's full-session fetch.
        previous = await self._episodes.get_recent_by_session(session_id, tenant_id, limit=1)

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

        # Mirrors this episode into the memory graph (MAG Batch D) -- best
        # effort, not blocking: see _graph_write_safety for why a Neo4j
        # failure here must never roll back or mask the writes above, which
        # already succeeded.
        await best_effort_graph_write(
            self._graph.upsert_episode_node(episode, tenant_id), "upsert episode node"
        )
        await best_effort_graph_write(
            self._graph.link_participated_in(user_id, session_id, tenant_id),
            "link participated_in",
        )
        if previous:
            await best_effort_graph_write(
                self._graph.link_temporally_follows(previous[0].id, episode.id, tenant_id),
                "link temporally_follows",
            )
        # Reuses the same content["entities"] field MAG Batch C's
        # EntityRetrieval already reads structurally, rather than inventing
        # a second entity-extraction mechanism for the same data.
        for entity_name in content.get("entities", []):
            await best_effort_graph_write(
                self._graph.link_mentions(episode.id, entity_name, tenant_id),
                f"link mentions ({entity_name})",
            )

        return episode

    async def _score_salience(self, content: dict[str, Any]) -> float:
        prompt = f"{SALIENCE_SYSTEM_PROMPT}\n\n{build_salience_user_message(content)}"
        for _ in range(_MAX_SALIENCE_ATTEMPTS):
            response = await self._chat_model.complete(prompt)
            try:
                parsed = json.loads(strip_markdown_fence(response))
                raw_score = parsed["salience_score"]
                # bool is a subclass of int in Python, so isinstance(True, (int,
                # float)) is True and float(True) == 1.0 -- without this guard a
                # model returning {"salience_score": true} would silently pass
                # the range check as a maximal score, on the first attempt, with
                # no retry. Same guard CausalRetrieval's _validate_scores uses.
                if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                    raise TypeError("salience_score must be a number")
                score = float(raw_score)
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
