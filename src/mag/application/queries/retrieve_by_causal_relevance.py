import json
import uuid
from typing import Any

from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryRepository
from src.mag.infrastructure._causal_prompt import (
    CAUSAL_SYSTEM_PROMPT,
    build_causal_user_message,
)
from src.mag.infrastructure._llm_json import strip_markdown_fence
from src.rag.domain.ports import ChatModel

# Same reasoning as ConsolidateEpisodes's _MAX_REFLECTION_ATTEMPTS (#149):
# complete() has no forced JSON mode, so a malformed or fenced response is a
# real, observed possibility, not a theoretical one -- retry a bounded
# number of times before giving up.
_MAX_CAUSAL_ATTEMPTS = 3


class CausalRetrieval:
    def __init__(
        self, episodic_memory_repository: EpisodicMemoryRepository, chat_model: ChatModel
    ) -> None:
        self._episodes = episodic_memory_repository
        self._chat_model = chat_model

    async def execute(
        self, tenant_id: uuid.UUID, session_id: uuid.UUID, query: str, top_k: int
    ) -> list[ScoredEpisode]:
        episodes = await self._episodes.get_by_session(session_id, tenant_id)
        if not episodes:
            return []

        episode_dicts = [
            {"content": e.content, "timestamp": e.timestamp.isoformat()} for e in episodes
        ]
        prompt = f"{CAUSAL_SYSTEM_PROMPT}\n\n{build_causal_user_message(query, episode_dicts)}"

        for _ in range(_MAX_CAUSAL_ATTEMPTS):
            response = await self._chat_model.complete(prompt)
            try:
                parsed = json.loads(strip_markdown_fence(response))
                scores = parsed["scores"]
                if not isinstance(scores, list):
                    raise TypeError("'scores' must be a list")
                score_map = _validate_scores(scores, len(episodes))
                return _to_scored_episodes(episodes, score_map, top_k)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        # Exhausted retries: fail safe, not fail loud -- same philosophy as
        # ConsolidateEpisodes._reflect's exhausted-retry branch. A retrieval
        # strategy that can't get a valid LLM response degrades to a flat
        # 0.0 band across every original candidate rather than raising or
        # returning [], since fusion (built separately) needs SOME score for
        # every episode it might dedupe against, not a crash. Original
        # get_by_session order is kept (oldest-first) since a 0.0 tie has no
        # other meaningful ordering signal.
        return [ScoredEpisode(episode=e, score=0.0) for e in episodes[:top_k]]


def _validate_scores(scores: list[Any], episode_count: int) -> dict[int, float]:
    # Mirrors ConsolidateEpisodes._validate_and_dedupe_facts: the outer
    # {"scores": [...]} envelope being valid JSON says nothing about what's
    # INSIDE each element -- an LLM can return a bare number instead of an
    # object, rename episode_index, or hand back score as a word instead of
    # a float. Validating every element here, inside the retry loop, folds
    # a malformed element into the same retry path as malformed JSON rather
    # than letting it crash outside the loop.
    validated: dict[int, float] = {}
    for item in scores:
        if not isinstance(item, dict):
            raise TypeError("each score entry must be a JSON object")
        episode_index = item["episode_index"]
        score = item["score"]
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise TypeError("episode_index must be an int")
        if not (1 <= episode_index <= episode_count):
            raise ValueError("episode_index out of range")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("score must be a number")
        score = float(score)
        if not (0.0 <= score <= 1.0):
            raise ValueError("score out of range")
        # Last-wins on a repeated episode_index within one response, same
        # last-wins precedent as ConsolidateEpisodes's fact_key dedup.
        validated[episode_index] = score
    return validated


def _to_scored_episodes(
    episodes: list[EpisodicMemory], score_map: dict[int, float], top_k: int
) -> list[ScoredEpisode]:
    # Every original candidate ends up with SOME score -- an episode whose
    # 1-based index never appears in a valid response element defaults to
    # 0.0 rather than being dropped, so fusion downstream can dedupe against
    # a complete candidate set.
    scored = [
        ScoredEpisode(episode=e, score=score_map.get(i, 0.0))
        for i, e in enumerate(episodes, start=1)
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]
