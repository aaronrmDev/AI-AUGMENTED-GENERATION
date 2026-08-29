import asyncio
import math
import uuid
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any

from src.mag.application.queries.retrieve_by_causal_relevance import CausalRetrieval
from src.mag.application.queries.retrieve_by_entity import EntityRetrieval
from src.mag.application.queries.retrieve_by_salience import SalienceRetrieval
from src.mag.application.queries.retrieve_by_semantic_similarity import (
    SemanticSimilarityRetrieval,
)
from src.mag.application.queries.retrieve_by_temporal_window import TemporalRetrieval
from src.mag.domain.entities import EpisodicMemory, ScoredEpisode
from src.mag.domain.ports import EpisodicMemoryIndex, EpisodicMemoryRepository
from src.rag.domain.ports import ChatModel

# Fetched per strategy before fusion narrows to top_k -- wide enough that
# fusion (which reranks by a combined score, not any one strategy's own
# ranking) has real candidates to work with instead of being handed an
# already-truncated top_k from each strategy and reranking within that.
_FETCH_MULTIPLIER = 3
_MIN_FETCH_K = 10


class RecencyDecayFusionRetrieval:
    # Per #75's own text, this "combin[es] the outputs of the other
    # strategies rather than acting as an independent one" -- so it's built
    # as an orchestrator over the other five, not a sixth data-access path.
    def __init__(
        self,
        episodic_memory_repository: EpisodicMemoryRepository,
        episodic_memory_index: EpisodicMemoryIndex,
        chat_model: ChatModel,
    ) -> None:
        self._temporal = TemporalRetrieval(episodic_memory_repository)
        self._salience = SalienceRetrieval(episodic_memory_repository)
        self._entity = EntityRetrieval(episodic_memory_repository)
        self._semantic = SemanticSimilarityRetrieval(episodic_memory_index)
        self._causal = CausalRetrieval(episodic_memory_repository, chat_model)

    async def execute(
        self,
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        top_k: int,
        query_embedding: list[float] | None = None,
        causal_query: str | None = None,
        entity: str | None = None,
        within: tuple[datetime, datetime] | None = None,
        weights: dict[str, float] | None = None,
        decay_half_life_hours: float = 24.0,
        now: datetime | None = None,
    ) -> list[ScoredEpisode]:
        now = now or datetime.now(UTC)
        fetch_k = max(top_k * _FETCH_MULTIPLIER, _MIN_FETCH_K)

        # Temporal and salience only need session_id, so they always run.
        # Semantic/causal/entity run only when the caller supplied enough to
        # run them -- explicit parameters, no query auto-decomposition (that
        # belongs to the orchestration layer described in OVERVIEW.md, not
        # here). Fusion runs whatever it was given enough to run, silently
        # skipping the rest, rather than guessing.
        coroutines: dict[str, Coroutine[Any, Any, list[ScoredEpisode]]] = {
            "temporal": self._temporal.execute(
                tenant_id=tenant_id, session_id=session_id, top_k=fetch_k, within=within
            ),
            "salience": self._salience.execute(
                tenant_id=tenant_id, session_id=session_id, top_k=fetch_k
            ),
        }
        if query_embedding is not None:
            coroutines["semantic"] = self._semantic.execute(
                tenant_id=tenant_id, query_embedding=query_embedding, top_k=fetch_k
            )
        if causal_query is not None:
            coroutines["causal"] = self._causal.execute(
                tenant_id=tenant_id, session_id=session_id, query=causal_query, top_k=fetch_k
            )
        if entity is not None:
            coroutines["entity"] = self._entity.execute(
                tenant_id=tenant_id, session_id=session_id, entity=entity, top_k=fetch_k
            )

        names = list(coroutines.keys())
        results = await asyncio.gather(*(coroutines[name] for name in names))
        per_strategy: dict[str, list[ScoredEpisode]] = dict(zip(names, results, strict=True))

        # Equal weighting unless the caller overrides -- a name the caller
        # weights but that isn't in `names` (e.g. weighting "causal" without
        # supplying causal_query) is silently inert, same as an included
        # strategy the caller's weights dict omits contributing 0.0: weights
        # is authoritative and complete once given, not merged with the
        # equal-weight default.
        strategy_weight = weights if weights is not None else {n: 1.0 / len(names) for n in names}

        fused_scores: dict[uuid.UUID, float] = {}
        episodes_by_id: dict[uuid.UUID, EpisodicMemory] = {}
        for name, scored_list in per_strategy.items():
            weight = strategy_weight.get(name, 0.0)
            for scored, normalized in zip(
                scored_list, _min_max_normalize(scored_list), strict=True
            ):
                episode = scored.episode
                episodes_by_id[episode.id] = episode
                decayed = normalized * _recency_decay(episode.timestamp, now, decay_half_life_hours)
                # Summed, not averaged: an episode multiple strategies agree
                # on accumulates more than one strategy's weighted
                # contribution -- agreement across strategies is itself a
                # relevance signal fusion should reward, not dilute. This is
                # also what dedupes overlapping hits (#12's requirement) --
                # one fused_scores entry per episode id regardless of how
                # many strategies surfaced it.
                fused_scores[episode.id] = fused_scores.get(episode.id, 0.0) + weight * decayed

        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            ScoredEpisode(episode=episodes_by_id[episode_id], score=score)
            for episode_id, score in ranked[:top_k]
        ]


def _min_max_normalize(scored: list[ScoredEpisode]) -> list[float]:
    if not scored:
        return []
    scores = [s.score for s in scored]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        # A flat distribution (including CausalRetrieval's all-0.0 exhausted-
        # retry floor) means every candidate was equally (ir)relevant by
        # this strategy's own measure -- normalizing to 1.0 keeps them in
        # fusion's running rather than making a strategy that couldn't
        # differentiate its own candidates zero out all of them.
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _recency_decay(timestamp: datetime, now: datetime, half_life_hours: float) -> float:
    # Standard half-life decay, applied uniformly to every candidate from
    # every strategy -- decay is about the episode's age, not about which
    # strategy found it. Clamped at 0: `now` is caller-supplied (for
    # testability against a fixed reference time, not the real wall clock),
    # so a timestamp after `now` is possible in a test fixture and must not
    # produce a decay factor above 1.0.
    age_hours = max((now - timestamp).total_seconds() / 3600.0, 0.0)
    return math.exp(-math.log(2) * age_hours / half_life_hours)
