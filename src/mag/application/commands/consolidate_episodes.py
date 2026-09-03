import json
import uuid
from typing import Any

from src.mag.application.commands.record_semantic_fact import RecordSemanticFact
from src.mag.domain.entities import EpisodicMemory, SemanticMemory
from src.mag.domain.ports import (
    EpisodicMemoryRepository,
    MemoryGraphRepository,
    SemanticMemoryIndex,
    SemanticMemoryRepository,
)
from src.mag.infrastructure._consolidation_prompt import CONSOLIDATION_SYSTEM_PROMPT
from src.mag.infrastructure._graph_write_safety import best_effort_graph_write
from src.mag.infrastructure._llm_json import format_episodes_for_reflection, strip_markdown_fence
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
        memory_graph_repository: MemoryGraphRepository,
    ) -> None:
        self._episodes = episodic_memory_repository
        self._facts = semantic_memory_repository
        self._record_fact = RecordSemanticFact(
            semantic_memory_repository=semantic_memory_repository,
            semantic_memory_index=semantic_memory_index,
            embedding_model=embedding_model,
            memory_graph_repository=memory_graph_repository,
        )
        self._chat_model = chat_model
        self._graph = memory_graph_repository

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
            # A consolidation-derived fact_key can collide with one a
            # user or agent already Invalidated/Archived via the memory
            # evolution operations (fact_key is unconstrained LLM free
            # text over the same per-user namespace those operations
            # manage, not a source-partitioned one) -- without carrying
            # the existing status through, this write would silently
            # un-archive/un-invalidate it, the same bug class MAG Batch
            # F's own review caught and fixed for UpdateMemory/
            # RefineMemory. find_by_key returns None for a genuinely new
            # fact_key, in which case there's nothing to preserve.
            existing = await self._facts.find_by_key(user_id, fact["fact_key"], tenant_id)
            recorded = await self._record_fact.execute(
                tenant_id=tenant_id,
                user_id=user_id,
                fact_key=fact["fact_key"],
                fact_value=fact["fact_value"],
                confidence=float(fact.get("confidence", 1.0)),
                source="consolidation",
                valid_until=existing.valid_until if existing else None,
                archived_at=existing.archived_at if existing else None,
            )
            written.append(recorded)
            # ABSTRACTS_TO from every episode reflected on to this fact --
            # the reflection was one LLM call over the whole batch, not a
            # 1:1 episode:fact mapping, so there's no finer-grained "which
            # specific episode produced this fact" signal to link instead.
            # This IS the graph's representation of consolidation
            # (DATABASE.md's own description of what this edge is for).
            for episode in episodes:
                await best_effort_graph_write(
                    self._graph.link_abstracts_to(episode.id, recorded.id, tenant_id),
                    "link abstracts_to",
                )

        # Marked regardless of whether reflection produced any facts --
        # see the design spec's Consolidation section and
        # EpisodicMemoryRepository.mark_consolidated's docstring for why a
        # fact-free episode still shouldn't be re-sent to the LLM forever.
        await self._episodes.mark_consolidated([e.id for e in episodes], tenant_id)
        return written

    async def _reflect(self, episodes: list[EpisodicMemory]) -> list[dict[str, Any]]:
        prompt = (
            f"{CONSOLIDATION_SYSTEM_PROMPT}\n\n"
            f"{format_episodes_for_reflection(episodes)}"
        )
        for _ in range(_MAX_REFLECTION_ATTEMPTS):
            response = await self._chat_model.complete(prompt)
            try:
                parsed = json.loads(strip_markdown_fence(response))
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
        # Stripped before use, not just before the emptiness check -- "lang"
        # and "lang " would otherwise dedupe as distinct keys here and later
        # persist as two distinct rows (RecordSemanticFact's uuid5 id is
        # derived from the exact string), defeating the dedup this function
        # exists to do.
        fact_key = fact_key.strip()
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
