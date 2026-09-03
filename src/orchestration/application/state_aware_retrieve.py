import uuid
from dataclasses import replace

from src.mag.application.commands.capture_episode import CaptureEpisode
from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
from src.mag.application.queries.retrieve_working_memory import RetrieveWorkingMemory
from src.mag.domain.entities import ScoredFact, WorkingMemoryTurn
from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import ChatModel, EmbeddingModel, Retriever

_REWRITE_PROMPT_TEMPLATE = (
    "You are rewriting a user's search query to make it more specific, "
    "using what is known about this user. Rewrite the query below into a "
    "single, more specific search query that incorporates the user's known "
    "preferences and context. If nothing is known about the user, return "
    "the original query unchanged. Respond with ONLY the rewritten query, "
    "nothing else -- no explanation, no quotes.\n\n"
    "Known facts about this user:\n{facts_text}\n\n"
    "Recent conversation:\n{turns_text}\n\n"
    "Original query: {query}\n\n"
    "Rewritten query:"
)
_RANKING_BOOST_WEIGHT = 0.2


def _facts_text(facts: list[ScoredFact]) -> str:
    if not facts:
        return "(none)"
    return "\n".join(f"- {f.fact.fact_key}: {f.fact.fact_value}" for f in facts)


def _turns_text(turns: list[WorkingMemoryTurn]) -> str:
    if not turns:
        return "(none)"
    return "\n".join(f"{t.role}: {t.content}" for t in turns)


def _words(text: str) -> set[str]:
    # Strips leading/trailing punctuation per token so "visualization,"
    # (a fact ending a clause) and "visualization." (a document ending a
    # sentence) count as the same word -- confirmed live that leaving
    # punctuation attached silently dropped real, intended overlap matches.
    return {word.strip(".,;:!?()\"'") for word in text.lower().split()} - {""}


def _keyword_overlap_ratio(content: str, facts: list[ScoredFact]) -> float:
    fact_words = {word for f in facts for word in _words(f.fact.fact_value)}
    if not fact_words:
        return 0.0
    return len(_words(content) & fact_words) / len(fact_words)


class StateAwareRetrieve:
    """State-Aware RAG: reads MAG's state before retrieval runs, rewrites
    the query to be personal instead of generic, boosts ranking for
    results matching the user's known context, and writes what came back
    back into MAG so the next turn has it too. "MAG makes RAG personal,
    and RAG in turn gives MAG something external to be personal about"
    (OVERVIEW.md's own framing).

    Deliberately does NOT implement RAG's Retriever ABC -- disclosed, not
    silent: Retriever.execute(tenant_id, query, top_k) has no user_id/
    session_id slot, and MAG's entire schema requires both, so forcing
    this into the existing port would mean fabricating a user identity or
    breaking the ABC's contract for every other implementer.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        find_semantic_facts: FindSemanticFacts,
        retrieve_working_memory: RetrieveWorkingMemory,
        chat_model: ChatModel,
        fallback_retriever: Retriever,
        capture_episode: CaptureEpisode,
        facts_top_k: int = 5,
        recent_turns_limit: int = 5,
        candidate_k: int = 20,
    ) -> None:
        self._embedder = embedding_model
        self._find_semantic_facts = find_semantic_facts
        self._retrieve_working_memory = retrieve_working_memory
        self._chat_model = chat_model
        self._fallback = fallback_retriever
        self._capture_episode = capture_episode
        self._facts_top_k = facts_top_k
        self._recent_turns_limit = recent_turns_limit
        # Same over-fetch-then-rerank shape as RerankingRetriever's own
        # candidate_k (src/rag/infrastructure/reranking_retriever.py) --
        # asking the fallback for only `top_k` results BEFORE boosting
        # would let embedding-only ranking permanently exclude a document
        # the boost step could otherwise have promoted into the caller's
        # actual top_k. Confirmed live: without this, a real top_k=1 call
        # returned whichever document the raw embedding favored, with the
        # boost step never getting a chance to act on anything else.
        self._candidate_k = candidate_k

    async def execute(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        query_embedding = self._embedder.embed(query)
        facts = await self._find_semantic_facts.by_similarity(
            query_embedding, user_id, tenant_id, self._facts_top_k
        )
        recent_turns = await self._retrieve_working_memory.execute(
            session_id, self._recent_turns_limit
        )

        enriched_query = await self._rewrite_query(query, facts, recent_turns)

        raw_results = await self._fallback.execute(
            tenant_id, enriched_query, max(top_k, self._candidate_k)
        )
        boosted_results = self._apply_ranking_boost(raw_results, facts)[:top_k]

        await self._capture_episode.execute(
            tenant_id,
            user_id,
            session_id,
            content={
                "type": "state_aware_retrieval",
                "query": query,
                "enriched_query": enriched_query,
                "top_result_content": boosted_results[0].content if boosted_results else None,
            },
        )

        return boosted_results

    async def _rewrite_query(
        self, query: str, facts: list[ScoredFact], recent_turns: list[WorkingMemoryTurn]
    ) -> str:
        prompt = _REWRITE_PROMPT_TEMPLATE.format(
            facts_text=_facts_text(facts), turns_text=_turns_text(recent_turns), query=query
        )
        rewritten = (await self._chat_model.complete(prompt)).strip()
        # A real model can still wrap its answer despite the "respond with
        # ONLY" instruction (spike-confirmed clean against qwen3.5, but
        # not guaranteed for every model/prompt combination) -- falling
        # back to the raw query on an empty response keeps this from ever
        # sending a blank string to retrieval.
        first_line = rewritten.splitlines()[0].strip() if rewritten else ""
        return first_line or query

    def _apply_ranking_boost(
        self, results: list[SearchResult], facts: list[ScoredFact]
    ) -> list[SearchResult]:
        boosted = [
            replace(
                r, score=r.score + _RANKING_BOOST_WEIGHT * _keyword_overlap_ratio(r.content, facts)
            )
            for r in results
        ]
        return sorted(boosted, key=lambda r: r.score, reverse=True)
