import re
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import ChatModel, Retriever

_RELEVANCE_PROMPT_TEMPLATE = (
    "Does the following passage directly help answer the query? Respond with "
    "ONLY YES or NO, nothing else.\n\nQuery: {query}\n\nPassage: {passage}"
)

_REFINE_PROMPT_TEMPLATE = (
    "The following search query did not retrieve enough relevant results. Write "
    "one alternative phrasing of the query that might retrieve better results -- "
    "same underlying information need, different wording or angle. Respond with "
    "ONLY the alternative query, nothing else.\n\nOriginal query: {query}"
)

# Same unbounded, whole-word match Batch C's final review established for
# SelfRAGAnswerQuestion's gate parsing, for the identical reason: a fixed
# prefix window both misses a compliant answer arriving late in the response
# and misreads "no" as a substring of an unrelated word. Unlike the gate's
# safe default (missing a needed retrieval risks a worse failure than an
# unnecessary one, so it defaults to YES/retrieve), CRAG's relevance check
# defaults the other way: CRAG exists specifically to keep untrustworthy
# content out, so an ambiguous or unparseable judgment defaults to NOT
# relevant.
_RELEVANCE_ANSWER_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


class CorrectiveRetriever(Retriever):
    def __init__(self, inner: Retriever, chat_model: ChatModel) -> None:
        self._inner = inner
        self._chat_model = chat_model

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        results = await self._inner.execute(tenant_id=tenant_id, query=query, top_k=top_k)
        if not results:
            return results

        relevant = [r for r in results if await self._is_relevant(query, r)]
        if len(relevant) > len(results) / 2:
            return relevant

        # Correction: a strict majority of the initial results failed relevance
        # review. Try one alternative phrasing of the query and re-search --
        # single-shot, no retry loop, so latency stays bounded and behavior
        # stays testable. Returns the corrected search's results directly
        # (RAG.md's "trying an alternative search"), not merged with whatever
        # passed the first pass.
        refined_query = await self._chat_model.complete(
            _REFINE_PROMPT_TEMPLATE.format(query=query)
        )
        return await self._inner.execute(
            tenant_id=tenant_id, query=refined_query.strip(), top_k=top_k
        )

    async def _is_relevant(self, query: str, result: SearchResult) -> bool:
        prompt = _RELEVANCE_PROMPT_TEMPLATE.format(query=query, passage=result.content)
        # complete(), not generate(): a relevance-classification prompt, not a
        # "use only the provided context" QA task -- see this module's
        # docstring-equivalent comment above and the design spec for the bug
        # this avoids (Batch C's final review found generate()'s hardcoded
        # RAG-answering system prompt silently breaks non-QA prompts routed
        # through it).
        response = await self._chat_model.complete(prompt)
        match = _RELEVANCE_ANSWER_PATTERN.search(response)
        return match is not None and match.group(1).lower() == "yes"
