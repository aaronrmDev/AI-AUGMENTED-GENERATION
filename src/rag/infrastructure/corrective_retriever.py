import asyncio
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

        # Evaluated concurrently, not sequentially: each check is an
        # independent LLM round-trip with no ordering dependency, so
        # asyncio.gather turns top_k serial round-trips into one round-trip
        # of wall time. This was flagged by this batch's final review as the
        # bulk of the treatment's measured latency overhead.
        relevance_flags = await asyncio.gather(
            *[self._is_relevant(query, r) for r in results]
        )
        relevant = [
            r for r, is_relevant in zip(results, relevance_flags, strict=True) if is_relevant
        ]
        # Decision rule: return whatever passed relevance review whenever
        # ANYTHING passed -- correct only when the set as a whole fails
        # (zero results pass). This batch's final review caught the original
        # ">half must pass" threshold as a design defect, not just a stricter
        # variant: in the ordinary "answer lives in exactly one of top_k
        # retrieved chunks" case, a single correct match can never be a
        # majority, so the threshold discarded a correctly-identified answer
        # chunk and replaced it with an unvalidated re-search on 5 of 7
        # measured questions. RAG.md's own text supports "any pass" as the
        # more faithful reading too: "documents that pass get used, and if
        # the set as a whole fails, CRAG triggers a correction" describes
        # zero passing as failure, not less-than-a-majority passing.
        if relevant:
            return relevant

        # Correction: nothing in the initial results passed relevance review.
        # Try one alternative phrasing of the query and re-search --
        # single-shot, no retry loop, so latency stays bounded and behavior
        # stays testable. Returns the corrected search's results directly
        # (RAG.md's "trying an alternative search"), not merged with whatever
        # passed the first pass (there was nothing to merge -- by definition,
        # nothing passed).
        refined_query = (
            await self._chat_model.complete(_REFINE_PROMPT_TEMPLATE.format(query=query))
        ).strip()
        # Guard against a degenerate completion (empty, or the model ignoring
        # "respond with ONLY the alternative query" and returning nothing
        # usable): fall back to the original query rather than re-searching
        # with an empty or malformed string.
        if not refined_query:
            refined_query = query
        return await self._inner.execute(tenant_id=tenant_id, query=refined_query, top_k=top_k)

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
