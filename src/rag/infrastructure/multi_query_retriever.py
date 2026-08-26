import asyncio
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import ChatModel, Retriever
from src.rag.infrastructure._result_fusion import reciprocal_rank_fusion

_PROMPT_TEMPLATE = (
    "Generate {n} genuinely different phrasings of the following question, "
    "each viewing it from a different angle (not just reworded synonyms of "
    "each other). Respond with ONLY the {n} phrasings, one per line, no "
    "numbering, no extra commentary.\n\nQuestion: {query}"
)


class MultiQueryRetriever(Retriever):
    def __init__(self, inner: Retriever, chat_model: ChatModel, num_queries: int = 4) -> None:
        self._inner = inner
        self._chat_model = chat_model
        self._num_queries = num_queries

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        prompt = _PROMPT_TEMPLATE.format(n=self._num_queries, query=query)
        # complete(), not generate(): this is a rephrasing task, not a
        # "use only the provided context" question -- generate()'s
        # RAG-answering system prompt is the wrong instruction for it (see
        # HyDERetriever.execute's comment for the sibling bug this avoided).
        response = await self._chat_model.complete(prompt)
        variants = [line.strip() for line in response.splitlines() if line.strip()]
        if not variants:
            variants = [query]

        result_lists = await asyncio.gather(
            *[self._inner.execute(tenant_id=tenant_id, query=v, top_k=top_k) for v in variants]
        )
        return reciprocal_rank_fusion(list(result_lists), top_k=top_k)
