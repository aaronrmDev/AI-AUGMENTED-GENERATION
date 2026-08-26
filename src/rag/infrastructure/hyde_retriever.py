import uuid

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import ChatModel, Retriever

_PROMPT_TEMPLATE = (
    "Write a short, confident, hypothetical answer to the following question, "
    "as if you already knew the answer with certainty. It's fine if the "
    "specific details you invent aren't accurate -- the goal is realistic "
    "phrasing and vocabulary, not factual correctness. Respond with ONLY the "
    "hypothetical answer, no preamble.\n\nQuestion: {query}"
)


class HyDERetriever(Retriever):
    def __init__(self, inner: Retriever, chat_model: ChatModel) -> None:
        self._inner = inner
        self._chat_model = chat_model

    async def execute(self, tenant_id: uuid.UUID, query: str, top_k: int) -> list[SearchResult]:
        prompt = _PROMPT_TEMPLATE.format(query=query)
        hypothetical_answer = await self._chat_model.generate(question=prompt, context="")
        return await self._inner.execute(
            tenant_id=tenant_id, query=hypothetical_answer, top_k=top_k
        )
