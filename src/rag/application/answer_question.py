import uuid

from src.rag.domain.entities import ChatAnswer
from src.rag.domain.ports import ChatModel, Retriever


class AnswerQuestion:
    def __init__(
        self, search_documents: Retriever, chat_model: ChatModel, top_k: int
    ) -> None:
        self._search = search_documents
        self._chat_model = chat_model
        self._top_k = top_k

    async def execute(self, tenant_id: uuid.UUID, question: str) -> ChatAnswer:
        sources = await self._search.execute(tenant_id=tenant_id, query=question, top_k=self._top_k)
        context = "\n\n".join(source.content for source in sources)
        answer = await self._chat_model.generate(question=question, context=context)
        return ChatAnswer(answer=answer, sources=sources)
