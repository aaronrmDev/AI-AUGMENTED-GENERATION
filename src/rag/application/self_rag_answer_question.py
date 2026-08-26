import uuid

from src.rag.domain.entities import ChatAnswer
from src.rag.domain.ports import ChatModel, Retriever

_GATE_PROMPT_TEMPLATE = (
    "Does answering the following question require looking up external, "
    "private, or recent information that you would not already know from "
    "general training -- or can it be answered correctly from general "
    "knowledge alone (common facts, basic arithmetic, well-known concepts)? "
    "Respond with ONLY YES (retrieval needed) or NO (no retrieval needed), "
    "nothing else.\n\nQuestion: {question}"
)


class SelfRAGAnswerQuestion:
    def __init__(self, search_documents: Retriever, chat_model: ChatModel, top_k: int) -> None:
        self._search = search_documents
        self._chat_model = chat_model
        self._top_k = top_k

    async def execute(self, tenant_id: uuid.UUID, question: str) -> ChatAnswer:
        gate_prompt = _GATE_PROMPT_TEMPLATE.format(question=question)
        gate_response = await self._chat_model.generate(question=gate_prompt, context="")
        # Tolerant parse, same convention as LLMReranker's score parsing:
        # check the first ~10 characters for "no" before "yes", since a
        # response starting "NO, this is..." should never be read as
        # containing "yes" from somewhere later in the sentence.
        needs_retrieval = "no" not in gate_response.strip().lower()[:10]

        if not needs_retrieval:
            answer = await self._chat_model.generate(question=question, context="")
            return ChatAnswer(answer=answer, sources=[])

        sources = await self._search.execute(tenant_id=tenant_id, query=question, top_k=self._top_k)
        context = "\n\n".join(source.content for source in sources)
        answer = await self._chat_model.generate(question=question, context=context)
        return ChatAnswer(answer=answer, sources=sources)
