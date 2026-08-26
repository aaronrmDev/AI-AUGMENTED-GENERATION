import re
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

# Whole-word match, unbounded (not a fixed-length prefix): the prior
# `"no" not in response.strip().lower()[:10]` both false-negatived on a
# compliant "NO" appearing after character 10 (e.g. "Based on my knowledge,
# NO") and false-positived on "no" appearing as a substring of an unrelated
# word inside the first 10 characters (e.g. "Unknown"). \b(yes|no)\b takes
# the first standalone yes/no token in the response, which is what the gate
# prompt actually asks the model to produce. A response containing neither
# word (or negating one, e.g. "NOT needed" -- rare given the prompt's "ONLY
# YES or NO" instruction) defaults to needs_retrieval=True: retrieving when
# it wasn't strictly necessary costs latency, but skipping a retrieval that
# was needed risks a hallucinated answer, so the safe default is to retrieve.
_GATE_ANSWER_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


class SelfRAGAnswerQuestion:
    def __init__(self, search_documents: Retriever, chat_model: ChatModel, top_k: int) -> None:
        self._search = search_documents
        self._chat_model = chat_model
        self._top_k = top_k

    async def execute(self, tenant_id: uuid.UUID, question: str) -> ChatAnswer:
        gate_prompt = _GATE_PROMPT_TEMPLATE.format(question=question)
        # complete(), not generate(): see HyDERetriever.execute's comment --
        # this is a classification prompt, not a "use only the provided
        # context" question, and generate()'s RAG-answering system prompt is
        # the wrong instruction for it.
        gate_response = await self._chat_model.complete(gate_prompt)
        match = _GATE_ANSWER_PATTERN.search(gate_response)
        needs_retrieval = True if match is None else match.group(1).lower() == "yes"

        if not needs_retrieval:
            # complete(), not generate(): answering directly from the
            # model's own knowledge is exactly the case generate()'s system
            # prompt refuses ("if the context doesn't contain the answer,
            # say so") when context="" -- every live NO-gate answer degraded
            # into a refusal under generate() until this was caught by this
            # batch's final review.
            answer = await self._chat_model.complete(question)
            return ChatAnswer(answer=answer, sources=[])

        sources = await self._search.execute(tenant_id=tenant_id, query=question, top_k=self._top_k)
        context = "\n\n".join(source.content for source in sources)
        answer = await self._chat_model.generate(question=question, context=context)
        return ChatAnswer(answer=answer, sources=sources)
