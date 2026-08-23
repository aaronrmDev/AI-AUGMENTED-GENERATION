from typing import Any, cast

from src.rag.domain.ports import ChatModel

_SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. "
    "If the context doesn't contain the answer, say so plainly rather than guessing."
)


class ClaudeChatModel(ChatModel):
    def __init__(self, client: Any, model_id: str) -> None:
        # Typed as Any rather than anthropic.AsyncAnthropic so this adapter
        # stays substitutable by the duck-typed fake the unit tests drive it
        # with, and so importing it costs nothing when no chat call is made.
        self._client = client
        self._model_id = model_id

    async def generate(self, question: str, context: str) -> str:
        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        # response.content[0] is not reliably the answer. The configured model
        # (claude-opus-5) runs adaptive thinking by default when the `thinking`
        # parameter is omitted, and thinking blocks lead the content list --
        # indexing [0].text raises AttributeError on a ThinkingBlock. Scan for
        # the first text block instead of assuming a position.
        for block in response.content:
            if block.type == "text":
                return cast(str, block.text)
        return ""
