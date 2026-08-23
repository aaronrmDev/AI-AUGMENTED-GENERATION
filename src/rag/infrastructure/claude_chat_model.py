from src.rag.domain.ports import ChatModel

_SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. "
    "If the context doesn't contain the answer, say so plainly rather than guessing."
)


class ClaudeChatModel(ChatModel):
    def __init__(self, client, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    async def generate(self, question: str, context: str) -> str:
        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        return response.content[0].text
