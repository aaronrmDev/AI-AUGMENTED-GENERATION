from typing import Any

from src.rag.domain.ports import ChatModel

_SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. "
    "If the context doesn't contain the answer, say so plainly rather than guessing."
)


class OllamaChatModel(ChatModel):
    def __init__(self, client: Any, model_id: str) -> None:
        # Typed as Any rather than ollama.AsyncClient for the same reason as
        # ClaudeChatModel: stays substitutable by the unit tests' fake, and
        # importing this module costs nothing when no chat call is made.
        self._client = client
        self._model_id = model_id
        # Populated by the most recent generate()/complete() call (#147) --
        # ollama's ChatResponse carries these as top-level fields (verified
        # against the installed ollama client's real ChatResponse.model_fields,
        # not assumed), but ChatModel.generate()'s own return type is a plain
        # str shared by every RAG technique in this codebase, so there's no
        # return-contract change available to carry them out. A caller that
        # wants real token counts for one specific call -- this project's own
        # evaluation harness -- reads these two attributes off the same
        # OllamaChatModel instance immediately after that call, instead of
        # the harness hardcoding 0 for both, which is what every runner did
        # before this existed.
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    async def generate(self, question: str, context: str) -> str:
        response = await self._client.chat(
            model=self._model_id,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        self._record_token_counts(response)
        # Unlike Anthropic's content-block list, ollama's Message keeps
        # `thinking` and `content` as separate fields (verified against the
        # installed ollama==0.6.2 client's actual Message model) -- content
        # is always the plain answer, even for a reasoning-capable model, so
        # there's no block-type scan needed here.
        return response.message.content or ""

    def _record_token_counts(self, response: Any) -> None:
        self.last_input_tokens = getattr(response, "prompt_eval_count", 0) or 0
        self.last_output_tokens = getattr(response, "eval_count", 0) or 0

    async def complete(self, prompt: str) -> str:
        # No system prompt: generate()'s _SYSTEM_PROMPT ("...say so plainly
        # rather than guessing") is a RAG-answering instruction that actively
        # fights any caller that wants a direct completion of an arbitrary
        # prompt with no document context -- HyDE's "invent a hypothetical
        # answer", Self-RAG's retrieval-gate check and its own no-context
        # "answer from your own knowledge" branch, and Multi-Query's "generate
        # query variants" all need the model to actually do the thing asked,
        # not refuse for lack of context. A real bug from exactly this
        # conflict (HyDE and Self-RAG's NO branch both silently degraded into
        # refusal generators under generate()) was caught by this batch's
        # final review.
        response = await self._client.chat(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        self._record_token_counts(response)
        return response.message.content or ""
