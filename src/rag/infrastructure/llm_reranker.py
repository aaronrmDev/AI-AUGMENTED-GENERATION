import re

from src.rag.domain.entities import SearchResult
from src.rag.domain.ports import ChatModel, Reranker

_SCORE_PATTERN = re.compile(r"-?\d+")


class LLMReranker(Reranker):
    def __init__(self, chat_model: ChatModel) -> None:
        self._chat_model = chat_model

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        scored = []
        for r in results:
            prompt = (
                f"Query: {query}\n\nCandidate passage:\n{r.content}\n\n"
                f"Score how relevant this passage is to answering the query, "
                f"from 0 (irrelevant) to 10 (directly answers it). "
                f"Respond with ONLY the integer score, nothing else."
            )
            response = await self._chat_model.generate(question=prompt, context="")
            match = _SCORE_PATTERN.search(response)
            score = int(match.group()) if match else 0
            scored.append((r, score))
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return [r for r, _ in ranked[:top_k]]
