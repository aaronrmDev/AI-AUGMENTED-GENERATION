import math

import tiktoken

from src.rag.domain.ports import Chunker, EmbeddingModel
from src.rag.infrastructure._sentence_splitter import split_sentences


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker(Chunker):
    def __init__(self, embedding_model: EmbeddingModel, chunk_size_tokens: int = 512) -> None:
        self._embedder = embedding_model
        self._chunk_size = chunk_size_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        embeddings = [self._embedder.embed(s) for s in sentences]
        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)
        ]

        if len(similarities) == 1:
            return [" ".join(sentences)]

        sorted_similarities = sorted(similarities)
        percentile_index = max(0, int(len(sorted_similarities) * 0.25) - 1)
        breakpoint_threshold = sorted_similarities[percentile_index]

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_tokens = len(self._encoding.encode(sentences[0]))

        for i, similarity in enumerate(similarities):
            next_sentence = sentences[i + 1]
            next_tokens = len(self._encoding.encode(next_sentence))
            is_breakpoint = similarity <= breakpoint_threshold
            would_overflow = current_tokens + next_tokens > self._chunk_size
            if current and (is_breakpoint or would_overflow):
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(next_sentence)
            current_tokens += next_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks
