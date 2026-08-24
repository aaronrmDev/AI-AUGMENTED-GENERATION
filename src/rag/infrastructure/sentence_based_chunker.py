import tiktoken

from src.rag.domain.ports import Chunker
from src.rag.infrastructure._sentence_splitter import split_sentences


class SentenceBasedChunker(Chunker):
    def __init__(self, chunk_size_tokens: int = 512) -> None:
        self._chunk_size = chunk_size_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self._encoding.encode(sentence))
            if current and current_tokens + sentence_tokens > self._chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks
