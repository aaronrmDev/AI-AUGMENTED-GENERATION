import tiktoken

from src.rag.domain.ports import Chunker


class FixedSizeChunker(Chunker):
    def __init__(self, chunk_size_tokens: int = 512, overlap_ratio: float = 0.1) -> None:
        # An overlap of 100% or more makes `step` (chunk_size - overlap) zero or
        # negative in chunk(), so `start` never advances and the loop runs
        # forever, appending chunks until the process runs out of memory.
        # Rejecting the ratio at construction turns a hang into an error.
        if overlap_ratio >= 1.0:
            raise ValueError("overlap_ratio must be < 1.0")
        self._chunk_size = chunk_size_tokens
        self._overlap = int(chunk_size_tokens * overlap_ratio)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        tokens = self._encoding.encode(text)
        if len(tokens) <= self._chunk_size:
            return [text]

        chunks: list[str] = []
        step = self._chunk_size - self._overlap
        start = 0
        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            chunks.append(self._encoding.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += step
        return chunks
