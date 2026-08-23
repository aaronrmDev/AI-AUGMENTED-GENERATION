import tiktoken


class FixedSizeChunker:
    def __init__(self, chunk_size_tokens: int = 512, overlap_ratio: float = 0.1) -> None:
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
