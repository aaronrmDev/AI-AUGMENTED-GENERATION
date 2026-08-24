from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker


class SlidingWindowChunker(FixedSizeChunker):
    """Mechanically identical to FixedSizeChunker -- a fixed-size window
    walked across the token stream with overlap. The only difference is the
    default overlap ratio: Fixed Size uses a modest overlap (0.1) as a
    boundary-softening measure; Sliding Window's overlap is the point (0.5),
    trading storage for narrative continuity across chunk boundaries, per
    RAG.md's own framing of the two strategies as a difference of degree and
    intent rather than a different algorithm.
    """

    def __init__(self, chunk_size_tokens: int = 512, overlap_ratio: float = 0.5) -> None:
        super().__init__(chunk_size_tokens, overlap_ratio)
