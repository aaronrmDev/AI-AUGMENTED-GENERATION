import math
import statistics

import tiktoken

from src.rag.domain.ports import Chunker, EmbeddingModel
from src.rag.infrastructure._sentence_splitter import split_sentences

# How far below a document's own mean consecutive-sentence similarity (in
# standard deviations) a similarity must fall before it counts as a real
# topic shift. A pure bottom-quartile rule has no floor: it always finds
# int(N*0.25) "breakpoints" in ANY document, even one where every
# consecutive similarity is identical, because a quantile always has a
# bottom 25% even when nothing in it is actually low -- verified empirically
# against the real corpus, where the quantile rule alone force-split at
# exactly its arithmetic floor with zero content-dependent signal. Gating on
# distance from the document's own mean fixes that: with zero variance
# (stdev == 0), the threshold equals the mean, and a similarity can never be
# strictly less than a constant it's tied with -- so a maximally cohesive
# document now correctly produces zero semantic breakpoints.
_BREAKPOINT_STDEV_MULTIPLIER = 1.0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
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

        # Population stdev, not sample: this is the whole population of
        # consecutive-pair similarities in THIS document, not a sample of a
        # larger one -- and pstdev is defined (as 0.0) for a single value,
        # so no separate len(similarities) == 1 special case is needed here.
        mean = statistics.fmean(similarities)
        stdev = statistics.pstdev(similarities)
        breakpoint_threshold = mean - _BREAKPOINT_STDEV_MULTIPLIER * stdev

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_tokens = len(self._encoding.encode(sentences[0]))

        for i, similarity in enumerate(similarities):
            next_sentence = sentences[i + 1]
            next_tokens = len(self._encoding.encode(next_sentence))
            # Strict "<": with zero variance every similarity equals the
            # threshold exactly, and "<=" would match all of them -- the
            # same force-split bug this threshold exists to prevent. Only a
            # similarity that falls genuinely below typical cohesion counts.
            is_breakpoint = similarity < breakpoint_threshold
            would_overflow = current_tokens + next_tokens > self._chunk_size
            if is_breakpoint or would_overflow:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(next_sentence)
            current_tokens += next_tokens

        chunks.append(" ".join(current))
        return chunks
