from src.rag.infrastructure.semantic_chunker import SemanticChunker


class _StubEmbedder:
    """Returns an identical vector for every input -- every consecutive
    cosine similarity is exactly 1.0, the strongest possible signal that
    there is NO topic shift anywhere. Used to test the breakpoint
    arithmetic in isolation from any real semantic model."""

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def test_uniformly_similar_sentences_produce_a_single_chunk():
    # Regression test for the force-split bug: a pure bottom-quartile rule
    # always finds int(N*0.25) "breakpoints" even when every similarity is
    # identical, because a quantile always has a bottom 25% even when
    # nothing in it is actually low. With zero real topic variation, this
    # chunker must decline to split at all.
    chunker = SemanticChunker(_StubEmbedder(), chunk_size_tokens=512)
    text = (
        "First sentence here. Second sentence here. Third sentence here. "
        "Fourth sentence here. Fifth sentence here. Sixth sentence here. "
        "Seventh sentence here. Eighth sentence here. Ninth sentence here."
    )
    chunks = chunker.chunk(text)
    assert len(chunks) == 1


def test_two_uniformly_similar_sentences_produce_a_single_chunk():
    chunker = SemanticChunker(_StubEmbedder(), chunk_size_tokens=512)
    chunks = chunker.chunk("First sentence here. Second sentence here.")
    assert len(chunks) == 1
