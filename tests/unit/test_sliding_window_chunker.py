from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker
from src.rag.infrastructure.sliding_window_chunker import SlidingWindowChunker


def test_is_a_fixed_size_chunker_subclass():
    chunker = SlidingWindowChunker()
    assert isinstance(chunker, FixedSizeChunker)


def test_default_overlap_is_higher_than_fixed_size_chunkers_default():
    default_chunker = FixedSizeChunker()
    sliding_chunker = SlidingWindowChunker()
    # Both use the same 512-token default chunk size; sliding window's
    # overlap (0.5) must exceed fixed size's own default (0.1) -- verified
    # indirectly, since overlap isn't a public attribute: a long text
    # produces more chunks under heavier overlap for the same chunk size,
    # since each chunk advances by a smaller step.
    text = " ".join(f"word{i}" for i in range(2000))
    assert len(sliding_chunker.chunk(text)) > len(default_chunker.chunk(text))


def test_short_text_produces_a_single_chunk():
    chunker = SlidingWindowChunker()
    chunks = chunker.chunk("A short piece of text.")
    assert chunks == ["A short piece of text."]


def test_empty_text_produces_no_chunks():
    chunker = SlidingWindowChunker()
    assert chunker.chunk("") == []
