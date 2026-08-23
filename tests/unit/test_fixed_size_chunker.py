from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker


def test_short_text_produces_a_single_chunk():
    chunker = FixedSizeChunker(chunk_size_tokens=512, overlap_ratio=0.1)
    chunks = chunker.chunk("This is a short piece of text.")
    assert len(chunks) == 1
    assert chunks[0] == "This is a short piece of text."


def test_long_text_produces_multiple_chunks_with_overlap():
    chunker = FixedSizeChunker(chunk_size_tokens=50, overlap_ratio=0.1)
    # ~500 words is comfortably more than 50 tokens worth of content.
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    # Overlap means the tail of one chunk should reappear near the head of the next.
    first_tail_words = chunks[0].split()[-3:]
    second_text = chunks[1]
    assert any(word in second_text for word in first_tail_words)


def test_empty_text_produces_no_chunks():
    chunker = FixedSizeChunker(chunk_size_tokens=512, overlap_ratio=0.1)
    assert chunker.chunk("") == []


def test_an_overlap_ratio_of_one_or_more_is_rejected():
    import pytest

    # A full-chunk overlap leaves chunk() with a step of zero, so `start` never
    # advances -- the constructor has to refuse it rather than let chunk() hang.
    for ratio in (1.0, 1.5):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size_tokens=512, overlap_ratio=ratio)


def test_every_chunk_is_at_most_the_configured_token_size():
    chunker = FixedSizeChunker(chunk_size_tokens=20, overlap_ratio=0.1)
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunker.chunk(text)
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    for c in chunks:
        assert len(encoding.encode(c)) <= 20
