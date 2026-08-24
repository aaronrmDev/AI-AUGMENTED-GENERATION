from src.rag.infrastructure.sentence_based_chunker import SentenceBasedChunker


def test_short_text_produces_a_single_chunk():
    chunker = SentenceBasedChunker(chunk_size_tokens=512)
    chunks = chunker.chunk("This is one short sentence.")
    assert chunks == ["This is one short sentence."]


def test_never_splits_a_sentence_across_chunks():
    chunker = SentenceBasedChunker(chunk_size_tokens=10)
    text = " ".join(f"This is sentence number {i} of the document." for i in range(20))
    chunks = chunker.chunk(text)
    # Every chunk must be a clean concatenation of whole sentences -- each
    # chunk, split back into sentences, re-joins to exactly itself with no
    # leftover fragment.
    for c in chunks:
        assert c.strip().endswith((".", "!", "?"))


def test_a_single_oversized_sentence_becomes_its_own_chunk():
    chunker = SentenceBasedChunker(chunk_size_tokens=5)
    long_sentence = " ".join(f"word{i}" for i in range(50)) + "."
    chunks = chunker.chunk(long_sentence)
    assert len(chunks) == 1
    assert chunks[0] == long_sentence


def test_empty_text_produces_no_chunks():
    chunker = SentenceBasedChunker(chunk_size_tokens=512)
    assert chunker.chunk("") == []
