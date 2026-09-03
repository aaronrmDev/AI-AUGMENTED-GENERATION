from src.rag.infrastructure.semantic_chunker import SemanticChunker


def test_groups_topically_similar_sentences_together(embedding_model):
    chunker = SemanticChunker(embedding_model, chunk_size_tokens=512)
    text = (
        "Cats are small domesticated carnivorous mammals. "
        "Many people keep cats as pets in their homes. "
        "Cats are known for their independence and agility. "
        "Quarterly revenue increased by twelve percent this year. "
        "The finance team attributed the growth to strong product sales. "
        "Operating margins also improved compared to the previous quarter."
    )
    chunks = chunker.chunk(text)
    # Expect roughly two topic groups (cats, finance) -- not asserting an
    # exact chunk count, since the breakpoint threshold is adaptive, but the
    # first sentence (cats) and the last sentence (finance) must not land in
    # the same chunk together, since that would mean the technique found no
    # topic shift in a document that clearly has one.
    first_chunk = next(c for c in chunks if "Cats are small" in c)
    last_chunk = next(c for c in chunks if "Operating margins" in c)
    assert first_chunk != last_chunk


def test_empty_text_produces_no_chunks(embedding_model):
    chunker = SemanticChunker(embedding_model, chunk_size_tokens=512)
    assert chunker.chunk("") == []


def test_short_single_topic_text_produces_one_chunk(embedding_model):
    chunker = SemanticChunker(embedding_model, chunk_size_tokens=512)
    chunks = chunker.chunk("Cats are small mammals. They are often kept as pets.")
    assert len(chunks) == 1
