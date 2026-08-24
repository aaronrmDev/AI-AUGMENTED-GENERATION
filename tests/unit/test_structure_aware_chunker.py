# tests/unit/test_structure_aware_chunker.py
from src.rag.infrastructure.structure_aware_chunker import StructureAwareChunker


def test_keeps_each_headings_content_together_when_it_fits():
    chunker = StructureAwareChunker(chunk_size_tokens=512)
    text = "# Section One\nSome content here.\n\n# Section Two\nOther content here."
    chunks = chunker.chunk(text)
    assert len(chunks) == 2
    assert "Section One" in chunks[0]
    assert "Some content here" in chunks[0]
    assert "Section Two" in chunks[1]


def test_never_splits_inside_a_fenced_code_block():
    chunker = StructureAwareChunker(chunk_size_tokens=5)
    text = "# Code Example\n```python\ndef f():\n    return 1\n```\nEnd of example."
    chunks = chunker.chunk(text)
    code_chunk = next(c for c in chunks if "def f():" in c)
    assert "```python" in code_chunk
    assert "return 1" in code_chunk
    assert "```" in code_chunk.split("```python", 1)[1]  # the closing fence is in the same chunk


def test_a_document_with_no_headings_falls_back_to_sentence_based():
    chunker = StructureAwareChunker(chunk_size_tokens=512)
    chunks = chunker.chunk("Just a plain paragraph with no headings at all. It has two sentences.")
    assert len(chunks) == 1
    assert "plain paragraph" in chunks[0]


def test_empty_text_produces_no_chunks():
    chunker = StructureAwareChunker(chunk_size_tokens=512)
    assert chunker.chunk("") == []


def test_a_headingless_document_with_a_fenced_code_block_keeps_the_fence_intact():
    # Regression test: the "no heading structure at all" case must not bypass
    # fence protection by handing the raw text straight to the fence-unaware
    # SentenceBasedChunker fallback. It has to flow through the same
    # per-section loop that keeps a fenced section whole.
    chunker = StructureAwareChunker(chunk_size_tokens=5)
    text = (
        "Intro sentence one. Intro sentence two.\n"
        "```python\n"
        "def f():\n"
        "    # This is a comment. This explains it further. And one more clause here.\n"
        "    return 1\n"
        "```\n"
        "Outro sentence one. Outro sentence two."
    )
    chunks = chunker.chunk(text)
    code_chunk = next(c for c in chunks if "def f():" in c)
    assert "```python" in code_chunk
    assert "return 1" in code_chunk
    assert "```" in code_chunk.split("```python", 1)[1]  # the closing fence is in the same chunk


def test_a_hash_comment_inside_a_fence_is_not_mistaken_for_a_real_heading():
    # Regression test: a document with no real Markdown headings, where the
    # only "#"-prefixed line is inside a fenced code block, must not be
    # misclassified by a fence-unaware heading check. The fence must survive
    # intact in a single chunk either way.
    chunker = StructureAwareChunker(chunk_size_tokens=5)
    text = (
        "Intro sentence one. Intro sentence two.\n"
        "```bash\n"
        "# this looks like a heading but is code\n"
        "echo hi. Then more code runs.\n"
        "```\n"
        "Outro sentence one. Outro sentence two."
    )
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert "```bash" in chunks[0]
    assert "# this looks like a heading but is code" in chunks[0]
    assert "```" in chunks[0].split("```bash", 1)[1]  # the closing fence is in the same chunk
