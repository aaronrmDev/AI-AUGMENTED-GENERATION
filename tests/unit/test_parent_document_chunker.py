# tests/unit/test_parent_document_chunker.py
from src.rag.infrastructure.parent_document_chunker import ParentDocumentChunker


def test_children_are_smaller_than_their_parent_and_reference_it_by_index():
    text = "First sentence. Second sentence. " * 200  # long enough to force multiple parents
    chunker = ParentDocumentChunker(parent_chunk_size_tokens=100, child_chunk_size_tokens=20)

    result = chunker.chunk_with_parents(text)

    assert len(result.parents) > 1
    assert len(result.children) > len(result.parents)
    for child_content, parent_index in result.children:
        assert 0 <= parent_index < len(result.parents)
        assert child_content in result.parents[parent_index]


def test_empty_text_produces_no_parents_and_no_children():
    chunker = ParentDocumentChunker()
    result = chunker.chunk_with_parents("")
    assert result.parents == []
    assert result.children == []


def test_short_text_produces_one_parent_and_at_least_one_child():
    chunker = ParentDocumentChunker(parent_chunk_size_tokens=1000, child_chunk_size_tokens=200)
    result = chunker.chunk_with_parents("A short document with just one sentence.")
    assert len(result.parents) == 1
    assert len(result.children) >= 1
    assert result.children[0][1] == 0
