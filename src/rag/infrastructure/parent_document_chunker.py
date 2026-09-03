from src.rag.domain.entities import ParentChildChunks
from src.rag.infrastructure.fixed_size_chunker import FixedSizeChunker


class ParentDocumentChunker:
    def __init__(
        self, parent_chunk_size_tokens: int = 1000, child_chunk_size_tokens: int = 200
    ) -> None:
        self._parent_chunker = FixedSizeChunker(chunk_size_tokens=parent_chunk_size_tokens)
        self._child_chunker = FixedSizeChunker(chunk_size_tokens=child_chunk_size_tokens)

    def chunk_with_parents(self, text: str) -> ParentChildChunks:
        parents = self._parent_chunker.chunk(text)
        children: list[tuple[str, int]] = []
        for i, parent in enumerate(parents):
            children.extend((child, i) for child in self._child_chunker.chunk(parent))
        return ParentChildChunks(parents=parents, children=children)
