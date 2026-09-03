from typing import cast

from sentence_transformers import SentenceTransformer

from src.rag.domain.ports import EmbeddingModel

_MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformersEmbedder(EmbeddingModel):
    def __init__(self) -> None:
        self._model = SentenceTransformer(_MODEL_NAME)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, convert_to_numpy=True)
        # .tolist() on an ndarray is typed as returning Any, so the cast is what
        # holds this method to its declared return type under mypy --strict.
        return cast(list[float], vector.tolist())
