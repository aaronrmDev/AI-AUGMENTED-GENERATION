from collections.abc import Sequence

from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm
from src.orchestration.domain.ports import QueryClassifier
from src.orchestration.domain.similarity import cosine_similarity
from src.orchestration.infrastructure.routing_exemplars import (
    DEFAULT_ROUTING_EXEMPLARS,
    RoutingExemplar,
)
from src.rag.domain.ports import EmbeddingModel


class PrototypeQueryClassifier(QueryClassifier):
    """Similarity-weighted vote over the k nearest labeled exemplars.

    Each paradigm's score is the share of neighbor similarity carried by
    exemplars labeled with it. An exemplar with two labels votes for both,
    which is how a RAG + MAG route can come out with both scores high.
    Exemplars are embedded once, here; queries arrive already embedded.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        exemplars: Sequence[RoutingExemplar] = DEFAULT_ROUTING_EXEMPLARS,
        k: int = 5,
    ) -> None:
        if not exemplars:
            raise ValueError("at least one exemplar is required")
        if k < 1:
            raise ValueError("k must be at least 1")
        self._k = k
        self._exemplars = [
            (exemplar, embedding_model.embed(exemplar.query)) for exemplar in exemplars
        ]

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        nearest = sorted(
            (
                (cosine_similarity(query_embedding, embedding), exemplar)
                for exemplar, embedding in self._exemplars
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )[: self._k]
        weighted = [(max(0.0, similarity), exemplar) for similarity, exemplar in nearest]
        total = sum(weight for weight, _ in weighted)
        if total == 0.0:
            return dict.fromkeys(PARADIGM_ORDER, 0.0)
        return {
            paradigm: sum(weight for weight, ex in weighted if paradigm in ex.paradigms) / total
            for paradigm in PARADIGM_ORDER
        }
