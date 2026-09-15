import pytest

from src.orchestration.domain.entities import Paradigm
from src.orchestration.infrastructure.prototype_query_classifier import (
    PrototypeQueryClassifier,
)
from src.orchestration.infrastructure.routing_exemplars import (
    DEFAULT_ROUTING_EXEMPLARS,
    RoutingExemplar,
)
from src.rag.domain.ports import EmbeddingModel

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG
_CONCEPT_ONE_QUERIES = (
    "What's our refund policy?",
    "What changed in the policy today?",
    "Continue where we left off yesterday",
    "Compare today's sales with last month",
    "Explain this code file",
    "What did I ask you to remember?",
)


class _TableEmbedder(EmbeddingModel):
    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return self._table[text]


_TABLE = {
    "cag example": [1.0, 0.0, 0.0],
    "rag example": [0.0, 1.0, 0.0],
    "mag example": [0.0, 0.0, 1.0],
    "rag mag example": [0.0, 0.7071, 0.7071],
}
_EXEMPLARS = (
    RoutingExemplar("cag example", frozenset({CAG})),
    RoutingExemplar("rag example", frozenset({RAG})),
    RoutingExemplar("mag example", frozenset({MAG})),
    RoutingExemplar("rag mag example", frozenset({RAG, MAG})),
)


async def test_the_single_nearest_exemplar_decides_when_k_is_one():
    classifier = PrototypeQueryClassifier(_TableEmbedder(_TABLE), _EXEMPLARS, k=1)
    assert await classifier.score("q", [1.0, 0.0, 0.0]) == {CAG: 1.0, MAG: 0.0, RAG: 0.0}


async def test_neighbors_vote_by_similarity_and_multi_label_exemplars_vote_for_each_label():
    classifier = PrototypeQueryClassifier(_TableEmbedder(_TABLE), _EXEMPLARS, k=2)
    scores = await classifier.score("q", [0.0, 1.0, 0.0])
    assert scores[RAG] == pytest.approx(1.0)
    assert scores[MAG] == pytest.approx(0.7071 / 1.7071, abs=1e-4)
    assert scores[CAG] == 0.0


async def test_neighbors_with_no_positive_similarity_give_all_zero_scores():
    classifier = PrototypeQueryClassifier(_TableEmbedder(_TABLE), _EXEMPLARS, k=1)
    assert await classifier.score("q", [-1.0, 0.0, 0.0]) == {CAG: 0.0, MAG: 0.0, RAG: 0.0}


async def test_exemplars_are_embedded_once_at_construction_not_per_query():
    embedder = _TableEmbedder(_TABLE)
    classifier = PrototypeQueryClassifier(embedder, _EXEMPLARS, k=2)
    await classifier.score("q", [1.0, 0.0, 0.0])
    await classifier.score("q", [0.0, 1.0, 0.0])
    assert embedder.calls == len(_EXEMPLARS)


def test_an_empty_exemplar_set_or_a_non_positive_k_is_rejected():
    with pytest.raises(ValueError):
        PrototypeQueryClassifier(_TableEmbedder(_TABLE), ())
    with pytest.raises(ValueError):
        PrototypeQueryClassifier(_TableEmbedder(_TABLE), _EXEMPLARS, k=0)


def test_the_default_exemplars_are_labeled_and_never_contain_a_concept_one_query():
    assert all(exemplar.paradigms for exemplar in DEFAULT_ROUTING_EXEMPLARS)
    queries = {exemplar.query.casefold() for exemplar in DEFAULT_ROUTING_EXEMPLARS}
    assert queries.isdisjoint(query.casefold() for query in _CONCEPT_ONE_QUERIES)
