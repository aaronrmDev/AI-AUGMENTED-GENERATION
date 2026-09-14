import threading
import uuid

from src.rag.domain.entities import SearchResult
from src.rag.infrastructure.cross_encoder_reranker import CrossEncoderReranker


class _LengthScoringModel:
    """Scores each (query, passage) pair by the passage's length, recording where it ran,
    so reranking can be tested without loading a real cross-encoder."""

    def __init__(self) -> None:
        self.thread_ids: list[int] = []
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.thread_ids.append(threading.get_ident())
        self.pairs.extend(pairs)
        return [float(len(passage)) for _, passage in pairs]


def _result(content: str) -> SearchResult:
    return SearchResult(
        document_id=uuid.uuid4(), chunk_id=uuid.uuid4(), content=content, score=0.5
    )


async def test_results_are_reordered_by_the_models_score_for_each_query_passage_pair():
    brief, longer = _result("brief"), _result("a much longer passage")
    model = _LengthScoringModel()

    reranked = await CrossEncoderReranker(model=model).rerank(
        query="q", results=[brief, longer], top_k=2
    )

    assert reranked == [longer, brief]
    assert model.pairs == [("q", "brief"), ("q", "a much longer passage")]


async def test_only_the_top_k_results_are_returned():
    results = [_result("a"), _result("bbb"), _result("cc")]

    reranked = await CrossEncoderReranker(model=_LengthScoringModel()).rerank(
        query="q", results=results, top_k=2
    )

    assert [r.content for r in reranked] == ["bbb", "cc"]


async def test_no_results_rerank_to_none_without_calling_the_model():
    model = _LengthScoringModel()

    assert await CrossEncoderReranker(model=model).rerank(query="q", results=[], top_k=3) == []
    assert model.pairs == []


async def test_scoring_runs_off_the_event_loop():
    # A cross-encoder forward pass per pair is CPU work that, on the loop, would
    # stall the other cascade tiers sharing it in a PARALLEL route.
    model = _LengthScoringModel()

    await CrossEncoderReranker(model=model).rerank(
        query="q", results=[_result("a"), _result("bb")], top_k=1
    )

    assert model.thread_ids
    assert threading.get_ident() not in model.thread_ids
