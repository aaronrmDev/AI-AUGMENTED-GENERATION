import uuid

import pytest

from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import ContextItem, Paradigm, TierOutcome, TierResult
from src.orchestration.domain.errors import QueryExceedsBudget
from tests.unit.orchestration_fakes import (
    FakeCascadeTier,
    FakeQueryClassifier,
    FakeSessionBudgetRecorder,
)
from tests.unit.rag_fakes import FakeChatModel, FakeEmbeddingModel

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG
_GENEROUS = TierTimeouts(cag=1.0, mag=1.0, rag=1.0)
_RAG_ONLY = {CAG: 0.0, MAG: 0.0, RAG: 1.0}
_QUESTION = "what changed today?"


def _hit(paradigm: Paradigm, content: str) -> TierResult:
    return TierResult(TierOutcome.HIT, [ContextItem(paradigm, content, 0.9, uuid.uuid4())])


def _ids() -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    return uuid.uuid4(), uuid.uuid4(), uuid.uuid4()


async def test_a_routed_question_flows_through_every_stage_and_records_its_budget():
    classifier = FakeQueryClassifier(_RAG_ONLY)
    cag = FakeCascadeTier(CAG, _hit(CAG, "stale"))
    rag = FakeCascadeTier(RAG, _hit(RAG, "fresh doc"))
    chat_model = FakeChatModel("the answer")
    recorder = FakeSessionBudgetRecorder()
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(),
        classifier,
        LatencyCascade([cag, rag], _GENEROUS),
        chat_model,
        budget_recorder=recorder,
    )
    tenant_id, user_id, session_id = _ids()

    result = await use_case.execute(tenant_id, user_id, session_id, _QUESTION)

    assert result.answer == "the answer"
    assert result.decision is not None
    assert result.decision.paradigms == frozenset({RAG})
    assert [attempt.paradigm for attempt in result.attempts] == [RAG]
    assert cag.requests == []
    assert result.allocation == allocate(128_000, {RAG})
    assert "fresh doc" in chat_model.last_context
    assert "(RAG)" in chat_model.last_context
    assert "stale" not in chat_model.last_context
    assert [source.content for source in result.sources] == ["fresh doc"]
    assert recorder.records == [(tenant_id, session_id, result.allocation, frozenset({RAG}))]
    assert result.degraded is False
    timings = result.timings
    assert min(
        timings.embed_ms, timings.route_ms, timings.cascade_ms,
        timings.assemble_ms, timings.generate_ms,
    ) >= 0.0


async def test_the_classifier_and_the_tiers_share_one_upstream_embedding_and_scope():
    embedder = FakeEmbeddingModel()
    classifier = FakeQueryClassifier(_RAG_ONLY)
    rag = FakeCascadeTier(RAG, _hit(RAG, "doc"))
    use_case = UnifiedAnswerQuestion(
        embedder, classifier, LatencyCascade([rag], _GENEROUS), FakeChatModel()
    )
    tenant_id, user_id, session_id = _ids()

    await use_case.execute(tenant_id, user_id, session_id, _QUESTION)

    assert classifier.calls == [(_QUESTION, embedder.embed(_QUESTION))]
    [request] = rag.requests
    assert request.query_embedding == embedder.embed(_QUESTION)
    assert (request.tenant_id, request.user_id, request.session_id, request.query) == (
        tenant_id, user_id, session_id, _QUESTION,
    )


async def test_without_a_classifier_the_source_cascade_answers_unrouted():
    cag = FakeCascadeTier(CAG, _hit(CAG, "cached"))
    rag = FakeCascadeTier(RAG, _hit(RAG, "fresh"))
    chat_model = FakeChatModel()
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(), None, LatencyCascade([cag, rag], _GENEROUS), chat_model
    )

    result = await use_case.execute(*_ids(), _QUESTION)

    assert result.decision is None
    assert [attempt.paradigm for attempt in result.attempts] == [CAG]
    assert "cached" in chat_model.last_context


async def test_a_question_longer_than_the_query_slice_is_rejected_before_any_work():
    classifier = FakeQueryClassifier(_RAG_ONLY)
    chat_model = FakeChatModel()
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(),
        classifier,
        LatencyCascade([FakeCascadeTier(RAG)], _GENEROUS),
        chat_model,
        total_context_tokens=100,
    )

    with pytest.raises(QueryExceedsBudget) as raised:
        await use_case.execute(*_ids(), "word " * 50)

    assert raised.value.query_slice == 10
    assert classifier.calls == []
    assert chat_model.last_question is None


async def test_items_that_do_not_fit_their_slice_are_reported_as_dropped():
    rag = FakeCascadeTier(RAG, _hit(RAG, "alpha " * 400))
    chat_model = FakeChatModel()
    use_case = UnifiedAnswerQuestion(
        FakeEmbeddingModel(),
        FakeQueryClassifier(_RAG_ONLY),
        LatencyCascade([rag], _GENEROUS),
        chat_model,
        total_context_tokens=200,
    )

    result = await use_case.execute(*_ids(), "q")

    assert result.dropped[RAG] == 1
    assert result.sources == []
    assert chat_model.last_context == ""


def test_a_non_positive_context_window_is_rejected():
    with pytest.raises(ValueError):
        UnifiedAnswerQuestion(
            FakeEmbeddingModel(),
            None,
            LatencyCascade([FakeCascadeTier(RAG)]),
            FakeChatModel(),
            total_context_tokens=0,
        )
