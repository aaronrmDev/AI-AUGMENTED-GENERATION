import uuid

import pytest
from pydantic import ValidationError

from src.api.schemas.sessions import AnswerRequest, CreateSessionRequest, answer_response
from src.orchestration.application.unified_answer_question import StageTimings, UnifiedAnswer
from src.orchestration.domain.budget_allocator import allocate
from src.orchestration.domain.entities import (
    ContextItem,
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierAttempt,
    TierOutcome,
)


def test_a_title_may_be_absent_or_up_to_200_characters():
    assert CreateSessionRequest().title is None
    assert CreateSessionRequest(title="x" * 200).title == "x" * 200
    with pytest.raises(ValidationError):
        CreateSessionRequest(title="x" * 201)


@pytest.mark.parametrize("question", ["", "   ", "x" * 4001])
def test_a_blank_or_oversized_question_is_refused(question):
    with pytest.raises(ValidationError):
        AnswerRequest(question=question)


def test_a_question_of_4000_characters_is_accepted():
    assert len(AnswerRequest(question="x" * 4000).question) == 4000


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (CreateSessionRequest, {"title": "t", "user_id": str(uuid.uuid4())}),
        (AnswerRequest, {"question": "q", "tenant_id": str(uuid.uuid4())}),
    ],
)
def test_unknown_fields_are_refused_rather_than_silently_ignored(model, payload):
    # Identity never comes from a body, so an ignored user_id grants nothing today. A
    # refused one fails closed, and keeps a client from believing it set something.
    with pytest.raises(ValidationError):
        model(**payload)


def _answer(decision: RoutingDecision | None, fallback, degraded: bool) -> UnifiedAnswer:
    source_id = uuid.UUID(int=7)
    contributing = frozenset({Paradigm.MAG, Paradigm.RAG})
    return UnifiedAnswer(
        answer="forty-five days",
        sources=[ContextItem(Paradigm.RAG, "Returns within forty-five days.", 0.8, source_id)],
        decision=decision,
        routing_fallback=fallback,
        attempts=[
            TierAttempt(Paradigm.MAG, TierOutcome.TIMEOUT, 50.4),
            TierAttempt(Paradigm.RAG, TierOutcome.HIT, 12.5),
        ],
        allocation=allocate(128_000, contributing),
        dropped={Paradigm.RAG: 1},
        degraded=degraded,
        timings=StageTimings(9.0, 1.0, 60.0, 0.5, 2.0, 900.0),
    )


def test_an_answer_maps_provenance_routing_and_degradation():
    decision = RoutingDecision(
        frozenset({Paradigm.RAG, Paradigm.MAG}), RoutingMode.PARALLEL, {Paradigm.RAG: 0.9}
    )

    response = answer_response(_answer(decision, "classifier_timeout", degraded=True))

    assert response.model_dump(mode="json") == {
        "answer": "forty-five days",
        "sources": [
            {
                "paradigm": "rag",
                "content": "Returns within forty-five days.",
                "source_id": "00000000-0000-0000-0000-000000000007",
            }
        ],
        "routing": {
            "paradigms": ["mag", "rag"],
            "mode": "parallel",
            "fallback": "classifier_timeout",
        },
        "attempts": [
            {"paradigm": "mag", "outcome": "timeout", "elapsed_ms": 50.4},
            {"paradigm": "rag", "outcome": "hit", "elapsed_ms": 12.5},
        ],
        "degraded": True,
        "dropped": {"rag": 1},
    }


def test_an_unrouted_answer_has_no_routing():
    assert answer_response(_answer(None, None, degraded=False)).routing is None
