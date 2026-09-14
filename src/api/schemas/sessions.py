from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.identity.domain.entities import MAX_SESSION_TITLE_CHARS, ChatSession
from src.orchestration.application.unified_answer_question import UnifiedAnswer
from src.orchestration.domain.entities import PARADIGM_ORDER

MAX_QUESTION_CHARS = 4000


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_SESSION_TITLE_CHARS)


class SessionResponse(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime
    # The latest turn's context-slice record, or None before the first answer.
    context_budget: dict[str, Any] | None

    @classmethod
    def of(cls, session: ChatSession) -> SessionResponse:
        return cls(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            context_budget=session.context_budget,
        )


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]


class AnswerRequest(BaseModel):
    # Bounded for the reason ChatRequest is: a question is embedded and forwarded to the
    # chat model at the caller's discretion.
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class SourceSchema(BaseModel):
    paradigm: str
    content: str
    source_id: uuid.UUID | None


class RoutingSchema(BaseModel):
    paradigms: list[str]
    mode: str
    fallback: str | None


class AttemptSchema(BaseModel):
    paradigm: str
    outcome: str
    elapsed_ms: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceSchema]
    routing: RoutingSchema | None
    attempts: list[AttemptSchema]
    degraded: bool
    dropped: dict[str, int]


def answer_response(answer: UnifiedAnswer) -> AnswerResponse:
    """Provenance, routing, and degradation, which a client shows a user. Router scores,
    similarity scores, and stage timings stay server-side (spec decision 8)."""
    decision = answer.decision
    routing = (
        None
        if decision is None
        else RoutingSchema(
            paradigms=[p.value for p in PARADIGM_ORDER if p in decision.paradigms],
            mode=decision.mode.value,
            fallback=answer.routing_fallback,
        )
    )
    return AnswerResponse(
        answer=answer.answer,
        sources=[
            SourceSchema(
                paradigm=item.paradigm.value, content=item.content, source_id=item.source_id
            )
            for item in answer.sources
        ],
        routing=routing,
        attempts=[
            AttemptSchema(
                paradigm=attempt.paradigm.value,
                outcome=attempt.outcome.value,
                elapsed_ms=attempt.elapsed_ms,
            )
            for attempt in answer.attempts
        ],
        degraded=answer.degraded,
        dropped={paradigm.value: count for paradigm, count in answer.dropped.items()},
    )
