import json
from collections.abc import AsyncIterator
from typing import Any

from src.orchestration.application.unified_answer_question import (
    AnswerChunkEvent,
    AnswerCompleteEvent,
    AnswerErrorEvent,
    BudgetStageEvent,
    RetrievalStageEvent,
    RoutingStageEvent,
    UnifiedAnswerEvent,
)
from src.orchestration.domain.entities import PARADIGM_ORDER


def _wire(event: UnifiedAnswerEvent) -> tuple[str, dict[str, Any]]:
    """Maps one domain event to its SSE event name and JSON-serializable payload.
    Never includes router scores, similarity scores, or stage timings -- the same
    restriction answer_response() already applies to the JSON endpoint."""
    if isinstance(event, RoutingStageEvent):
        decision = event.decision
        return "routing", {
            "paradigms": (
                []
                if decision is None
                else [p.value for p in PARADIGM_ORDER if p in decision.paradigms]
            ),
            "mode": None if decision is None else decision.mode.value,
            "fallback": event.routing_fallback,
        }
    if isinstance(event, RetrievalStageEvent):
        return "retrieval", {
            "attempts": [
                {
                    "paradigm": attempt.paradigm.value,
                    "outcome": attempt.outcome.value,
                    "elapsed_ms": attempt.elapsed_ms,
                }
                for attempt in event.attempts
            ],
            "degraded": event.degraded,
        }
    if isinstance(event, BudgetStageEvent):
        return "budget", {
            "sources": [
                {
                    "paradigm": source.paradigm.value,
                    "content": source.content,
                    "source_id": str(source.source_id) if source.source_id else None,
                }
                for source in event.sources
            ],
            "dropped": {paradigm.value: count for paradigm, count in event.dropped.items()},
        }
    if isinstance(event, AnswerChunkEvent):
        return "chunk", {"text": event.text}
    if isinstance(event, AnswerCompleteEvent):
        return "done", {}
    assert isinstance(event, AnswerErrorEvent)
    return "error", {"detail": event.message}


async def encode_sse(events: AsyncIterator[UnifiedAnswerEvent]) -> AsyncIterator[bytes]:
    async for event in events:
        name, payload = _wire(event)
        yield f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
