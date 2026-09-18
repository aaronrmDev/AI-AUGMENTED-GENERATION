import json

from src.api.sse import encode_sse
from src.orchestration.application.unified_answer_question import (
    AnswerChunkEvent,
    AnswerCompleteEvent,
    AnswerErrorEvent,
    BudgetStageEvent,
    RetrievalStageEvent,
    RoutingStageEvent,
)
from src.orchestration.domain.entities import (
    ContextItem,
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierAttempt,
    TierOutcome,
)

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG


async def _aiter(items):
    for item in items:
        yield item


async def _frames(events):
    return [frame async for frame in encode_sse(_aiter(events))]


async def test_a_routing_event_with_a_decision_encodes_paradigms_mode_and_fallback():
    decision = RoutingDecision(frozenset({RAG}), RoutingMode.CASCADE, {RAG: 0.9})

    [frame] = await _frames([RoutingStageEvent(decision, None)])

    assert frame == (
        b"event: routing\n"
        b'data: {"paradigms": ["rag"], "mode": "cascade", "fallback": null}\n\n'
    )


async def test_a_routing_event_with_no_decision_encodes_empty_paradigms_and_null_mode():
    [frame] = await _frames([RoutingStageEvent(None, "classifier_timeout")])

    assert frame == (
        b"event: routing\n"
        b'data: {"paradigms": [], "mode": null, "fallback": "classifier_timeout"}\n\n'
    )


async def test_a_retrieval_event_encodes_attempts_and_degraded():
    attempt = TierAttempt(RAG, TierOutcome.HIT, 12.5)

    [frame] = await _frames([RetrievalStageEvent([attempt], True)])

    assert frame == (
        b"event: retrieval\n"
        b'data: {"attempts": [{"paradigm": "rag", "outcome": "hit", "elapsed_ms": 12.5}],'
        b' "degraded": true}\n\n'
    )


async def test_a_budget_event_encodes_sources_and_dropped_counts():
    item = ContextItem(RAG, "some text", 0.8, None)

    [frame] = await _frames([BudgetStageEvent([item], {CAG: 0, MAG: 0, RAG: 2})])

    payload = json.loads(frame.decode().split("data: ", 1)[1])
    assert payload["sources"] == [{"paradigm": "rag", "content": "some text", "source_id": None}]
    assert payload["dropped"] == {"cag": 0, "mag": 0, "rag": 2}


async def test_a_chunk_event_encodes_its_text():
    [frame] = await _frames([AnswerChunkEvent("hello")])

    assert frame == b'event: chunk\ndata: {"text": "hello"}\n\n'


async def test_a_chunk_event_with_an_embedded_newline_round_trips_through_json():
    [frame] = await _frames([AnswerChunkEvent("line one\nline two")])

    text_line, data_line = frame.split(b"\n", 1)
    assert text_line == b"event: chunk"
    # The frame carries exactly three real newlines -- the one ending the
    # "event: chunk" line, the one ending the data line, and the blank line
    # terminating the SSE event -- because json.dumps escaped the chunk's own
    # embedded newline as the two characters backslash-n, never a real line
    # break, so it doesn't introduce a fourth.
    assert frame.count(b"\n") == 3
    payload = json.loads(data_line.decode().removeprefix("data: ").rstrip("\n"))
    assert payload["text"] == "line one\nline two"


async def test_a_complete_event_encodes_as_done_with_no_payload():
    [frame] = await _frames([AnswerCompleteEvent()])

    assert frame == b"event: done\ndata: {}\n\n"


async def test_an_error_event_encodes_its_message_as_detail():
    [frame] = await _frames([AnswerErrorEvent("answer generation failed")])

    assert frame == b'event: error\ndata: {"detail": "answer generation failed"}\n\n'
