import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.domain.entities import (
    ContextItem,
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierOutcome,
    TierRequest,
    TierResult,
)
from src.orchestration.infrastructure.in_memory_access_tracker import (
    InMemoryAccessFrequencyTracker,
)
from tests.unit.orchestration_fakes import FakeCascadeTier

CAG, MAG, RAG = Paradigm.CAG, Paradigm.MAG, Paradigm.RAG
_GENEROUS = TierTimeouts(cag=1.0, mag=1.0, rag=1.0)
_TIGHT = TierTimeouts(cag=0.05, mag=0.05, rag=0.05)
_SLOW = 0.5
_NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _request() -> TierRequest:
    return TierRequest(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        query="q",
        query_embedding=[1.0, 0.0],
    )


def _result(outcome: TierOutcome, paradigm: Paradigm, content: str = "x", source_id=None):
    return TierResult(outcome, [ContextItem(paradigm, content, 0.9, source_id)])


def _route(*paradigms: Paradigm, mode: RoutingMode = RoutingMode.CASCADE) -> RoutingDecision:
    return RoutingDecision(frozenset(paradigms), mode, {})


def _outcomes(result):
    return [(attempt.paradigm, attempt.outcome) for attempt in result.attempts]


async def test_a_cag_only_route_that_hits_returns_before_mag_or_rag_run():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG, "cached"))
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, mag, rag], _GENEROUS).run(_request(), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.HIT)]
    assert mag.requests == [] and rag.requests == []
    assert [item.content for item in result.items] == ["cached"]
    assert result.satisfied == frozenset({CAG})
    assert result.degraded is False


async def test_the_unrouted_cascade_stops_at_the_first_hit_in_cheapest_first_order():
    cag = FakeCascadeTier(CAG)
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, mag, rag], _GENEROUS).run(_request())
    assert _outcomes(result) == [(CAG, TierOutcome.MISS), (MAG, TierOutcome.HIT)]
    assert rag.requests == []


async def test_routing_to_rag_keeps_a_stale_cag_hit_out_of_the_answer():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG, "thirty days"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "forty-five days"))
    cascade = LatencyCascade([cag, rag], _GENEROUS)

    unrouted = await cascade.run(_request())
    assert [item.content for item in unrouted.items] == ["thirty days"]

    cag.requests.clear()
    routed = await cascade.run(_request(), _route(RAG))
    assert [item.content for item in routed.items] == ["forty-five days"]
    assert cag.requests == []


async def test_a_cag_miss_falls_through_to_rag_without_trying_an_unrouted_mag():
    cag = FakeCascadeTier(CAG)
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, mag, rag], _GENEROUS).run(_request(), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.MISS), (RAG, TierOutcome.HIT)]
    assert mag.requests == []


async def test_a_rag_only_route_never_touches_cag_or_mag():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG))
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, mag, rag], _GENEROUS).run(_request(), _route(RAG))
    assert _outcomes(result) == [(RAG, TierOutcome.HIT)]
    assert cag.requests == [] and mag.requests == []


async def test_a_partial_cag_match_is_kept_and_supplemented_by_rag():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.PARTIAL, CAG, "partial"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "full"))
    result = await LatencyCascade([cag, rag], _GENEROUS).run(_request(), _route(CAG))
    assert [item.content for item in result.items] == ["partial", "full"]
    assert result.satisfied == frozenset({RAG})


async def test_a_mag_and_rag_route_runs_both_even_when_mag_hits():
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([mag, rag], _GENEROUS).run(_request(), _route(MAG, RAG))
    assert _outcomes(result) == [(MAG, TierOutcome.HIT), (RAG, TierOutcome.HIT)]
    assert result.satisfied == frozenset({MAG, RAG})


async def test_a_routed_paradigm_with_no_configured_tier_falls_back_to_rag():
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([rag], _GENEROUS).run(_request(), _route(MAG))
    assert _outcomes(result) == [(RAG, TierOutcome.HIT)]


async def test_a_tier_that_exceeds_its_timeout_is_recorded_skipped_and_cancelled():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG), delay_seconds=_SLOW)
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, rag], _TIGHT).run(_request(), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.TIMEOUT), (RAG, TierOutcome.HIT)]
    assert result.degraded is True
    await asyncio.sleep(0.01)
    assert cag.cancelled is True


async def test_a_tier_that_raises_is_recorded_as_an_error_and_skipped():
    cag = FakeCascadeTier(CAG, error=RuntimeError("cache offline"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    result = await LatencyCascade([cag, rag], _GENEROUS).run(_request(), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.ERROR), (RAG, TierOutcome.HIT)]
    assert result.degraded is True


async def test_a_cancellation_raised_inside_a_tier_is_never_swallowed():
    cag = FakeCascadeTier(CAG, error=asyncio.CancelledError())
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    with pytest.raises(asyncio.CancelledError):
        await LatencyCascade([cag, rag], _GENEROUS).run(_request(), _route(CAG))


async def test_cancelling_the_caller_also_cancels_the_running_tier():
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG), delay_seconds=5.0)
    cascade = LatencyCascade([rag], TierTimeouts(cag=1.0, mag=1.0, rag=10.0))
    task = asyncio.create_task(cascade.run(_request(), _route(RAG)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.01)
    assert rag.cancelled is True


async def test_parallel_mode_runs_every_eligible_tier_and_merges_their_items():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG, "c"))
    mag = FakeCascadeTier(MAG, _result(TierOutcome.PARTIAL, MAG, "m"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "r"))
    decision = _route(CAG, MAG, mode=RoutingMode.PARALLEL)
    result = await LatencyCascade([cag, mag, rag], _GENEROUS).run(_request(), decision)
    assert [attempt.paradigm for attempt in result.attempts] == [CAG, MAG, RAG]
    assert sorted(item.content for item in result.items) == ["c", "m", "r"]
    assert result.satisfied == frozenset({CAG, RAG})


async def test_when_rag_times_out_earlier_items_come_back_as_a_degraded_best_effort():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.PARTIAL, CAG, "partial"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG), delay_seconds=_SLOW)
    result = await LatencyCascade([cag, rag], _TIGHT).run(_request(), _route(CAG))
    assert _outcomes(result) == [(CAG, TierOutcome.PARTIAL), (RAG, TierOutcome.TIMEOUT)]
    assert [item.content for item in result.items] == ["partial"]
    assert result.degraded is True


async def test_a_timed_out_rag_attempt_finishes_in_the_background_into_the_findings_sink():
    document_id = uuid.uuid4()
    rag = FakeCascadeTier(
        RAG, _result(TierOutcome.HIT, RAG, "late", document_id), delay_seconds=0.2
    )
    received: list[list[ContextItem]] = []

    async def sink(request: TierRequest, items: list[ContextItem]) -> None:
        received.append(items)

    tracker = InMemoryAccessFrequencyTracker()
    request = _request()
    cascade = LatencyCascade(
        [rag], _TIGHT, access_tracker=tracker, on_rag_findings=sink, clock=lambda: _NOW
    )
    result = await cascade.run(request, _route(RAG))
    assert _outcomes(result) == [(RAG, TierOutcome.TIMEOUT)]
    assert received == []

    await cascade.drain()
    assert [[item.content for item in items] for items in received] == [["late"]]
    assert tracker.access_count(request.tenant_id, document_id, timedelta(hours=1), _NOW) == 1


async def test_without_a_findings_sink_a_timed_out_rag_attempt_is_cancelled():
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG), delay_seconds=_SLOW)
    await LatencyCascade([rag], _TIGHT).run(_request(), _route(RAG))
    await asyncio.sleep(0.01)
    assert rag.cancelled is True


async def test_a_rag_hit_records_one_access_per_returned_document():
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    rag = FakeCascadeTier(
        RAG,
        TierResult(
            TierOutcome.HIT,
            [
                ContextItem(RAG, "a1", 0.9, doc_a),
                ContextItem(RAG, "a2", 0.8, doc_a),
                ContextItem(RAG, "b", 0.7, doc_b),
            ],
        ),
    )
    tracker = InMemoryAccessFrequencyTracker()
    request = _request()
    cascade = LatencyCascade([rag], _GENEROUS, access_tracker=tracker, clock=lambda: _NOW)
    await cascade.run(request, _route(RAG))
    window = timedelta(hours=1)
    assert tracker.access_count(request.tenant_id, doc_a, window, _NOW) == 1
    assert tracker.access_count(request.tenant_id, doc_b, window, _NOW) == 1


def test_a_cascade_without_a_rag_tier_is_rejected():
    with pytest.raises(ValueError):
        LatencyCascade([FakeCascadeTier(CAG)])


def test_two_tiers_for_one_paradigm_are_rejected():
    with pytest.raises(ValueError):
        LatencyCascade([FakeCascadeTier(RAG), FakeCascadeTier(RAG)])


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_timeouts_are_rejected(bad):
    with pytest.raises(ValueError):
        TierTimeouts(cag=bad)


async def test_drain_waits_for_a_cancelled_tier_to_finish_cleaning_up():
    # A timed-out tier is cancelled, not awaited, so the answer isn't held up
    # by its cleanup -- but shutdown must not race that cleanup either.
    cag = FakeCascadeTier(
        CAG, _result(TierOutcome.HIT, CAG), delay_seconds=_SLOW, cleanup_seconds=0.2
    )
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))
    cascade = LatencyCascade([cag, rag], _TIGHT)

    await cascade.run(_request(), _route(CAG))
    await asyncio.sleep(0.01)
    assert cag.cancelled is True
    assert cag.cleaned_up is False

    await cascade.drain()
    assert cag.cleaned_up is True


def test_non_positive_background_limits_are_rejected():
    with pytest.raises(ValueError):
        LatencyCascade([FakeCascadeTier(RAG)], background_timeout=0.0)
    with pytest.raises(ValueError):
        LatencyCascade([FakeCascadeTier(RAG)], max_background=0)


async def test_parallel_mode_records_a_tier_timeout_and_still_merges_the_others():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG, "c"), delay_seconds=_SLOW)
    mag = FakeCascadeTier(MAG, _result(TierOutcome.HIT, MAG, "m"))
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "r"))
    decision = _route(CAG, MAG, mode=RoutingMode.PARALLEL)
    result = await LatencyCascade([cag, mag, rag], _TIGHT).run(_request(), decision)
    assert (CAG, TierOutcome.TIMEOUT) in _outcomes(result)
    assert sorted(item.content for item in result.items) == ["m", "r"]
    assert result.degraded is True


async def test_a_cancellation_raised_by_one_parallel_tier_cancels_its_siblings():
    cag = FakeCascadeTier(CAG, error=asyncio.CancelledError())
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG), delay_seconds=5.0)
    cascade = LatencyCascade([cag, rag], TierTimeouts(cag=1.0, mag=1.0, rag=10.0))
    with pytest.raises(asyncio.CancelledError):
        await cascade.run(_request(), _route(CAG, mode=RoutingMode.PARALLEL))
    await asyncio.sleep(0.01)
    assert rag.cancelled is True


async def test_a_timed_out_cag_tier_is_cancelled_even_when_a_findings_sink_is_configured():
    cag = FakeCascadeTier(CAG, _result(TierOutcome.HIT, CAG), delay_seconds=_SLOW)
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG))

    async def sink(request: TierRequest, items: list[ContextItem]) -> None:
        return None

    await LatencyCascade([cag, rag], _TIGHT, on_rag_findings=sink).run(_request(), _route(CAG))
    await asyncio.sleep(0.01)
    assert cag.cancelled is True


async def test_a_background_rag_attempt_past_its_background_timeout_is_cancelled_unreported():
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "late"), delay_seconds=_SLOW)
    received: list[list[ContextItem]] = []

    async def sink(request: TierRequest, items: list[ContextItem]) -> None:
        received.append(items)

    cascade = LatencyCascade([rag], _TIGHT, on_rag_findings=sink, background_timeout=0.05)
    await cascade.run(_request(), _route(RAG))
    await cascade.drain()
    assert received == []
    assert rag.cancelled is True


async def test_timed_out_rag_attempts_beyond_the_background_cap_are_cancelled_not_kept():
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "late"), delay_seconds=0.3)
    received: list[list[ContextItem]] = []

    async def sink(request: TierRequest, items: list[ContextItem]) -> None:
        received.append(items)

    cascade = LatencyCascade([rag], _TIGHT, on_rag_findings=sink, max_background=1)
    await cascade.run(_request(), _route(RAG))  # kept, finishing in the background
    await cascade.run(_request(), _route(RAG))  # over the cap: cancelled instead
    await asyncio.sleep(0.01)
    assert rag.cancelled is True
    await cascade.drain()
    assert len(received) == 1


async def test_a_background_rag_attempt_that_raises_never_reaches_the_sink():
    rag = FakeCascadeTier(RAG, delay_seconds=0.1, error=RuntimeError("index offline"))
    received: list[list[ContextItem]] = []

    async def sink(request: TierRequest, items: list[ContextItem]) -> None:
        received.append(items)

    cascade = LatencyCascade([rag], _TIGHT, on_rag_findings=sink)
    result = await cascade.run(_request(), _route(RAG))
    assert _outcomes(result) == [(RAG, TierOutcome.TIMEOUT)]
    await cascade.drain()
    assert received == []


async def test_a_findings_sink_that_raises_does_not_escape_drain():
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "late"), delay_seconds=0.1)

    async def sink(request: TierRequest, items: list[ContextItem]) -> None:
        raise RuntimeError("memory store offline")

    cascade = LatencyCascade([rag], _TIGHT, on_rag_findings=sink)
    await cascade.run(_request(), _route(RAG))
    await cascade.drain()


class _BrokenTracker(InMemoryAccessFrequencyTracker):
    def record_access(self, tenant_id, document_id, at):
        raise RuntimeError("tracker offline")


async def test_an_access_tracker_that_raises_does_not_fail_a_rag_hit():
    rag = FakeCascadeTier(RAG, _result(TierOutcome.HIT, RAG, "doc", uuid.uuid4()))
    cascade = LatencyCascade([rag], _GENEROUS, access_tracker=_BrokenTracker())
    result = await cascade.run(_request(), _route(RAG))
    assert _outcomes(result) == [(RAG, TierOutcome.HIT)]
    assert result.degraded is False


async def test_an_access_tracker_that_raises_does_not_stop_background_findings():
    rag = FakeCascadeTier(
        RAG, _result(TierOutcome.HIT, RAG, "late", uuid.uuid4()), delay_seconds=0.1
    )
    received: list[list[ContextItem]] = []

    async def sink(request: TierRequest, items: list[ContextItem]) -> None:
        received.append(items)

    cascade = LatencyCascade(
        [rag], _TIGHT, access_tracker=_BrokenTracker(), on_rag_findings=sink
    )
    await cascade.run(_request(), _route(RAG))
    await cascade.drain()
    assert len(received) == 1
