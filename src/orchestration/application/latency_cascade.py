import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from src.orchestration.domain.entities import (
    PARADIGM_ORDER,
    CascadeResult,
    ContextItem,
    Paradigm,
    RoutingDecision,
    RoutingMode,
    TierAttempt,
    TierOutcome,
    TierRequest,
    TierResult,
)
from src.orchestration.domain.ports import AccessFrequencyTracker, CascadeTier

logger = logging.getLogger(__name__)

RagFindingsSink = Callable[[TierRequest, list[ContextItem]], Awaitable[None]]

# Provisional, not measured: a background RAG completion gets this long after
# its tier timeout before it is cancelled, and at most this many run at once.
# They bound resource use when a RAG backend hangs; no run in this project has
# exercised either limit under real load yet.
DEFAULT_BACKGROUND_TIMEOUT = 30.0
DEFAULT_MAX_BACKGROUND = 16


@dataclass(frozen=True)
class TierTimeouts:
    # Seconds. Defaults are Concept 5's own timeout guards.
    cag: float = 0.010
    mag: float = 0.050
    rag: float = 2.0

    def __post_init__(self) -> None:
        if min(self.cag, self.mag, self.rag) <= 0.0:
            raise ValueError("tier timeouts must be positive")

    def for_paradigm(self, paradigm: Paradigm) -> float:
        return {Paradigm.CAG: self.cag, Paradigm.MAG: self.mag, Paradigm.RAG: self.rag}[paradigm]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


class _Collector:
    def __init__(self) -> None:
        self.items: list[ContextItem] = []
        self.attempts: list[TierAttempt] = []
        self.satisfied: set[Paradigm] = set()
        self.degraded = False

    def add(self, attempt: TierAttempt, result: TierResult | None) -> None:
        self.attempts.append(attempt)
        if result is None:
            self.degraded = True
            return
        self.items.extend(result.items)
        if result.outcome is TierOutcome.HIT:
            self.satisfied.add(attempt.paradigm)

    def result(self) -> CascadeResult:
        return CascadeResult(
            list(self.items), list(self.attempts), frozenset(self.satisfied), self.degraded
        )


class LatencyCascade:
    """Concept 5's Latency-Adaptive Fallback Cascade, steered by a routing decision.

    - decision=None is the source's cascade as drawn: CAG, MAG, RAG, first
      HIT wins. It is the ablation baseline the router has to beat.
    - A CASCADE decision tries routed paradigms cheapest-first plus RAG as
      the universal last resort, stopping once every routed paradigm has a
      HIT. A paradigm the router left out is never attempted -- that is what
      keeps a freshness query away from a stale frozen cache.
    - A PARALLEL decision (router unsure) runs every eligible tier at once.

    A timeout or exception degrades the answer instead of failing it; a
    CancelledError always propagates. Each tier must own its unit of work:
    a timed-out tier is cancelled mid-flight, and a cancelled SQLAlchemy
    query terminates its connection, so tiers sharing one session would
    break each other (see SessionScopedSemanticFactSearch).

    Tiers also share one event loop, so a tier must never do blocking CPU
    work inline: in a PARALLEL route it stalls every sibling tier's completion,
    and their timeouts fire on wall time they never got to use. Measured: a
    RAG tier embedding its query on the loop pushed CAG attempts to p50 9.8ms,
    with 15 of 60 timing out at 10ms; with the embedding off the loop, none
    did. CagTier offloads its matching to a thread, and a RAG retriever that
    embeds should share a CachingEmbeddingModel with UnifiedAnswerQuestion so
    that its embed of the already-embedded question is a lookup.

    A timed-out tier is cancelled rather than awaited, so its cleanup never
    holds up the answer; drain() waits for that cleanup, as well as for any
    background RAG completion. A RAG attempt that times out keeps running in
    the background when on_rag_findings is set ("if RAG is slow -> return
    best-effort + async update"), bounded by background_timeout and
    max_background so a hung RAG backend cannot accumulate tasks. That work
    lives in this process only -- src/workers/ now exists, but only for Celery-
    dispatched ingestion jobs, which have no notion of this cascade's own best-effort
    background RAG follow-ups -- so drain() before shutdown.
    """

    def __init__(
        self,
        tiers: Sequence[CascadeTier],
        timeouts: TierTimeouts | None = None,
        access_tracker: AccessFrequencyTracker | None = None,
        on_rag_findings: RagFindingsSink | None = None,
        clock: Callable[[], datetime] = _utc_now,
        background_timeout: float = DEFAULT_BACKGROUND_TIMEOUT,
        max_background: int = DEFAULT_MAX_BACKGROUND,
    ) -> None:
        by_paradigm = {tier.paradigm: tier for tier in tiers}
        if len(by_paradigm) != len(tiers):
            raise ValueError("configure at most one tier per paradigm")
        if Paradigm.RAG not in by_paradigm:
            raise ValueError("a RAG tier is required: it is the cascade's last resort")
        if background_timeout <= 0.0:
            raise ValueError("background_timeout must be positive")
        if max_background < 1:
            raise ValueError("max_background must be at least 1")
        self._tiers = by_paradigm
        self._timeouts = timeouts if timeouts is not None else TierTimeouts()
        self._access_tracker = access_tracker
        self._on_rag_findings = on_rag_findings
        self._clock = clock
        self._background_timeout = background_timeout
        self._max_background = max_background
        # Strong references for work nobody awaits: the event loop only keeps
        # weak ones, so an unreferenced task can be garbage-collected mid-flight.
        self._background: set[asyncio.Task[None]] = set()
        self._cancelled: set[asyncio.Task[TierResult]] = set()

    async def run(
        self, request: TierRequest, decision: RoutingDecision | None = None
    ) -> CascadeResult:
        if decision is None:
            return await self._run_unrouted(request)
        eligible = [
            p
            for p in PARADIGM_ORDER
            if p in self._tiers and (p in decision.paradigms or p is Paradigm.RAG)
        ]
        if decision.mode is RoutingMode.PARALLEL:
            return await self._run_parallel(request, eligible)
        return await self._run_routed(request, decision.paradigms, eligible)

    async def drain(self) -> None:
        """Wait for background RAG completions and for cancelled tiers' cleanup."""
        while self._background or self._cancelled:
            await asyncio.gather(*self._background, *self._cancelled, return_exceptions=True)

    async def _run_unrouted(self, request: TierRequest) -> CascadeResult:
        collector = _Collector()
        for paradigm in PARADIGM_ORDER:
            if paradigm not in self._tiers:
                continue
            attempt, result = await self._attempt(paradigm, request)
            collector.add(attempt, result)
            if result is not None and result.outcome is TierOutcome.HIT:
                break
        return collector.result()

    async def _run_routed(
        self, request: TierRequest, routed: frozenset[Paradigm], eligible: list[Paradigm]
    ) -> CascadeResult:
        collector = _Collector()
        for paradigm in eligible:
            if routed <= collector.satisfied:
                break
            attempt, result = await self._attempt(paradigm, request)
            collector.add(attempt, result)
        return collector.result()

    async def _run_parallel(
        self, request: TierRequest, eligible: list[Paradigm]
    ) -> CascadeResult:
        collector = _Collector()
        tasks = [asyncio.create_task(self._attempt(p, request)) for p in eligible]
        try:
            outcomes = await asyncio.gather(*tasks)
        except BaseException:
            # gather does not cancel the remaining children when one of them
            # raises -- a tier-raised CancelledError included -- so without this
            # their tier work would keep running with nobody waiting for it.
            for task in tasks:
                task.cancel()
            raise
        for attempt, result in outcomes:
            collector.add(attempt, result)
        return collector.result()

    async def _attempt(
        self, paradigm: Paradigm, request: TierRequest
    ) -> tuple[TierAttempt, TierResult | None]:
        started = time.perf_counter()
        task = asyncio.create_task(self._tiers[paradigm].attempt(request))
        try:
            # shield: a timeout must not cancel the task outright, because a
            # slow RAG attempt may still be worth finishing in the background.
            result = await asyncio.wait_for(
                asyncio.shield(task), self._timeouts.for_paradigm(paradigm)
            )
        except TimeoutError:
            if (
                paradigm is Paradigm.RAG
                and self._on_rag_findings is not None
                and len(self._background) < self._max_background
            ):
                self._finish_in_background(task, request, self._on_rag_findings)
            else:
                self._cancel_tier(task)
            return TierAttempt(paradigm, TierOutcome.TIMEOUT, _elapsed_ms(started)), None
        except asyncio.CancelledError:
            self._cancel_tier(task)
            raise
        except Exception:
            logger.warning(
                "cascade tier %s raised; continuing without it", paradigm.value, exc_info=True
            )
            return TierAttempt(paradigm, TierOutcome.ERROR, _elapsed_ms(started)), None
        if paradigm is Paradigm.RAG and result.outcome is TierOutcome.HIT:
            self._record_rag_access(request, result.items)
        return TierAttempt(paradigm, result.outcome, _elapsed_ms(started)), result

    def _cancel_tier(self, task: asyncio.Task[TierResult]) -> None:
        task.cancel()
        self._cancelled.add(task)
        task.add_done_callback(self._cancelled.discard)

    def _finish_in_background(
        self, task: asyncio.Task[TierResult], request: TierRequest, sink: RagFindingsSink
    ) -> None:
        async def finish() -> None:
            try:
                # wait_for cancels the task if it outlives the background deadline.
                result = await asyncio.wait_for(task, self._background_timeout)
            except TimeoutError:
                # Either the background deadline passed or the tier timed out on
                # its own; the finding is dropped either way.
                logger.warning(
                    "background RAG attempt timed out (background limit %.1fs); dropped",
                    self._background_timeout,
                )
                return
            except Exception:
                logger.warning("background RAG attempt failed after timing out", exc_info=True)
                return
            if result.outcome is not TierOutcome.HIT:
                return
            self._record_rag_access(request, result.items)
            try:
                await sink(request, result.items)
            except Exception:
                logger.warning("on_rag_findings raised for a background result", exc_info=True)

        background = asyncio.create_task(finish())
        self._background.add(background)
        background.add_done_callback(self._background.discard)

    def _record_rag_access(self, request: TierRequest, items: list[ContextItem]) -> None:
        # Pattern 1's "if result was frequent -> flag for CAG pre-loading":
        # feeds the tracker WarmCache/TieringPolicy already read. One access
        # per document, however many of its chunks came back. Analytics only
        # inform later pre-loading, so a failure here is logged and never costs
        # the answer that produced it.
        if self._access_tracker is None:
            return
        try:
            now = self._clock()
            source_ids = [item.source_id for item in items]
            document_ids: dict[uuid.UUID, None] = dict.fromkeys(
                source_id for source_id in source_ids if source_id is not None
            )
            for document_id in document_ids:
                self._access_tracker.record_access(request.tenant_id, document_id, now)
        except Exception:
            logger.warning("recording RAG access for CAG pre-loading failed", exc_info=True)
