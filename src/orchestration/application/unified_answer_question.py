import asyncio
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from src.orchestration.application.assemble_context import AssembledContext, assemble_context
from src.orchestration.application.latency_cascade import LatencyCascade
from src.orchestration.domain.budget_allocator import DEFAULT_SHARES, allocate
from src.orchestration.domain.entities import (
    BudgetAllocation,
    BudgetShares,
    CascadeResult,
    ContextItem,
    Paradigm,
    RoutingDecision,
    TierAttempt,
    TierRequest,
)
from src.orchestration.domain.errors import QueryExceedsBudget
from src.orchestration.domain.paradigm_router import (
    DEFAULT_SELECT_THRESHOLD,
    DEFAULT_UNCERTAINTY_MARGIN,
    decide,
    fallback_decision,
)
from src.orchestration.domain.ports import QueryClassifier, SessionBudgetRecorder
from src.rag.domain.ports import ChatModel, EmbeddingModel
from src.shared.tokenization import count_tokens

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKENS = 128_000
# Seconds. The router comparison (evaluation/reports/orchestration-meta-layer-router.md)
# measured the local qwen3.5 classifier at p50 18.9s and p95 60.0s, against 0.1ms for
# the lexical classifier and 2.4ms for the prototype one. At this default the LLM
# classifier therefore falls back to routing every tier on at least half of all
# queries, going by its median alone. That is deliberate:
# waiting tens of seconds to pick between tiers budgeted at 10ms-2s defeats the
# cascade, so a request-path classifier has to be a millisecond-scale one.
DEFAULT_CLASSIFIER_TIMEOUT = 2.0

RoutingFallback = Literal["classifier_timeout", "classifier_error"]


@dataclass(frozen=True)
class StageTimings:
    embed_ms: float
    route_ms: float
    cascade_ms: float
    assemble_ms: float
    record_ms: float  # 0.0 when no SessionBudgetRecorder is configured
    generate_ms: float


@dataclass(frozen=True)
class UnifiedAnswer:
    answer: str
    sources: list[ContextItem]
    decision: RoutingDecision | None
    # None when the classifier decided the route (or none was configured).
    routing_fallback: RoutingFallback | None
    attempts: list[TierAttempt]
    allocation: BudgetAllocation
    dropped: dict[Paradigm, int]
    degraded: bool
    timings: StageTimings


@dataclass(frozen=True)
class RoutingStageEvent:
    """The router's decision, once made -- the streaming counterpart of
    UnifiedAnswer.decision/routing_fallback."""

    decision: RoutingDecision | None
    routing_fallback: RoutingFallback | None


@dataclass(frozen=True)
class RetrievalStageEvent:
    """The cascade's result, once every attempted tier has answered or given up --
    the streaming counterpart of UnifiedAnswer.attempts/degraded."""

    attempts: list[TierAttempt]
    degraded: bool


@dataclass(frozen=True)
class BudgetStageEvent:
    """Context assembly's result -- the streaming counterpart of
    UnifiedAnswer.sources/dropped. Never carries the raw BudgetAllocation numbers,
    for the same reason answer_response() withholds them from the JSON endpoint
    (spec decision 8)."""

    sources: list[ContextItem]
    dropped: dict[Paradigm, int]


@dataclass(frozen=True)
class AnswerChunkEvent:
    """One text delta from the chat model's own stream()."""

    text: str


@dataclass(frozen=True)
class AnswerCompleteEvent:
    """The stream's normal terminal event; carries no data."""


@dataclass(frozen=True)
class AnswerErrorEvent:
    """The stream's terminal event on failure. `message` is always a generic,
    safe-to-show string -- the real exception is logged, never surfaced here."""

    message: str


UnifiedAnswerEvent = (
    RoutingStageEvent
    | RetrievalStageEvent
    | BudgetStageEvent
    | AnswerChunkEvent
    | AnswerCompleteEvent
    | AnswerErrorEvent
)


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000


class UnifiedAnswerQuestion:
    """Section 3.4 Pattern 1, "The Smart Router": route, cascade, budget, answer.

    With classifier=None the cascade runs unrouted, exactly as Concept 5
    draws it -- the ablation baseline the router is measured against. A
    classifier that times out or raises doesn't fail the request: the query
    is routed with fallback_decision(), every tier in parallel.

    The budget is recorded after the cascade and before generation: a
    session that doesn't exist, or isn't this user's, fails the request
    before an answer is paid for, though the retrieval work is already spent.

    stream() answers the same question as execute(), sharing the same
    _decide_route/_do_budget/_record_budget helpers, but reports progress as
    domain events instead of returning one final UnifiedAnswer -- see
    docs/superpowers/specs/2026-09-17-streaming-answers-design.md for why its
    query-budget check has to stay eager (in stream() itself, not in the lazy
    generator stream() hands back) and why a failure after streaming has
    already started becomes an AnswerErrorEvent rather than a raised exception.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        classifier: QueryClassifier | None,
        cascade: LatencyCascade,
        chat_model: ChatModel,
        *,
        total_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        shares: BudgetShares = DEFAULT_SHARES,
        select_threshold: float = DEFAULT_SELECT_THRESHOLD,
        uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
        classifier_timeout: float = DEFAULT_CLASSIFIER_TIMEOUT,
        budget_recorder: SessionBudgetRecorder | None = None,
    ) -> None:
        if total_context_tokens <= 0:
            raise ValueError("total_context_tokens must be positive")
        if classifier_timeout <= 0.0:
            raise ValueError("classifier_timeout must be positive")
        self._embedder = embedding_model
        self._classifier = classifier
        self._cascade = cascade
        self._chat_model = chat_model
        self._total = total_context_tokens
        self._shares = shares
        self._select_threshold = select_threshold
        self._uncertainty_margin = uncertainty_margin
        self._classifier_timeout = classifier_timeout
        self._budget_recorder = budget_recorder

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> UnifiedAnswer:
        # Same floor allocate() applies, so this check and the allocation
        # below can never disagree about the Query slice's size.
        query_slice = math.floor(self._total * self._shares.query)
        query_tokens = count_tokens(question)
        if query_tokens > query_slice:
            raise QueryExceedsBudget(query_tokens, query_slice)

        started = time.perf_counter()
        # Off the event loop: milliseconds of CPU that would otherwise stall
        # every tier and every other request sharing the loop.
        embedding = await asyncio.to_thread(self._embedder.embed, question)
        embedded = time.perf_counter()

        decision, routing_fallback = await self._decide_route(question, embedding)
        routed = time.perf_counter()

        request = TierRequest(tenant_id, user_id, session_id, question, embedding)
        cascade_result = await self._cascade.run(request, decision)
        cascaded = time.perf_counter()

        allocation, assembled = self._do_budget(cascade_result)
        assembled_at = time.perf_counter()

        await self._record_budget(
            tenant_id, user_id, session_id, allocation, cascade_result.contributing
        )
        recorded = time.perf_counter()

        answer = await self._chat_model.generate(question=question, context=assembled.text)
        generated = time.perf_counter()

        return UnifiedAnswer(
            answer=answer,
            sources=assembled.included,
            decision=decision,
            routing_fallback=routing_fallback,
            attempts=cascade_result.attempts,
            allocation=allocation,
            dropped=assembled.dropped,
            degraded=cascade_result.degraded,
            timings=StageTimings(
                embed_ms=_ms(started, embedded),
                route_ms=_ms(embedded, routed),
                cascade_ms=_ms(routed, cascaded),
                assemble_ms=_ms(cascaded, assembled_at),
                record_ms=_ms(assembled_at, recorded) if self._budget_recorder else 0.0,
                generate_ms=_ms(recorded, generated),
            ),
        )

    async def stream(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> AsyncIterator[UnifiedAnswerEvent]:
        query_slice = math.floor(self._total * self._shares.query)
        query_tokens = count_tokens(question)
        if query_tokens > query_slice:
            raise QueryExceedsBudget(query_tokens, query_slice)
        return self._stream_events(tenant_id, user_id, session_id, question)

    async def _stream_events(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID, question: str
    ) -> AsyncIterator[UnifiedAnswerEvent]:
        try:
            embedding = await asyncio.to_thread(self._embedder.embed, question)
            decision, routing_fallback = await self._decide_route(question, embedding)
            yield RoutingStageEvent(decision, routing_fallback)

            request = TierRequest(tenant_id, user_id, session_id, question, embedding)
            cascade_result = await self._cascade.run(request, decision)
            yield RetrievalStageEvent(cascade_result.attempts, cascade_result.degraded)

            allocation, assembled = self._do_budget(cascade_result)
            yield BudgetStageEvent(assembled.included, assembled.dropped)

            await self._record_budget(
                tenant_id, user_id, session_id, allocation, cascade_result.contributing
            )

            async for delta in self._chat_model.stream(question=question, context=assembled.text):
                yield AnswerChunkEvent(delta)
            yield AnswerCompleteEvent()
        except Exception:
            logger.exception("answer streaming failed")
            yield AnswerErrorEvent("answer generation failed")

    async def _decide_route(
        self, question: str, embedding: list[float]
    ) -> tuple[RoutingDecision | None, RoutingFallback | None]:
        if self._classifier is None:
            return None, None
        return await self._route(self._classifier, question, embedding)

    def _do_budget(
        self, cascade_result: CascadeResult
    ) -> tuple[BudgetAllocation, AssembledContext]:
        allocation = allocate(self._total, cascade_result.contributing, self._shares)
        return allocation, assemble_context(cascade_result.items, allocation)

    async def _record_budget(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        allocation: BudgetAllocation,
        contributing: frozenset[Paradigm],
    ) -> None:
        if self._budget_recorder is not None:
            await self._budget_recorder.record(
                tenant_id, user_id, session_id, allocation, contributing
            )

    async def _route(
        self, classifier: QueryClassifier, question: str, embedding: list[float]
    ) -> tuple[RoutingDecision, RoutingFallback | None]:
        try:
            scores = await asyncio.wait_for(
                classifier.score(question, embedding), self._classifier_timeout
            )
            return decide(scores, self._select_threshold, self._uncertainty_margin), None
        except TimeoutError:
            # The classifier_timeout limit, or a timeout the classifier raised itself.
            logger.warning(
                "query classifier timed out (limit %.2fs); routing every tier in parallel",
                self._classifier_timeout,
            )
            return fallback_decision(), "classifier_timeout"
        except Exception:
            logger.warning("query classifier failed; routing every tier in parallel", exc_info=True)
            return fallback_decision(), "classifier_error"
