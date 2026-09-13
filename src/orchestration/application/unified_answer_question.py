import math
import time
import uuid
from dataclasses import dataclass

from src.orchestration.application.assemble_context import assemble_context
from src.orchestration.application.latency_cascade import LatencyCascade
from src.orchestration.domain.budget_allocator import DEFAULT_SHARES, allocate
from src.orchestration.domain.entities import (
    BudgetAllocation,
    BudgetShares,
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
)
from src.orchestration.domain.ports import QueryClassifier, SessionBudgetRecorder
from src.rag.domain.ports import ChatModel, EmbeddingModel
from src.shared.tokenization import count_tokens

DEFAULT_CONTEXT_TOKENS = 128_000


@dataclass(frozen=True)
class StageTimings:
    embed_ms: float
    route_ms: float
    cascade_ms: float
    assemble_ms: float
    generate_ms: float


@dataclass(frozen=True)
class UnifiedAnswer:
    answer: str
    sources: list[ContextItem]
    decision: RoutingDecision | None
    attempts: list[TierAttempt]
    allocation: BudgetAllocation
    dropped: dict[Paradigm, int]
    degraded: bool
    timings: StageTimings


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000


class UnifiedAnswerQuestion:
    """Section 3.4 Pattern 1, "The Smart Router": route, cascade, budget, answer.

    With classifier=None the cascade runs unrouted, exactly as Concept 5
    draws it -- the ablation baseline the router is measured against.
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
        budget_recorder: SessionBudgetRecorder | None = None,
    ) -> None:
        if total_context_tokens <= 0:
            raise ValueError("total_context_tokens must be positive")
        self._embedder = embedding_model
        self._classifier = classifier
        self._cascade = cascade
        self._chat_model = chat_model
        self._total = total_context_tokens
        self._shares = shares
        self._select_threshold = select_threshold
        self._uncertainty_margin = uncertainty_margin
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
        embedding = self._embedder.embed(question)
        embedded = time.perf_counter()

        decision: RoutingDecision | None = None
        if self._classifier is not None:
            scores = await self._classifier.score(question, embedding)
            decision = decide(scores, self._select_threshold, self._uncertainty_margin)
        routed = time.perf_counter()

        request = TierRequest(tenant_id, user_id, session_id, question, embedding)
        cascade_result = await self._cascade.run(request, decision)
        cascaded = time.perf_counter()

        allocation = allocate(self._total, cascade_result.contributing, self._shares)
        assembled = assemble_context(cascade_result.items, allocation)
        assembled_at = time.perf_counter()

        answer = await self._chat_model.generate(question=question, context=assembled.text)
        generated = time.perf_counter()

        if self._budget_recorder is not None:
            await self._budget_recorder.record(
                tenant_id, session_id, allocation, cascade_result.contributing
            )

        return UnifiedAnswer(
            answer=answer,
            sources=assembled.included,
            decision=decision,
            attempts=cascade_result.attempts,
            allocation=allocation,
            dropped=assembled.dropped,
            degraded=cascade_result.degraded,
            timings=StageTimings(
                embed_ms=_ms(started, embedded),
                route_ms=_ms(embedded, routed),
                cascade_ms=_ms(routed, cascaded),
                assemble_ms=_ms(cascaded, assembled_at),
                generate_ms=_ms(assembled_at, generated),
            ),
        )
