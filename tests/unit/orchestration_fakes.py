import asyncio
import hashlib
import uuid

from src.orchestration.domain.entities import (
    BudgetAllocation,
    CacheHit,
    Paradigm,
    TierOutcome,
    TierRequest,
    TierResult,
    WarmEntry,
)
from src.orchestration.domain.ports import (
    CascadeTier,
    FrozenCache,
    QueryClassifier,
    SessionBudgetRecorder,
    WarmStore,
)
from src.orchestration.domain.sync_mixer import content_hash
from src.rag.domain.ports import EmbeddingModel


class FakeFrozenCache(FrozenCache):
    def __init__(self) -> None:
        self._entries: dict[tuple[uuid.UUID, uuid.UUID], str] = {}
        self.preload_calls: list[tuple[uuid.UUID, uuid.UUID, str]] = []
        self.evict_calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def preload(self, tenant_id: uuid.UUID, document_id: uuid.UUID, content: str) -> None:
        self.preload_calls.append((tenant_id, document_id, content))
        self._entries[(tenant_id, document_id)] = content

    def lookup(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> CacheHit | None:
        content = self._entries.get((tenant_id, document_id))
        if content is None:
            return None
        return CacheHit(content_hash=content_hash(content), kv_cache=None)

    def evict(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        self.evict_calls.append((tenant_id, document_id))
        self._entries.pop((tenant_id, document_id), None)

    def contains(self, tenant_id: uuid.UUID, document_id: uuid.UUID) -> bool:
        return (tenant_id, document_id) in self._entries


class FakeWarmStore(WarmStore):
    def __init__(self) -> None:
        self._entries: dict[tuple[uuid.UUID, uuid.UUID, uuid.UUID], str] = {}
        self.promote_calls: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID, str]] = []
        self.demote_calls: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = []

    async def promote(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID, content: str
    ) -> None:
        self.promote_calls.append((tenant_id, user_id, document_id, content))
        self._entries[(tenant_id, user_id, document_id)] = content

    async def lookup(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> WarmEntry | None:
        content = self._entries.get((tenant_id, user_id, document_id))
        if content is None:
            return None
        return WarmEntry(content_hash=content_hash(content), content=content)

    async def demote(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> None:
        self.demote_calls.append((tenant_id, user_id, document_id))
        self._entries.pop((tenant_id, user_id, document_id), None)

    async def contains(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> bool:
        return (tenant_id, user_id, document_id) in self._entries


class FakeBagOfWordsEmbeddingModel(EmbeddingModel):
    """Deterministic, cheap stand-in that actually varies by direction
    instead of collapsing to one (unlike rag_fakes.FakeEmbeddingModel,
    whose length-derived vectors are always parallel and would make every
    cosine similarity comparison trivially 1.0) -- needed here because
    CacheWarmedRetrieve's hit/miss decision genuinely depends on real
    directional similarity between two embeddings, not just their
    presence. A plain hashed bag-of-words vector, normalized: shared words
    push cosine similarity up, disjoint words push it down, deterministic
    across runs (hashlib, not Python's randomized built-in hash()).
    """

    _DIM = 64

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._DIM
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._DIM
            vector[index] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


class FakeCascadeTier(CascadeTier):
    def __init__(
        self,
        paradigm: Paradigm,
        result: TierResult | None = None,
        delay_seconds: float = 0.0,
        error: BaseException | None = None,
    ) -> None:
        self._paradigm = paradigm
        self._result = result if result is not None else TierResult(TierOutcome.MISS)
        self._delay = delay_seconds
        self._error = error
        self.requests: list[TierRequest] = []
        self.cancelled = False

    @property
    def paradigm(self) -> Paradigm:
        return self._paradigm

    async def attempt(self, request: TierRequest) -> TierResult:
        self.requests.append(request)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self._error is not None:
            raise self._error
        return self._result


class FakeQueryClassifier(QueryClassifier):
    def __init__(
        self,
        scores: dict[Paradigm, float],
        delay_seconds: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._scores = scores
        self._delay = delay_seconds
        self._error = error
        self.calls: list[tuple[str, list[float]]] = []

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        self.calls.append((query, query_embedding))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return dict(self._scores)


class FakeSessionBudgetRecorder(SessionBudgetRecorder):
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.records: list[
            tuple[uuid.UUID, uuid.UUID, uuid.UUID, BudgetAllocation, frozenset[Paradigm]]
        ] = []

    async def record(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        allocation: BudgetAllocation,
        contributing: frozenset[Paradigm],
    ) -> None:
        if self._error is not None:
            raise self._error
        self.records.append((tenant_id, user_id, session_id, allocation, contributing))
