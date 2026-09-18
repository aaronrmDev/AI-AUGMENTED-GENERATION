"""Shared real-infrastructure environment for the orchestration meta-layer's
integration tests: Postgres with RLS for MAG, Qdrant for RAG, real MiniLM
embeddings, and a real distilgpt2 HFFrozenCache for CAG that holds a
SUPERSEDED copy of the return policy while Qdrant holds the current one.
Not collected by pytest (no test_ prefix)."""
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evaluation.scenarios.orchestration_meta_layer_thresholds import (
    CAG_HIT as CAG_HIT,
)
from evaluation.scenarios.orchestration_meta_layer_thresholds import (
    CAG_PARTIAL as CAG_PARTIAL,
)
from evaluation.scenarios.orchestration_meta_layer_thresholds import (
    MAG_HIT as MAG_HIT,
)
from evaluation.scenarios.orchestration_meta_layer_thresholds import (
    MAG_PARTIAL as MAG_PARTIAL,
)
from src.identity.infrastructure.db import get_sessionmaker, set_tenant_context
from src.mag.domain.entities import SemanticMemory
from src.mag.infrastructure.postgres_semantic_memory_repository import (
    PostgresSemanticMemoryRepository,
)
from src.orchestration.application.cache_warmed_retrieve import CacheWarmedRetrieve
from src.orchestration.application.cascade_tiers import CagTier, MagTier, RagTier
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.domain.entities import Paradigm, TierRequest
from src.orchestration.domain.ports import CascadeTier, QueryClassifier
from src.orchestration.infrastructure.hf_frozen_cache import HFFrozenCache
from src.orchestration.infrastructure.session_scoped_semantic_fact_search import (
    SessionScopedSemanticFactSearch,
)
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import Chunk
from src.rag.domain.ports import ChatModel, EmbeddingModel
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"

POLICY_V1 = (
    "Our return policy allows customers to return unopened items within thirty days "
    "of purchase for a full refund."
)
# The current policy never mentions the superseded value. An earlier version
# ended "The window changed from thirty days today.", and the live model
# answered a freshness question by quoting exactly that sentence -- correct
# context, an answer naming only the old number.
POLICY_V2 = (
    "Our return policy allows customers to return unopened items within forty-five days "
    "of purchase for a full refund. The return window was extended to forty-five days today."
)
SHIPPING = (
    "Standard shipping takes five to seven business days. Expedited shipping arrives "
    "within two business days."
)
FACT_KEY = "preferred_shipping_speed"
FACT_VALUE = "The user prefers expedited two-day shipping on every order."

FRESHNESS_QUERY = "What changed in the return policy today?"
POLICY_QUERY = "What is the return policy for unopened items?"
PREFERENCE_QUERY = "What shipping speed did I say I prefer?"

# Generous on purpose: these tests check routing correctness against real
# stores. Latency against Concept 5's budgets is the evaluation runner's
# job, where first-call warm-up can be measured separately.
CORRECTNESS_TIMEOUTS = TierTimeouts(cag=5.0, mag=5.0, rag=10.0)


class FixedScoresClassifier(QueryClassifier):
    def __init__(self, scores: dict[Paradigm, float]) -> None:
        self._scores = scores

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        return dict(self._scores)


class ContextEchoChatModel(ChatModel):
    async def generate(self, question: str, context: str) -> str:
        return context

    async def complete(self, prompt: str) -> str:
        return prompt

    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        # Two chunks (never one), unless there's nothing to say -- a real
        # integration test reassembling multiple chunks exercises real
        # reassembly, not a degenerate single-chunk case.
        if not context:
            return
        midpoint = max(1, len(context) // 2)
        yield context[:midpoint]
        yield context[midpoint:]


@dataclass
class OrchestrationEnv:
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    session_id: uuid.UUID
    policy_document_id: uuid.UUID
    search: SearchDocuments
    warmed: CacheWarmedRetrieve
    # The tiers' and the recorder's own units of work come from here, never
    # from the test's db_session.
    sessionmaker: async_sessionmaker[AsyncSession]
    # The one embedder a UnifiedAnswerQuestion built on this env should take:
    # the RAG retriever shares it, so re-embedding the question inside the
    # RAG tier is a lookup instead of CPU work on the event loop.
    embedder: CachingEmbeddingModel

    def tiers(self) -> list[CascadeTier]:
        return [
            CagTier(self.warmed, hit_threshold=CAG_HIT, partial_threshold=CAG_PARTIAL),
            MagTier(
                SessionScopedSemanticFactSearch(self.sessionmaker),
                hit_threshold=MAG_HIT,
                partial_threshold=MAG_PARTIAL,
            ),
            RagTier(self.search, top_k=2),
        ]

    def cascade(self, timeouts: TierTimeouts = CORRECTNESS_TIMEOUTS) -> LatencyCascade:
        return LatencyCascade(self.tiers(), timeouts)

    def request(
        self, query: str, embedding_model: EmbeddingModel, user_id: uuid.UUID | None = None
    ) -> TierRequest:
        return TierRequest(
            self.tenant_id,
            user_id or self.user_id,
            self.session_id,
            query,
            embedding_model.embed(query),
        )


async def create_user_and_session(
    db_session: AsyncSession, tenant_id: uuid.UUID
) -> tuple[uuid.UUID, uuid.UUID]:
    await set_tenant_context(db_session, tenant_id)
    now = datetime.now(UTC)
    user_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO users (id, email, hashed_password, tenant_id, created_at, updated_at) "
            "VALUES (:id, :email, :hashed_password, :tenant_id, :created_at, :updated_at)"
        ),
        {
            "id": user_id, "email": f"{user_id}@example.com", "hashed_password": VALID_HASH,
            "tenant_id": tenant_id, "created_at": now, "updated_at": now,
        },
    )
    session_id = uuid.uuid4()
    await db_session.execute(
        text(
            "INSERT INTO sessions (id, user_id, tenant_id, title) "
            "VALUES (:id, :user_id, :tenant_id, :title)"
        ),
        {"id": session_id, "user_id": user_id, "tenant_id": tenant_id, "title": "t"},
    )
    await db_session.commit()
    return user_id, session_id


async def build_env(
    db_session: AsyncSession,
    qdrant_url: str,
    embedding_model: EmbeddingModel,
    distilgpt2_tokenizer: Any,
    distilgpt2_model: Any,
) -> OrchestrationEnv:
    tenant_id = uuid.uuid4()
    user_id, session_id = await create_user_and_session(db_session, tenant_id)

    vector_store = QdrantVectorStore(qdrant_url)
    await vector_store.ensure_collection()
    policy_id = uuid.uuid4()
    for document_id, content in ((policy_id, POLICY_V2), (uuid.uuid4(), SHIPPING)):
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=document_id,
            content=content,
            embedding=embedding_model.embed(content),
        )
        await vector_store.upsert(chunk, tenant_id)
    embedder = CachingEmbeddingModel(embedding_model)
    search = SearchDocuments(embedder, vector_store)

    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    cache.preload(tenant_id, policy_id, POLICY_V1)
    warmed = CacheWarmedRetrieve(embedding_model, cache, search, similarity_threshold=CAG_HIT)
    warmed.note_warmed(tenant_id, policy_id, POLICY_V1)

    await set_tenant_context(db_session, tenant_id)
    await PostgresSemanticMemoryRepository(db_session).save(
        SemanticMemory(
            id=uuid.uuid4(),
            user_id=user_id,
            fact_key=FACT_KEY,
            fact_value=FACT_VALUE,
            embedding=embedding_model.embed(FACT_VALUE),
        ),
        tenant_id,
    )
    # Committed, because the MAG tier reads through its own sessions and can
    # only see committed rows.
    await db_session.commit()
    return OrchestrationEnv(
        tenant_id,
        user_id,
        session_id,
        policy_id,
        search,
        warmed,
        get_sessionmaker(db_session.bind),
        embedder,
    )
