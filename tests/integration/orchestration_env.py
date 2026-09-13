"""Shared real-infrastructure environment for the orchestration meta-layer's
integration tests: Postgres with RLS for MAG, Qdrant for RAG, real MiniLM
embeddings, and a real distilgpt2 HFFrozenCache for CAG that holds a
SUPERSEDED copy of the return policy while Qdrant holds the current one.
Not collected by pytest (no test_ prefix)."""
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.identity.infrastructure.db import set_tenant_context
from src.mag.application.queries.find_semantic_facts import FindSemanticFacts
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
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.entities import Chunk
from src.rag.domain.ports import ChatModel, EmbeddingModel
from src.rag.infrastructure.qdrant_vector_store import QdrantVectorStore

VALID_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaHZhbHVl"

POLICY_V1 = (
    "Our return policy allows customers to return unopened items within thirty days "
    "of purchase for a full refund."
)
POLICY_V2 = (
    "Our return policy allows customers to return unopened items within forty-five days "
    "of purchase for a full refund. The window changed from thirty days today."
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

# Measured with real MiniLM on 2026-09-13 (plan Task 9, Step 1):
#   CAG must hit:  freshness query vs POLICY_V1 0.5771, policy query vs POLICY_V1 0.7700
#   CAG must miss: shipping query vs POLICY_V1 0.2833
#   MAG must hit:  preference query vs FACT_VALUE 0.5202
#   MAG must miss: policy query vs FACT_VALUE 0.3290
# Hit = midpoint of the lowest must-hit and highest must-miss score, floored to
# two decimals. Partial = midpoint of the must-miss score and the hit threshold.
# That is a deliberate change from the plan's "must-miss minus 0.05", which
# would have turned a clearly unrelated query into a PARTIAL match and put the
# stale policy into its context.
CAG_HIT = 0.43
CAG_PARTIAL = 0.35
MAG_HIT = 0.42
MAG_PARTIAL = 0.37

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


@dataclass
class OrchestrationEnv:
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    session_id: uuid.UUID
    policy_document_id: uuid.UUID
    search: SearchDocuments
    warmed: CacheWarmedRetrieve
    repository: PostgresSemanticMemoryRepository

    def tiers(self) -> list[CascadeTier]:
        return [
            CagTier(self.warmed, hit_threshold=CAG_HIT, partial_threshold=CAG_PARTIAL),
            MagTier(
                FindSemanticFacts(self.repository),
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
    search = SearchDocuments(embedding_model, vector_store)

    cache = HFFrozenCache(tokenizer=distilgpt2_tokenizer, model=distilgpt2_model)
    cache.preload(tenant_id, policy_id, POLICY_V1)
    warmed = CacheWarmedRetrieve(embedding_model, cache, search, similarity_threshold=CAG_HIT)
    warmed.note_warmed(tenant_id, policy_id, POLICY_V1)

    # set_tenant_context is transaction-local, and PostgresSemanticMemoryRepository
    # (like every MAG repository) relies on its caller having set it -- so the
    # MAG tier's reads in these tests run inside this same transaction.
    await set_tenant_context(db_session, tenant_id)
    repository = PostgresSemanticMemoryRepository(db_session)
    await repository.save(
        SemanticMemory(
            id=uuid.uuid4(),
            user_id=user_id,
            fact_key=FACT_KEY,
            fact_value=FACT_VALUE,
            embedding=embedding_model.embed(FACT_VALUE),
        ),
        tenant_id,
    )
    return OrchestrationEnv(tenant_id, user_id, session_id, policy_id, search, warmed, repository)
