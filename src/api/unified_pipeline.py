from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.identity.domain.ports import ChatSessionRepository
from src.orchestration.application.answer_in_session import AnswerInSession
from src.orchestration.application.cascade_tiers import MagTier, RagTier
from src.orchestration.application.latency_cascade import LatencyCascade, TierTimeouts
from src.orchestration.application.unified_answer_question import UnifiedAnswerQuestion
from src.orchestration.infrastructure.postgres_session_budget_recorder import (
    PostgresSessionBudgetRecorder,
)
from src.orchestration.infrastructure.prototype_query_classifier import (
    PrototypeQueryClassifier,
)
from src.orchestration.infrastructure.session_scoped_semantic_fact_search import (
    SessionScopedSemanticFactSearch,
)
from src.rag.application.search_documents import SearchDocuments
from src.rag.domain.ports import ChatModel, EmbeddingModel, VectorStore
from src.rag.infrastructure.caching_embedding_model import CachingEmbeddingModel

# Measured on the orchestration integration corpus, in
# evaluation/scenarios/orchestration_meta_layer_thresholds.py. src/ can't import
# evaluation/, so the serving values live here, and a unit test keeps them equal.
MAG_HIT = 0.42
MAG_PARTIAL = 0.37
RAG_TOP_K = 5


@dataclass(frozen=True)
class UnifiedPipeline:
    answer_in_session: AnswerInSession
    cascade: LatencyCascade


def build_unified_pipeline(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    sessions: ChatSessionRepository,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    chat_model: ChatModel,
) -> UnifiedPipeline:
    """The session-scoped unified answer path, as the API serves it.

    MAG and RAG tiers only. A CAG tier needs a warmed frozen cache in this process and a
    worker to warm it, and neither exists yet, so the cascade answers a CAG-only route
    from RAG (spec decision 5). One CachingEmbeddingModel serves the question's
    embedding, the router, and SearchDocuments, so the RAG tier's embedding of the
    question is a lookup.
    """
    embedder = CachingEmbeddingModel(embedding_model)
    cascade = LatencyCascade(
        [
            MagTier(
                SessionScopedSemanticFactSearch(sessionmaker),
                hit_threshold=MAG_HIT,
                partial_threshold=MAG_PARTIAL,
            ),
            RagTier(SearchDocuments(embedder, vector_store), top_k=RAG_TOP_K),
        ],
        TierTimeouts(),
    )
    answerer = UnifiedAnswerQuestion(
        embedder,
        PrototypeQueryClassifier(embedder),
        cascade,
        chat_model,
        budget_recorder=PostgresSessionBudgetRecorder(sessionmaker),
    )
    return UnifiedPipeline(AnswerInSession(sessions, answerer), cascade)
